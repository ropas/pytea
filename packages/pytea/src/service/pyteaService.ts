/*
 * pyteaService.ts
 * Copyright (c) Seoul National University
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo (starvessel@naver.com)
 *
 * Main class of PyTea analyzer.
 * Managing imported or will be imported scripts, parsed statements and lsp services.
 */
import chalk from 'chalk';
import * as fs from 'fs';
import * as path from 'path';
import { performance } from 'perf_hooks';

import { getFileInfo } from 'pyright-internal/analyzer/analyzerNodeInfo';
import { AnalyzerService } from 'pyright-internal/analyzer/service';
import { ConsoleInterface, StandardConsole } from 'pyright-internal/common/console';
import { convertOffsetToPosition } from 'pyright-internal/common/positionUtils';
import { Range } from 'pyright-internal/common/textRange';
import { ParseNodeType } from 'pyright-internal/parser/parseNodes';

import { fetchAddr } from '../backend/backUtils';
import { Context, ContextSet } from '../backend/context';
import { RandomGen } from '../backend/randomGen';
import { CodeSource, ShContFlag, ShValue, SVAddr, SVObject, SVString, SVType } from '../backend/sharpValues';
import { TorchBackend } from '../backend/torchBackend';
import { IRWriter } from '../frontend/IRReaderWriter';
import { ThStmt } from '../frontend/torchStatements';
import { FilePathStore } from './executionPaths';
import { defaultOptions, PyCmdArgs, PyteaOptions } from './pyteaOptions';
import {
    CustomObjectPrinter,
    formatCodeSource,
    getStmtsFromDir,
    postJsonRpc,
    PyZ3RPCResult,
    PyZ3RPCResultType,
    reducedToString,
} from './pyteaUtils';

let _globalService: PyteaService | undefined;

export class PyteaService {
    private _options?: PyteaOptions;
    private _service?: AnalyzerService;

    private _console: ConsoleInterface;

    private _pathStore: FilePathStore;
    private _libStmt: Map<string, [string, ThStmt]>;
    private _projectStmt?: Map<string, [string, ThStmt]>;

    private _mainStmt?: ThStmt;
    private _builtinSet?: ContextSet<SVAddr>;

    private _timeLog: [string, number][];
    private _currTime: number;
    private _requestId: number;
    private _randomGen: Map<string, RandomGen>;

    constructor(service?: AnalyzerService, options?: PyteaOptions, console?: ConsoleInterface, setDefault?: boolean) {
        if (setDefault) _globalService = this;

        this._console = console || new StandardConsole();

        this._timeLog = [];
        this._currTime = performance.now();

        this._options = options ?? defaultOptions;
        this._service = service;

        this._libStmt = new Map();
        this._pathStore = new FilePathStore();
        this._requestId = Math.floor(Math.random() * 10000000);
        this._randomGen = new Map();
    }

    get options(): PyteaOptions | undefined {
        return this._options;
    }

    static getGlobalService(): PyteaService | undefined {
        return _globalService;
    }

    static setGlobalService(service: PyteaService): void {
        _globalService = service;
    }

    static ignoreAssert(): boolean {
        const options = _globalService?.options;
        return options ? options.ignoreAssert : true;
    }

    static shouldCheckImmediate(): boolean {
        const options = _globalService?.options;
        return options ? options.immediateConstraintCheck : true;
    }

    static isBoxAllowed(): boolean {
        const options = _globalService?.options;
        return options ? options.boxDataLoader : true;
    }

    static getCmdArgs(): PyCmdArgs {
        const options = _globalService?.options;
        return options ? options.pythonCmdArgs : {};
    }

    static getSubcommand(): string {
        const options = _globalService?.options;
        return options ? options.pythonSubcommand : '';
    }

    static getVariableRange(): { [varName: string]: null | number | [number | null, number | null] } {
        const options = _globalService?.options;
        return options ? options.variableRange : {};
    }

    static getVariableSeedRng(varName: string): RandomGen | undefined {
        return _globalService?._randomGen.get(varName);
    }

    static log(...message: any[]): void {
        _globalService?._console.info(message.map((x) => `${x}`).join(' '));
    }

    setPyrightAnalyzerService(service: AnalyzerService) {
        if (this._service !== service) {
            this._service?.dispose();
        }

        this._service = service;
    }

    setOptions(options: PyteaOptions) {
        this._options = options;
    }

    getPathStore(): FilePathStore {
        return this._pathStore;
    }

    // check library or entry file is fully loaded.
    validate(): boolean {
        let valid = true;

        if (!this._service) {
            this._console.error('Pyright service is not set.');
        }

        if (!this._options) {
            this._console.error('PyTea service option is not set. Please check pyteaconfig.json.');
        }

        if (!this._options?.entryPath) {
            this._console.error('Python entry point is not set.');
            valid = false;
        }

        if (!this._options?.extractIR && !this._options?.pyteaLibPath) {
            this._console.error('Invalid PyTea library path. Please check library path correctly.');
            valid = false;
        }

        return valid;
    }

    // translate Python scripts starting from entryPath and every Pytea library script
    // return error message (string) or undefined
    translateAll(entryPath: string): string | undefined {
        let errMsg = this.translatePyteaLib();
        if (errMsg) return errMsg;
        errMsg = this.translateMainEntry(entryPath);
        if (errMsg) return errMsg;
    }

    // return error message (string) or undefined
    translatePyteaLib(): string | undefined {
        if (!this._service) {
            return `pyright service is not set.`;
        }

        this._clearTimeLog();

        // translate and cache pytea library implementations
        if (
            this._service &&
            this._libStmt.size === 0 &&
            this._options?.pyteaLibPath &&
            !(this._options?.extractIR === true)
        ) {
            this._libStmt = getStmtsFromDir(this._service, this._options.pyteaLibPath);
            this._libStmt.forEach(([path]) => {
                this._pathStore.addPath(path);
            });
            this._pushTimeLog('Translate library scripts');
        }
    }

    // return error message (string) or undefined
    translateMainEntry(entryPath: string): string | undefined {
        if (!entryPath) {
            return 'path is blank';
        }

        if (!fs.existsSync(entryPath)) {
            return `path ${entryPath} does not exists`;
        }

        if (path.extname(entryPath) !== '.py') {
            return `entry point ${entryPath} is not a python script`;
        }

        if (!this._service) {
            return `pyright service is not set.`;
        }

        // translate project scripts
        const projectPath = path.join(entryPath, '..');
        this._projectStmt = getStmtsFromDir(this._service, projectPath);
        this._projectStmt.forEach(([path]) => {
            this._pathStore.addPath(path);
        });

        this._pushTimeLog('Translate project scripts');

        return;
    }

    async analyze(): Promise<ContextSet<ShValue | ShContFlag>> {
        if (!this.validate()) {
            return Promise.reject('failed to validate PyTea service.');
        }

        // set random variable seeds
        this._randomGen.clear();
        const seedList = this._options?.variableSeed;
        if (seedList) {
            Object.entries(seedList).forEach(([prefix, seed]) => {
                this._randomGen.set(prefix, new RandomGen(seed));
            });
        }

        // run builtins
        const builtinsPair = this._libStmt.get('builtins');
        if (!builtinsPair) {
            return Promise.reject('cannot find PyTea implemenation of Python builtins.');
        }
        const builtins = builtinsPair[1];

        const builtinSet = this._builtinSet ?? TorchBackend.runBuiltin(builtins, 'builtins');
        this._builtinSet = builtinSet;

        const entryPath = this._options!.entryPath;
        const entryName = path.basename(entryPath, path.extname(entryPath));
        const stmtPair = this._projectStmt?.get(entryName);

        if (!stmtPair) {
            this._mainStmt = undefined;
            return Promise.reject(`cannot parse entry file '${entryPath}'`);
        }

        const stmt = stmtPair[1];
        this._pushTimeLog('Running builtin libraries');

        this._mainStmt = stmt;

        const startSet = builtinSet.map((ctx) => {
            // set __name__ to '__main__'
            const [nameAddr, newHeap] = ctx.heap.allocNew(SVString.create('__main__', undefined), undefined);
            return ctx.setRelPath(entryName).setEnv(ctx.env.setId('__name__', nameAddr)).setHeap(newHeap);
        });

        const result = TorchBackend.runWithFailure(
            startSet,
            stmt,
            this._console,
            this.options?.timeout,
            this.options?.maxPath
        );

        this._pushTimeLog('Running entry file');

        return result;
    }

    // spawnZ3Py(): ChildProcess | undefined {
    //     if (this._z3Process) return this._z3Process;

    //     if (!this.options?.pyteaLibPath) {
    //         this._console.error(`pyteaLibPath is not set'. skip z3`);
    //         return;
    //     }

    //     const pyteaPath = path.join(this.options.pyteaLibPath, '..', 'z3wrapper', 'pyteaserver.py');

    //     if (!fs.existsSync(pyteaPath)) {
    //         this._console.error(`cannot found pytea server script at '${pyteaPath}'. skip z3`);
    //         return;
    //     }

    //     const child = spawn('python', [pyteaPath, `--port=${this.options.z3Port}`]);
    //     child.on('exit', () => {
    //         this._z3Process = undefined;
    //     });
    //     this._z3Process = child;

    //     return child;
    // }

    async runZ3Py(result: ContextSet<ShValue | ShContFlag>): Promise<PyZ3RPCResult[]> {
        const port = this._options?.z3Port ?? defaultOptions.z3Port;
        const address = `http://localhost:${port}/`;

        const randNum = Math.floor(Math.random() * 10000);
        const ping = await postJsonRpc(address, -1, 'ping', randNum);
        if (ping !== randNum) {
            return Promise.reject(`Z3 Python server send wrong respond. send: ${randNum}, got: ${ping}`);
        }

        const jsonList: object[] = [];
        result.getList().forEach((ctx) => {
            jsonList.push(ctx.ctrSet.getConstraintObject());
        });
        result.getStopped().forEach((ctx) => {
            jsonList.push(ctx.ctrSet.getConstraintObject());
        });
        if (jsonList.length === 0) {
            return Promise.resolve([]);
        }

        // random segmented id
        this._requestId = (this._requestId + 1) % 10000000;

        const respond = (await postJsonRpc(address, this._requestId, 'solve', jsonList)) as PyZ3RPCResult[];

        this._pushTimeLog('Running Z3 solver');

        return respond;
    }

    printLog(result: ContextSet<ShValue | ShContFlag>): void {
        const logLevel = this._options!.logLevel;

        if (logLevel === 'none') {
            this._noneLog(result);
            return;
        }

        this._console.info('\n------------- constraint generator result -------------\n');
        switch (logLevel) {
            case 'result-only':
                this._resultOnlyLog(result);
                break;
            case 'reduced':
                this._reducedLog(result);
                break;
            case 'full':
                this._fullLog(result);
                break;
            default:
                this._noneLog(result);
                break;
        }
    }

    printLogWithZ3(result: ContextSet<ShValue | ShContFlag>, z3Result: PyZ3RPCResult[]): void {
        const logLevel = this._options!.logLevel;

        switch (logLevel) {
            case 'result-only':
                this._resultOnlyLogWithZ3(result, z3Result);
                break;
            case 'reduced':
                this._reducedLog(result);
                break;
            case 'full':
                this._fullLog(result);
                break;
            default:
                this._noneLog(result);
                break;
        }
    }

    extractIR(resultPath: string): void {
        let sourceMap = '';
        this._projectStmt?.forEach((stmt, path) => {
            sourceMap = sourceMap + (sourceMap ? '\n' : '') + IRWriter.makeIRString(stmt[1], path);
        });
        fs.writeFileSync(resultPath, sourceMap);
    }

    // Dynamic communications with Backend
    // import resolution order: (e.g. from A.B import C)
    //      1. project script   (A/B.py)
    //      2. __init__.py from project (A/B/__init__.py)
    //      3. library script (site-packages/A/B.py)
    //      4. __init__.py from project (site-packages/A/B/__init__.py)
    //
    // boolean value indicates imported from __init__
    getImportModuleStmt(qualPath: string): [ThStmt | undefined, boolean] {
        const initPath = qualPath + '.__init__';
        let pair: [string, ThStmt] | undefined;
        let fromInit = false;

        if (this._projectStmt?.has(qualPath)) {
            pair = this._projectStmt.get(qualPath);
        } else if (this._projectStmt?.has(initPath)) {
            pair = this._projectStmt.get(initPath);
            fromInit = true;
        } else if (this._libStmt.has(qualPath)) {
            pair = this._libStmt.get(qualPath);
        } else if (this._libStmt.has(initPath)) {
            pair = this._libStmt.get(initPath);
            fromInit = true;
        }

        return [pair ? pair[1] : undefined, fromInit];
    }

    getSourceRange(source: CodeSource | undefined): [string, Range] | undefined {
        if (!source) return;

        const node = source;
        if (!('fileId' in node)) {
            let moduleNode = node;
            while (moduleNode.nodeType !== ParseNodeType.Module) {
                moduleNode = moduleNode.parent!;
            }

            const fileInfo = getFileInfo(moduleNode)!;

            const filePath = fileInfo.filePath;
            const lines = fileInfo.lines;
            const start = convertOffsetToPosition(node.start, lines);
            const end = convertOffsetToPosition(node.start + node.length, lines);
            return [filePath, { start, end }];
        } else {
            const filePath = this._pathStore.getPath(node.fileId);
            if (!filePath) return;
            return [filePath, node.range];
        }
    }

    private _genCustomPrinter(result: Context<unknown>): CustomObjectPrinter[] {
        const { env, heap } = result;

        const printer: CustomObjectPrinter[] = [];
        const tuple = fetchAddr(env.getId('tuple'), heap);
        if (tuple?.type === SVType.Object) {
            const mro = fetchAddr(tuple.getAttr('__mro__'), heap);
            if (mro) {
                function tuplePrinter(value: SVObject): string {
                    const valueStr = value.indices.map((v) => v.toString()).join(', ');
                    return `[${value.addr.addr}](${valueStr})`;
                }
                printer.push([mro, tuplePrinter]);
            }
        }

        return printer;
    }

    private _noneLog(result: ContextSet<ShValue | ShContFlag>): void {
        // do nothing.
    }

    private _resultOnlyLog(result: ContextSet<ShValue | ShContFlag>): void {
        const success = result.getList();
        const stopped = result.getStopped();
        const failed = result.getFailed();

        const succCnt = success.count();
        const stopCnt = stopped.count();

        success.forEach((ctx, i) => {
            if (ctx.logs.count() > 0) {
                this._console.info(
                    chalk.green(`success path #${i + 1}`) + `\n\n${chalk.bold('LOGS')}:\n${this._logsToString(ctx)}\n`
                );
            }
        });

        stopped.forEach((ctx, i) => {
            const source = ctx.retVal.source;

            let logs = '';
            if (ctx.logs.count() > 0) {
                logs = `\n${chalk.bold('LOGS')}:\n${this._logsToString(ctx)}\n`;
            }

            this._console.info(
                chalk.yellow(`stopped path #${succCnt + i + 1}`) +
                    chalk.bold(`: ${ctx.retVal.reason}`) +
                    ` - ${formatCodeSource(source, this._pathStore)}\n` +
                    `\n${chalk.bold('CALL STACK')}:\n${this._callStackToString(ctx)}\n` +
                    logs +
                    '\n'
            );
        });

        let timeout = 0;
        failed.forEach((ctx, i) => {
            const source = ctx.retVal.source;
            if (ctx.isTimedOut()) {
                timeout++;
                return;
            }

            let logs = '';
            if (ctx.logs.count() > 0) {
                logs = `\n${chalk.bold('LOGS')}:\n${this._logsToString(ctx)}\n`;
            }

            this._console.info(
                chalk.red(`failed path #${succCnt + stopCnt + i + 1}`) +
                    chalk.bold(`: ${ctx.retVal.reason}`) +
                    `- ${formatCodeSource(source, this._pathStore)}\n` +
                    `\n${chalk.bold('CALL STACK')}:\n${this._callStackToString(ctx)}\n` +
                    logs +
                    '\n'
            );
        });

        this._pushTimeLog('printing results');

        const totalPaths = success.count() + stopped.count() + failed.count();
        this._console.info(
            `${chalk.bold('<OVERALL: total ' + totalPaths.toString() + ' paths>\n')}` +
                chalk.green(`potential success path #: ${success.count()}\n`) +
                chalk.yellow(`potential unreachable path #: ${stopped.count()}\n`) +
                chalk.red(`immediate failed path #: ${failed.count()}\n`) +
                (timeout ? `  - timed out path #: ${timeout}\n` : '')
        );
    }

    private _resultOnlyLogWithZ3(result: ContextSet<ShValue | ShContFlag>, z3Result: PyZ3RPCResult[]): void {
        const success = result.getList();
        const stopped = result.getStopped();
        const failed = result.getFailed();

        let validCnt = 0;
        let invalidCnt = 0;
        let unreachCnt = 0;
        let undecCnt = 0;
        let idx = 0;
        const z3Len = z3Result.length;

        success.forEach((ctx, i) => {
            const z3 = idx < z3Len ? z3Result[idx] : undefined;
            if (z3) {
                switch (z3.type) {
                    case PyZ3RPCResultType.Valid:
                    case PyZ3RPCResultType.Sat:
                        validCnt += 1;
                        break;
                    case PyZ3RPCResultType.Unsat:
                        invalidCnt += 1;
                        break;
                    case PyZ3RPCResultType.Unreachable:
                        unreachCnt += 1;
                        break;
                    case PyZ3RPCResultType.DontKnow:
                    case PyZ3RPCResultType.Timeout:
                        undecCnt += 1;
                        break;
                }
            }

            if (ctx.logs.count() > 0) {
                this._console.info(
                    chalk.green(`success path #${i + 1}`) + `\n\n${chalk.bold('LOGS')}:\n${this._logsToString(ctx)}\n`
                );
            }
            idx += 1;
        });

        stopped.forEach((ctx, i) => {
            const z3 = idx < z3Len ? z3Result[idx] : undefined;
            if (z3) {
                switch (z3.type) {
                    case PyZ3RPCResultType.Valid:
                    case PyZ3RPCResultType.Sat:
                        validCnt += 1;
                        break;
                    case PyZ3RPCResultType.Unsat:
                        invalidCnt += 1;
                        break;
                    case PyZ3RPCResultType.Unreachable:
                        unreachCnt += 1;
                        break;
                    case PyZ3RPCResultType.DontKnow:
                    case PyZ3RPCResultType.Timeout:
                        undecCnt += 1;
                        break;
                }
            }

            const source = ctx.retVal.source;

            let logs = '';
            if (ctx.logs.count() > 0) {
                logs = `\n${chalk.bold('LOGS')}:\n${this._logsToString(ctx)}\n`;
            }

            this._console.info(
                chalk.yellow(`stopped path #${idx + 1}`) +
                    chalk.bold(`: ${ctx.retVal.reason}`) +
                    ` - ${formatCodeSource(source, this._pathStore)}\n` +
                    `\n${chalk.bold('CALL STACK')}:\n${this._callStackToString(ctx)}\n` +
                    logs +
                    '\n'
            );

            idx += 1;
        });

        let timeout = 0;
        failed.forEach((ctx, i) => {
            const source = ctx.retVal.source;
            if (ctx.isTimedOut()) {
                timeout++;
                return;
            }

            let logs = '';
            if (ctx.logs.count() > 0) {
                logs = `\n${chalk.bold('LOGS')}:\n${this._logsToString(ctx)}\n`;
            }

            this._console.info(
                chalk.red(`failed path #${idx + 1}`) +
                    chalk.bold(`: ${ctx.retVal.reason}`) +
                    `- ${formatCodeSource(source, this._pathStore)}\n` +
                    `\n${chalk.bold('CALL STACK')}:\n${this._callStackToString(ctx)}\n` +
                    logs +
                    '\n'
            );

            idx += 1;
        });

        this._pushTimeLog('printing results');

        this._console.info(
            `${chalk.bold('<OVERALL: total ' + idx + ' paths>\n')}` +
                chalk.green(`valid path #: ${validCnt}\n`) +
                chalk.red(`invalid path #: ${invalidCnt + failed.count()}\n`) +
                (undecCnt > 0 ? chalk.yellow(`undecidable path #: ${undecCnt}\n`) : '') +
                (unreachCnt > 0 ? chalk.gray(`unreachable path #: ${unreachCnt}\n`) : '') +
                (timeout ? `  - constraint generator timed out path #: ${timeout}\n` : '')
        );
    }

    private _reducedLog(result: ContextSet<ShValue | ShContFlag>): void {
        const success = result.getList();
        const failed = result.getFailed();
        const stopped = result.getStopped();

        const jsonList: string[] = [];

        const succCnt = success.count();
        const stopCnt = stopped.count();

        // if (this._mainStmt) {
        //     this._console.info(
        //         chalk.yellow(`PARSED STATEMENTS:`) + chalk.gray(`\n${ThStmt.toString(this._mainStmt)}\n`)
        //     );
        // }

        success.forEach((ctx, i) => {
            jsonList.push(ctx.ctrSet.getConstraintJSON(this._pathStore));

            const objPrinter = this._genCustomPrinter(ctx);

            let heapLog = '';
            // TODO: currently assume that address 1 is main module object
            //       do not hardcode.
            const module = ctx.heap.getVal(1);
            if (module?.type === SVType.Object) {
                heapLog =
                    `REDUCED HEAP: (size: ${ctx.heap.valMap.count()} values)\n` +
                    module.attrs
                        .map((v, k) => {
                            return `  ${k} => ${reducedToString(v, ctx.heap, objPrinter)}`;
                        })
                        .join('\n');
            }

            this._console.info(
                chalk.green(`success path #${i + 1}\n\n`) +
                    heapLog +
                    `\n\n${chalk.bold('LOGS')}:\n${this._logsToString(ctx)}` +
                    `\n\n${chalk.bold('CONSTRAINTS')}:\n${ctx.ctrSet.toString(this._pathStore)}\n`
            );
        });

        stopped.forEach((ctx, i) => {
            jsonList.push(ctx.ctrSet.getConstraintJSON(this._pathStore));

            const source = ctx.retVal.source;
            const objPrinter = this._genCustomPrinter(ctx);

            const heapLog = ctx.env.addrMap
                .filter((v) => v.addr >= 0)
                .map((addr, key) => {
                    return `  ${key} => ${reducedToString(addr, ctx.heap, objPrinter)}`;
                })
                .join('\n');

            this._console.info(
                chalk.yellow(`stopped path #${succCnt + i + 1}`) +
                    chalk.bold(`: ${ctx.retVal.reason}`) +
                    `- ${formatCodeSource(source, this._pathStore)}\n` +
                    `\n\n${chalk.bold('REDUCED HEAP')} (size: ${ctx.heap.valMap.count()} values):\n${heapLog}` +
                    `\n\n${chalk.bold('CALL STACK')}:\n` +
                    this._callStackToString(ctx) +
                    `\n\n${chalk.bold('LOGS')}:\n${this._logsToString(ctx)}` +
                    `\n\n${chalk.bold('CONSTRAINTS')}:\n` +
                    ctx.ctrSet.toString(this._pathStore) +
                    '\n'
            );
        });

        failed.forEach((ctx, i) => {
            const source = ctx.retVal.source;
            const objPrinter = this._genCustomPrinter(ctx);

            const heapLog = ctx.env.addrMap
                .filter((v) => v.addr >= 0)
                .map((addr, key) => {
                    return `  ${key} => ${reducedToString(addr, ctx.heap, objPrinter)}`;
                })
                .join('\n');

            this._console.info(
                chalk.red(`failed path #${succCnt + stopCnt + i + 1}`) +
                    chalk.bold(`: ${ctx.retVal.reason}`) +
                    `- ${formatCodeSource(source, this._pathStore)}\n` +
                    `\n\n${chalk.bold('REDUCED HEAP')} (size: ${ctx.heap.valMap.count()} values):\n${heapLog}` +
                    `\n\n${chalk.bold('CALL STACK')}:\n` +
                    this._callStackToString(ctx) +
                    `\n\n${chalk.bold('LOGS')}:\n${this._logsToString(ctx)}` +
                    `\n\n${chalk.bold('CONSTRAINTS')}:\n` +
                    ctx.ctrSet.toString(this._pathStore) +
                    '\n'
            );
        });

        this._pushTimeLog('printing results');

        const totalPaths = success.count() + stopped.count() + failed.count();
        this._console.info(
            `${chalk.bold('<OVERALL: total ' + totalPaths.toString() + ' paths>\n')}` +
                chalk.green(`potential success path #: ${success.count()}\n`) +
                chalk.yellow(`potential unreachable path #: ${stopped.count()}\n`) +
                chalk.red(`immediate failed path #: ${failed.count()}\n\n`) +
                `${chalk.bold('PROCESS TIMES')}:\n` +
                this._timeLog.map(([name, interval]) => `  ${name}: ${(interval / 1000).toFixed(4)}s`).join('\n') +
                '\n'
        );
    }

    private _fullLog(result: ContextSet<ShValue | ShContFlag>): void {
        const success = result.getList();
        const failed = result.getFailed();
        const stopped = result.getStopped();

        const succCnt = success.count();
        const stopCnt = stopped.count();

        if (this._mainStmt) {
            this._console.info(
                chalk.yellow(`PARSED STATEMENTS:`) + chalk.gray(`\n${ThStmt.toString(this._mainStmt)}\n`)
            );
        }

        success.forEach((ctx, i) => {
            this._console.info(
                chalk.green(`success path #${i + 1}`) +
                    `\nLOGS:\n${this._logsToString(ctx)}\n` +
                    `CONSTRAINTS:\n${ctx.ctrSet.toString(this._pathStore)}\n` +
                    `ENV:\n${ctx.env.toString()}\n` +
                    `HEAP (size: ${ctx.heap.valMap.count()} values):\n${ctx.heap
                        .filter((_, key) => key >= 0)
                        .toString()}\n`
            );
        });

        stopped.forEach((ctx, i) => {
            const source = ctx.retVal.source;

            this._console.info(
                chalk.yellow(`stopped path #${succCnt + i + 1}`) +
                    chalk.bold(`: ${ctx.retVal.reason}`) +
                    `- ${formatCodeSource(source, this._pathStore)}\n` +
                    `LOGS:\n${this._logsToString(ctx)}\n` +
                    'CONSTRAINTS:\n' +
                    ctx.ctrSet.toString(this._pathStore) +
                    '\n\nCALL STACK:\n' +
                    this._callStackToString(ctx) +
                    `\nENV:\n${ctx.env.toString()}\n` +
                    `\nHEAP (size: ${ctx.heap.valMap.count()} values):\n${ctx.heap
                        .filter((_, key) => key >= 0)
                        .toString()}\n`
            );
        });

        failed.forEach((ctx, i) => {
            const source = ctx.retVal.source;

            this._console.info(
                chalk.red(`failed path #${succCnt + stopCnt + i + 1}`) +
                    chalk.bold(`: ${ctx.retVal.reason}`) +
                    `- ${formatCodeSource(source, this._pathStore)}\n` +
                    `LOGS:\n${this._logsToString(ctx)}\n` +
                    'CONSTRAINTS:\n' +
                    ctx.ctrSet.toString(this._pathStore) +
                    '\n\nCALL STACK:\n' +
                    this._callStackToString(ctx) +
                    `\nENV:\n${ctx.env.toString()}\n` +
                    `\nHEAP (size: ${ctx.heap.valMap.count()} values):\n${ctx.heap
                        .filter((_, key) => key >= 0)
                        .toString()}\n`
            );
        });

        this._pushTimeLog('printing results');

        const totalPaths = success.count() + stopped.count() + failed.count();
        this._console.info(
            `${chalk.bold('<OVERALL: total ' + totalPaths.toString() + ' paths>\n')}` +
                chalk.green(`potential success path #: ${success.count()}\n`) +
                chalk.yellow(`potential unreachable path #: ${stopped.count()}\n`) +
                chalk.red(`immediate failed path #: ${failed.count()}\n\n`) +
                `${chalk.bold('PROCESS TIMES')}:\n` +
                this._timeLog.map(([name, interval]) => `  ${name}: ${(interval / 1000).toFixed(4)}s`).join('\n') +
                '\n'
        );
    }

    private _unittestLog(passOrFail: boolean, result: ContextSet<ShValue | ShContFlag>): boolean {
        const success = result.getList();
        const failed = result.getFailed();

        const jsonList: string[] = [];

        let hasSVError = false;

        success.forEach((ctx, i) => {
            jsonList.push(ctx.ctrSet.getConstraintJSON(this._pathStore));

            let heapLog = '';
            // TODO: currently assume that address 1 is main module object
            //       do not hardcode.
            const module = ctx.heap.getVal(1);
            if (module?.type === SVType.Object) {
                heapLog =
                    `REDUCED HEAP: (size: ${ctx.heap.valMap.count()})\n` +
                    module.attrs
                        .map((v, k) => {
                            return `  ${k} => ${reducedToString(v, ctx.heap)}`;
                        })
                        .join('\n');
            }

            ctx.logs.forEach((value, i) => {
                if (value.type === SVType.Error) {
                    this._console.info(
                        `success path #${i + 1}\n\nLOGS:${this._logsToString(
                            ctx
                        )}\n\nCONSTRAINTS:\n${ctx.ctrSet.toString()}\n\n${heapLog}`
                    );
                    hasSVError = true;
                }
            });
        });

        if (passOrFail) {
            return failed.count() === 0 && !hasSVError;
        } else {
            return success.count() === 0 && !hasSVError;
        }
    }

    private _clearTimeLog(): void {
        this._currTime = performance.now();
        this._timeLog = [];
    }

    private _pushTimeLog(logName: string): void {
        const temp = this._currTime;
        this._currTime = performance.now();
        this._timeLog.push([logName, this._currTime - temp]);
    }

    private _logsToString(ctx: Context<unknown>): string {
        return ctx.logs
            .map((log) => {
                const posStr = formatCodeSource(log.source);

                if (log.type === SVType.Error) {
                    return `${log.reason} - ${posStr}`;
                } else {
                    return `${log.toString()} - ${posStr}`;
                }
            })
            .join('\n');
    }

    private _callStackToString(ctx: Context<unknown>): string {
        return ctx.callStack
            .filter(([f, source]) => {
                // filter callKV libcall / internal $call
                if (typeof f !== 'string') f = f.name;

                return !(f === 'callKV' || (f.endsWith('self$call') && !source));
            })
            .map(([func, node]) => {
                let funName = typeof func === 'string' ? func : func.name;
                if (funName.endsWith('$TMP$')) {
                    funName = funName.slice(0, -5);
                }
                return `${funName} - ${formatCodeSource(node)}`;
            })
            .reverse()
            .join('\n');
    }
}
