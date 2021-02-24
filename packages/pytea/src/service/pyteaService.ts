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
import { ParseNode, ParseNodeType } from 'pyright-internal/parser/parseNodes';

import { Context, ContextSet } from '../backend/context';
import { CodeRange, CodeSource, ShContFlag, ShValue, SVAddr, SVString, SVType } from '../backend/sharpValues';
import { TorchBackend } from '../backend/torchBackend';
import { IRWriter } from '../frontend/IRReaderWriter';
import { ThStmt } from '../frontend/torchStatements';
import { FilePathStore } from './executionPaths';
import { defaultOptions, PyCmdArgs, PyteaOptions } from './pyteaOptions';
import { getStmtsFromDir, reducedToString } from './pyteaUtils';

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

    constructor(service?: AnalyzerService, options?: PyteaOptions, console?: ConsoleInterface, setDefault?: boolean) {
        if (setDefault) _globalService = this;

        this._console = console || new StandardConsole();

        this._timeLog = [];
        this._currTime = performance.now();

        this._options = options ?? defaultOptions;
        this._service = service;

        this._libStmt = new Map();
        this._pathStore = new FilePathStore();
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

    analyze(): ContextSet<ShValue | ShContFlag> | undefined {
        if (!this.validate()) {
            this._console.error('failed to validate PyTea service.');
            return;
        }

        const builtinsPair = this._libStmt.get('builtins');
        if (!builtinsPair) {
            this._console.error('cannot find PyTea implemenation of Python builtins.');
            return;
        }
        const builtins = builtinsPair[1];

        const builtinSet = this._builtinSet ?? TorchBackend.runBuiltin(builtins, 'builtins');
        this._builtinSet = builtinSet;

        const entryPath = this._options!.entryPath;
        const entryName = path.basename(entryPath, path.extname(entryPath));
        const stmtPair = this._projectStmt?.get(entryName);

        if (!stmtPair) {
            this._mainStmt = undefined;
            this._console.error(`cannot parse entry file '${entryPath}'`);
            return;
        }

        const stmt = stmtPair[1];
        this._pushTimeLog('Running builtin libraries');

        this._mainStmt = stmt;

        const startSet = builtinSet.map((ctx) => {
            // set __name__ to '__main__'
            const [nameAddr, newHeap] = ctx.heap.allocNew(SVString.create('__main__'));
            return ctx.setRelPath(entryName).setEnv(ctx.env.setId('__name__', nameAddr)).setHeap(newHeap);
        });
        const result = TorchBackend.run(startSet, stmt);

        this._pushTimeLog('Running entry file');

        return result;
    }

    printLog(result: ContextSet<ShValue | ShContFlag>): void {
        const logLevel = this._options!.logLevel;
        switch (logLevel) {
            case 'none':
                this._noneLog(result);
                break;
            case 'result-only':
                this._resultOnlyLog(result);
                break;
            case 'reduced':
                this._reducedLog(result);
                break;

            case 'full':
                this._fullLog(result);
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

    formatParseNodeRange(node: ParseNode): string {
        let moduleNode = node;
        while (moduleNode.nodeType !== ParseNodeType.Module) {
            moduleNode = moduleNode.parent!;
        }

        const fileInfo = getFileInfo(moduleNode)!;

        const filePath = fileInfo.filePath;
        const lines = fileInfo.lines;
        const start = convertOffsetToPosition(node.start, lines);
        const end = convertOffsetToPosition(node.start + node.length, lines);

        return `[${start.line + 1}:${start.character} - ${end.line + 1}:${end.character}] (${filePath})`;
    }

    formatCodeSource(node?: CodeSource): string {
        if (!node) return 'internal';

        // check ParseNode or not
        if (!('fileId' in node)) {
            return this.formatParseNodeRange(node);
        } else {
            const { start, end } = node.range;
            return `[${start.line + 1}:${start.character} - ${end.line + 1}:${end.character}]`;
        }
    }

    getSourceRange(source?: CodeSource): [string, Range] | undefined {
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

    private _noneLog(result: ContextSet<ShValue | ShContFlag>): void {
        // do nothing.
    }

    private _resultOnlyLog(result: ContextSet<ShValue | ShContFlag>): void {
        const success = result.getList();
        const failed = result.getFailed();
        const stopped = result.getStopped();

        failed.forEach((ctx, i) => {
            const source = ctx.retVal.source;

            this._console.info(`failed path #${i + 1}: ${ctx.retVal.reason} - ${this.formatCodeSource(source)}\n\n`);
        });

        this._pushTimeLog('printing results');

        this._console.info(
            chalk.green(`potential success path #: ${success.count()}\n`) +
                chalk.yellow(`potential unreachable path #: ${stopped.count()}\n`) +
                chalk.red(`immediate failed path #: ${failed.count()}\n\n`) +
                'RUNNING TIMES:\n' +
                this._timeLog.map(([name, interval]) => `  ${name}: ${(interval / 1000).toFixed(4)}s`).join('\n')
        );
    }

    private _reducedLog(result: ContextSet<ShValue | ShContFlag>): void {
        const success = result.getList();
        const failed = result.getFailed();
        const stopped = result.getStopped();

        const jsonList: string[] = [];

        if (this._mainStmt) {
            this._console.info(
                chalk.yellow(`PARSED STATEMENTS:`) + chalk.gray(`\n${ThStmt.toString(this._mainStmt)}\n`)
            );
        }

        success.forEach((ctx, i) => {
            jsonList.push(ctx.ctrSet.getConstraintJSON());

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

            this._console.info(
                chalk.green(`success path #${i + 1}`) +
                    `\n\nLOGS:\n${this._logsToString(ctx)}\n\nCONSTRAINTS:\n${ctx.ctrSet.toString()}\n\n${heapLog}`
            );
        });

        stopped.forEach((ctx, i) => {
            jsonList.push(ctx.ctrSet.getConstraintJSON());

            const source = ctx.retVal.source;

            const heapLog = ctx.env.addrMap
                .filter((v) => v.addr >= 0)
                .map((addr, key) => {
                    return `  ${key} => ${reducedToString(addr, ctx.heap)}`;
                })
                .join('\n');

            this._console.info(
                chalk.yellow(`stopped path #${i + 1}`) +
                    `: ${ctx.retVal.reason} - ${this.formatCodeSource(source)}\n\n` +
                    `LOGS:\n${this._logsToString(ctx)}\n\n` +
                    'CONSTRAINTS:\n' +
                    ctx.ctrSet.toString() +
                    '\n\nCALL STACK:\n' +
                    this._callStackToString(ctx) +
                    `\n\nREDUCED HEAP (${ctx.heap.valMap.count()}):\n${heapLog}`
            );
        });

        failed.forEach((ctx, i) => {
            const source = ctx.retVal.source;

            const heapLog = ctx.env.addrMap
                .filter((v) => v.addr >= 0)
                .map((addr, key) => {
                    return `  ${key} => ${reducedToString(addr, ctx.heap)}`;
                })
                .join('\n');

            this._console.info(
                chalk.red(`failed path #${i + 1}`) +
                    `: ${ctx.retVal.reason} - ${this.formatCodeSource(source)}\n\n` +
                    `LOGS:\n${this._logsToString(ctx)}\n\n` +
                    'CONSTRAINTS:\n' +
                    ctx.ctrSet.toString() +
                    '\n\nCALL STACK:\n' +
                    this._callStackToString(ctx) +
                    `\n\nREDUCED HEAP (${ctx.heap.valMap.count()}):\n${heapLog}`
            );
        });

        this._pushTimeLog('printing results');

        this._console.info(
            chalk.green(`potential success path #: ${success.count()}\n`) +
                chalk.yellow(`potential unreachable path #: ${stopped.count()}\n`) +
                chalk.red(`immediate failed path #: ${failed.count()}\n\n`) +
                'RUNNING TIMES:\n' +
                this._timeLog.map(([name, interval]) => `  ${name}: ${(interval / 1000).toFixed(4)}s`).join('\n')
        );
    }

    private _fullLog(result: ContextSet<ShValue | ShContFlag>): void {
        const success = result.getList();
        const failed = result.getFailed();
        const stopped = result.getStopped();

        if (this._mainStmt) {
            this._console.info(
                chalk.yellow(`PARSED STATEMENTS:`) + chalk.gray(`\n${ThStmt.toString(this._mainStmt)}\n`)
            );
        }

        success.forEach((ctx, i) => {
            this._console.info(
                chalk.green(`success path #${i + 1}`) +
                    `\nLOGS:\n${this._logsToString(ctx)}\n` +
                    `CONSTRAINTS:\n${ctx.ctrSet.toString()}\n` +
                    `ENV:\n${ctx.env.toString()}\n` +
                    `HEAP (size: ${ctx.heap.valMap.count()}):\n${ctx.heap.filter((_, key) => key >= 0).toString()}\n`
            );
        });

        stopped.forEach((ctx, i) => {
            const source = ctx.retVal.source;

            this._console.info(
                chalk.yellow(`stopped path #${i + 1}`) +
                    `: ${ctx.retVal.reason} / at ${ctx.relPath} ${this.formatCodeSource(source)}\n` +
                    `LOGS:\n${this._logsToString(ctx)}\n` +
                    'CONSTRAINTS:\n' +
                    ctx.ctrSet.toString() +
                    '\n\nCALL STACK:\n' +
                    this._callStackToString(ctx) +
                    `\nENV:\n${ctx.env.toString()}\n` +
                    `\nHEAP (${ctx.heap.valMap.count()}):\n${ctx.heap.filter((_, key) => key >= 0).toString()}`
            );
        });

        failed.forEach((ctx, i) => {
            const source = ctx.retVal.source;

            this._console.info(
                chalk.red(`failed path #${i + 1}`) +
                    `: ${ctx.retVal.reason} / at ${ctx.relPath} ${this.formatCodeSource(source)}\n` +
                    `LOGS:\n${this._logsToString(ctx)}\n` +
                    'CONSTRAINTS:\n' +
                    ctx.ctrSet.toString() +
                    '\n\nCALL STACK:\n' +
                    this._callStackToString(ctx) +
                    `\nENV:\n${ctx.env.toString()}\n` +
                    `\nHEAP (${ctx.heap.valMap.count()}):\n${ctx.heap.filter((_, key) => key >= 0).toString()}`
            );
        });

        this._pushTimeLog('printing results');

        this._console.info(
            chalk.green(`potential success path #: ${success.count()}\n`) +
                chalk.yellow(`potential unreachable path #: ${stopped.count()}\n`) +
                chalk.red(`immediate failed path #: ${failed.count()}\n\n`) +
                'RUNNING TIMES:\n' +
                this._timeLog.map(([name, interval]) => `  ${name}: ${(interval / 1000).toFixed(4)}s`).join('\n')
        );
    }

    private _unittestLog(passOrFail: boolean, result: ContextSet<ShValue | ShContFlag>): boolean {
        const success = result.getList();
        const failed = result.getFailed();

        const jsonList: string[] = [];

        let hasSVError = false;

        success.forEach((ctx, i) => {
            jsonList.push(ctx.ctrSet.getConstraintJSON());

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
                const posStr = this.formatCodeSource(log.source);

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
            .filter(([f, _]) => {
                // filter callKV libcall
                if (typeof f === 'string') {
                    return f !== 'callKV';
                } else {
                    return f.name !== 'callKV';
                }
            })
            .map(([func, node]) => `${typeof func === 'string' ? func : func.name} - ${this.formatCodeSource(node)}`)
            .reverse()
            .join('\n');
    }
}
