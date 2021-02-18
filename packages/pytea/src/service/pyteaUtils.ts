/*
 * pyteaUtils.ts
 * Copyright (c) Seoul National University.
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo (mersshs@gmail.com)
 *
 * Utility functions for PyTea service.
 */
import { spawn } from 'child_process';
import { CommandLineOptions } from 'command-line-args';
import * as fs from 'fs';
import * as path from 'path';
import tmp from 'tmp';
import * as util from 'util';

import { getFileInfo } from 'pyright-internal/analyzer/analyzerNodeInfo';
import { AnalyzerService } from 'pyright-internal/analyzer/service';
import { ConfigOptions } from 'pyright-internal/common/configOptions';
import { ConsoleInterface } from 'pyright-internal/common/console';
import {
    combinePaths,
    ensureTrailingDirectorySeparator,
    getDirectoryPath,
    getPathComponents,
    normalizePath,
} from 'pyright-internal/common/pathUtils';
import { convertOffsetToPosition } from 'pyright-internal/common/positionUtils';
import { ParseNodeType } from 'pyright-internal/parser/parseNodes';

import { fetchAddr } from '../backend/backUtils';
import {
    Constraint,
    ConstraintType,
    CtrAnd,
    CtrBroad,
    CtrEq,
    CtrExpBool,
    CtrFail,
    CtrForall,
    CtrLt,
    CtrLte,
    CtrNeq,
    CtrNot,
    CtrOr,
} from '../backend/constraintType';
import { ContextSet } from '../backend/context';
import { ShHeap } from '../backend/sharpEnvironments';
import { CodeRange, CodeSource, ShValue, SVSize, SVType } from '../backend/sharpValues';
import { ExpBool, ExpNum, ExpShape, ExpString, SymExp, SymInt, SymVal } from '../backend/symExpressions';
import { TorchIRFrontend } from '../frontend/torchFrontend';
import { ThStmt } from '../frontend/torchStatements';
import { FilePathStore } from './executionPaths';
import { PyteaOptions } from './pyteaOptions';

export class NodeConsole implements ConsoleInterface {
    logger: ReturnType<typeof util.debuglog>;

    constructor(loggerName: string) {
        this.logger = util.debuglog(loggerName);
    }

    warn(message: string) {
        this.logger('\n' + message + '\n');
    }

    info(message: string) {
        this.logger('\n' + message + '\n');
    }

    log(message: string) {
        this.logger('\n' + message + '\n');
    }

    error(message: string) {
        this.logger('\nERROR: ' + message + '\n');
    }
}

export function getPyteaLibPath() {
    const pylibDir = 'pylib';
    let moduleDirectory = (global as any).__rootDirectory;
    if (!moduleDirectory) {
        return undefined;
    }

    moduleDirectory = getDirectoryPath(ensureTrailingDirectorySeparator(normalizePath(moduleDirectory)));

    const pylibPath = combinePaths(moduleDirectory, pylibDir);
    if (fs.existsSync(pylibPath)) {
        return pylibPath;
    }

    // In the debug version of Pytea, the code is one level
    // deeper, so we need to look one level up for the typeshed fallback.
    const debugPylibPath = combinePaths(getDirectoryPath(moduleDirectory), pylibDir);
    if (fs.existsSync(debugPylibPath)) {
        return debugPylibPath;
    }

    return undefined;
}

export function buildPyteaOption(
    args?: CommandLineOptions,
    basePath?: string,
    baseOptions?: PyteaOptions
): PyteaOptions | string {
    const cwd = basePath ?? path.normalize(process.cwd());
    args = args ?? {};

    const rawEntryPath: string = args['file'];
    const rawConfigPath: string = args['configPath'];
    const rawLibPath: string = args.libPath
        ? normalizePath(combinePaths(cwd, args.libPath))
        : baseOptions?.pyteaLibPath ?? '';

    const entryPath: string = rawEntryPath
        ? normalizePath(combinePaths(cwd, rawEntryPath))
        : baseOptions?.entryPath ?? '';
    let configPath: string = rawConfigPath
        ? normalizePath(combinePaths(cwd, rawConfigPath))
        : baseOptions?.configPath ?? '';

    if (!configPath && !entryPath) {
        return `neither configPath nor file path is found: ${entryPath}`;
    }
    if (entryPath && !fs.existsSync(entryPath)) {
        return `file path '${entryPath}' does not exist`;
    }

    let options: Partial<PyteaOptions> = {};
    options.configPath = configPath;

    // find config by entryPath if configPath is not set
    let isDir = false;
    if (!configPath && entryPath) {
        if (fs.lstatSync(entryPath).isDirectory()) {
            isDir = true;
        }
        const dirPath = isDir ? entryPath : path.dirname(entryPath);
        configPath = combinePaths(dirPath, 'pyteaconfig.json');
    }

    if (configPath && !fs.existsSync(configPath)) {
        console.log(`config json '${configPath}' does not exist. use default options`);
        configPath = '';
    }

    let dirPath: string;

    try {
        if (configPath) {
            dirPath = path.dirname(configPath);
            options = JSON.parse(fs.readFileSync(configPath).toString());
            if (options.entryPath) options.entryPath = normalizePath(combinePaths(dirPath, options.entryPath));
        } else {
            options = { ...baseOptions };
        }
    } catch (e) {
        throw `'${configPath}' is not a valid JSON file`;
    }

    // entry path is explicitly given && given path is not dir -> set entry path explicitly
    if (entryPath && !isDir) options.entryPath = entryPath;

    if (!options.entryPath || !fs.existsSync(options.entryPath)) {
        return `file path '${options.entryPath}' does not exist`;
    }

    dirPath = path.dirname(options.entryPath);

    if (rawLibPath) {
        options.pyteaLibPath = rawLibPath;
    } else if (!options.pyteaLibPath) {
        // default libpath should be bundled with pytea.js
        options.pyteaLibPath = getPyteaLibPath();
        if (!options.pyteaLibPath) options.pyteaLibPath = path.join(__dirname, 'pylib');
    } else {
        options.pyteaLibPath = normalizePath(combinePaths(dirPath, options.pyteaLibPath));
    }

    if (!fs.existsSync(options.pyteaLibPath)) {
        return `pytea library path '${options.pyteaLibPath}' does not exist`;
    }

    options = { ...baseOptions, ...options };

    // override by runtime node args
    if (args.logLevel !== undefined) options.logLevel = args.logLevel;
    if (args.extractIR !== undefined) options.extractIR = args.extractIR;

    return options as PyteaOptions;
}

// search every directory recursively and return every .py file paths (relative to projectPath)
// e.g.) ['LibCall.py', 'torch/__init__.py', ...]
export function getScriptRelPaths(projectPath: string, configOptions: ConfigOptions): string[] {
    const fileNames: string[] = [];
    const venvPath = configOptions.venvPath
        ? combinePaths(configOptions.projectRoot, configOptions.venvPath)
        : undefined;

    function iterDir(dirPath: string, prefix: string): void {
        fs.readdirSync(dirPath, { withFileTypes: true }).forEach((dirent) => {
            const fullPath = path.join(dirPath, dirent.name);
            const relPath = path.join(prefix, dirent.name);

            // ignore venv
            if (venvPath && fullPath === venvPath) {
                return;
            }
            if (dirent.isDirectory()) {
                // ignore venv
                if (fs.existsSync(path.join(fullPath, 'pyvenv.cfg'))) {
                    return;
                }
                iterDir(fullPath, relPath);
            } else if (dirent.isFile()) {
                if (path.extname(dirent.name) === '.py') {
                    fileNames.push(relPath);
                }
            }
        });
    }

    iterDir(projectPath, '');

    return fileNames;
}

// filePath should be relative to import base directory.
// e.g.) torch/functional.py => torch.functional
//       torch/__init__.py => torch
export function filePathToQualId(path: string): string {
    const dotPaths = getPathComponents(path)
        .filter((comp) => !['', '.', '..'].includes(comp))
        .join('.');

    if (dotPaths.endsWith('.py')) {
        return dotPaths.slice(0, -3);
    } else if (dotPaths.endsWith('.__init__.py')) {
        return dotPaths.slice(0, -12);
    }

    return dotPaths;
}

// return 'module qualPath => [file full path, ThStmt]'
// e.g.) "torch.functional" => <some statement>
export function getStmtsFromDir(service: AnalyzerService, dirPath: string): Map<string, [string, ThStmt]> {
    // Always enable "test mode".
    const parser = new TorchIRFrontend();
    const configOptions = service.getConfigOptions();

    const libFileNames = getScriptRelPaths(dirPath, configOptions);
    const libFilePaths = libFileNames.map((fn) => path.resolve(dirPath, fn));

    const program = service.backgroundAnalysisProgram.program;
    program.addTrackedFiles(libFilePaths);

    while (program.analyze()) {
        // Continue to call analyze until it completes. Since we're not
        // specifying a timeout, it should complete the first time.
    }

    // analyze single pytorch entry file
    const libMap: Map<string, [string, ThStmt]> = new Map();
    for (const fpId in libFilePaths) {
        const fp = libFilePaths[fpId];
        const fn = libFileNames[fpId];

        if (fp.endsWith('LibCall.py')) {
            continue;
        }

        const sourceFile = program.getSourceFile(fp);
        if (!sourceFile) {
            console.log(`Source file not found for ${fp}`);
            continue;
        }

        let stmt: ThStmt | undefined;
        try {
            const parseResult = service.getParseResult(fp);

            if (parseResult?.parseTree) {
                stmt = parser.translate(parseResult.parseTree);
            }
        } catch (e) {
            console.log(`Frontend parse failed: ${fp}\n${e}`);
            continue;
        }

        if (!stmt) {
            console.log(`library script parse error: ${fp}`);
        } else {
            libMap.set(filePathToQualId(fn), [fp, stmt]);
        }
    }

    return libMap;
}

// 'src.module.A' -> ['src', 'src.module', 'src.module.A']
// '..A.B' (from ..A import B) -> ['..', '..A', '..A.B']
// '.A.B', 'C.D' -> ['C', 'C.A', 'C.A.B']
// '..A', 'C.D.E' -> ['C', 'C.A']
export function scanQualPath(qualPath: string, currPath?: string): string[] {
    let leadingDots = 0;
    while (leadingDots < qualPath.length && qualPath[leadingDots] === '.') {
        leadingDots++;
    }

    const paths = qualPath.substr(leadingDots).split('.');
    for (let i = 0; i < paths.length - 1; i++) {
        paths[i + 1] = `${paths[i]}.${paths[i + 1]}`;
    }
    if (leadingDots > 0) {
        if (currPath === undefined) {
            const dots = '.'.repeat(leadingDots);
            return [dots, ...paths.map((p) => dots + p)];
        } else {
            const basePaths = scanQualPath(currPath);
            basePaths.splice(-leadingDots, leadingDots);

            if (basePaths.length === 0) {
                return paths;
            } else {
                const base = basePaths[basePaths.length - 1];
                paths.forEach((p) => basePaths.push(`${base}.${p}`));
                return basePaths;
            }
        }
    } else {
        return paths;
    }
}

export function runZ3Py<T>(result: ContextSet<T>): void {
    const pyteaPath = path.join(__dirname, 'z3wrapper', 'json2z3.py');

    if (!fs.existsSync(pyteaPath)) {
        console.log(`cannot found json2z3.py at '${pyteaPath}'. skip z3`);
        return;
    }

    const jsonList: string[] = [];
    result.getList().forEach((ctx) => {
        jsonList.push(ctx.ctrSet.getConstraintJSON());
    });
    result.getStopped().forEach((ctx) => {
        jsonList.push(ctx.ctrSet.getConstraintJSON());
    });

    if (jsonList.length === 0) {
        return;
    }

    const jsonStr = `[\n${jsonList.join(',\n')}\n]`;

    tmp.file((err, path) => {
        if (!err) {
            console.log(`save constraint json file to ${path}`);
            fs.writeFileSync(path, jsonStr);
            spawn('python', [pyteaPath, path]);
        }
    });
}

export function exportConstraintSet<T>(result: ContextSet<T>, path: string): void {
    const jsonList: string[] = [];
    result.getList().forEach((ctx) => {
        jsonList.push(ctx.ctrSet.getConstraintJSON());
    });
    result.getStopped().forEach((ctx) => {
        jsonList.push(ctx.ctrSet.getConstraintJSON());
    });

    const jsonStr = `[\n${jsonList.join(',\n')}\n]`;
    fs.writeFileSync(path, jsonStr);
}

// if value is address, return fetchAddr(value, heap)
// if that object has attr 'shape' and that is SVSize, return `Tensor ${value.size}`
export function reducedToString(value: ShValue, heap: ShHeap): string {
    const obj = fetchAddr(value, heap);
    if (obj) {
        if (obj.type === SVType.Object) {
            const shape = obj.getAttr('shape');
            if (shape instanceof SVSize) {
                return `Tensor ${SymExp.toString(shape.shape)}`;
            }
        }

        return obj.toString();
    } else {
        return value.toString();
    }
}

// make ParseNode to CodeRange
export namespace CodeSourcePositioner {
    // export function cleanContextSet<T>(ctxSet: ContextSet<T>, pathStore: FilePathStore): ContextSet<T> {
    //     return ctxSet.map((ctx) => cleanContext(ctx, pathStore));
    // }

    // export function cleanContext<T>(ctx: Context<T>, pathStore: FilePathStore): Context<T> {
    //     const { env, heap, ctrSet, callStack, logs, imported, failed, retVal } = ctx;

    //     ctx = ctx
    //         .set('env', cleanEnv(env, pathStore))
    //         .set('heap', cleanHeap(heap, pathStore))
    //         .set('ctrSet', cleanConstraintSet(ctrSet, pathStore))
    //         .set(
    //             'logs',
    //             logs.map((value) => cleanShValue(value, pathStore))
    //         )
    //         .set('imported', cleanEnv(imported, pathStore));

    //     ctx = ctx.set(
    //         'callStack',
    //         callStack.map(([func, source]) => [
    //             typeof func === 'string' ? func : cleanShValue(func, pathStore),
    //             cleanSource(source, pathStore),
    //         ]) as typeof callStack
    //     );

    //     if (failed) {
    //         ctx = ctx.set('failed', cleanShValue(failed, pathStore));
    //     }

    //     // force assume that source holder is Shvalue
    //     if ('source' in retVal) {
    //         ctx = (ctx.setRetVal(cleanShValue(retVal, pathStore)) as unknown) as Context<T>;
    //     }

    //     return ctx;
    // }

    // export function cleanEnv(env: ShEnv, pathStore: FilePathStore): ShEnv {
    //     return env.set(
    //         'addrMap',
    //         env.addrMap.map((addr) => cleanShValue(addr, pathStore))
    //     );
    // }

    // export function cleanHeap(heap: ShHeap, pathStore: FilePathStore): ShHeap {
    //     return heap.set(
    //         'valMap',
    //         heap.valMap.map((value) => cleanShValue(value, pathStore))
    //     );
    // }

    // export function cleanShValue(value: SVAddr, pathStore: FilePathStore): SVAddr;
    // export function cleanShValue(value: SVError, pathStore: FilePathStore): SVError;
    // export function cleanShValue(value: ShValue, pathStore: FilePathStore): ShValue {
    //     switch (value.type) {
    //         case SVType.Addr:
    //             return SVAddr.create(value.addr, cleanSource(value.source, pathStore));
    //         case SVType.Int:
    //             return SVInt.create(cleanSymExp(value.value, pathStore), cleanSource(value.source, pathStore));
    //         case SVType.Float:
    //             return SVFloat.create(cleanSymExp(value.value, pathStore), cleanSource(value.source, pathStore));
    //         case SVType.String:
    //             return SVString.create(cleanSymExp(value.value, pathStore), cleanSource(value.source, pathStore));
    //         case SVType.Bool:
    //             return SVBool.create(cleanSymExp(value.value, pathStore), cleanSource(value.source, pathStore));
    //         case SVType.Object: {
    //             const attrs = {
    //                 type: value.type,
    //                 id: value.id,
    //                 attrs: value.attrs.map(val => cleanShValue(val, pathStore)),
    //                 indices: value.indices.map(val => cleanShValue(val, pathStore)),
    //                 keyValues: value.keyValues.map(val => cleanShValue(val, pathStore)),
    //                 addr: cleanShValue(value.addr, pathStore),
    //                 shape: value.shape ? cleanSymExp(value.shape, pathStore) : value.shape,
    //                 source: cleanSource(value.source, pathStore)
    //             }

    //             if (value instanceof SVSize) {
    //                 return new SVSize(attrs)
    //             }
    //             return new SVObject(attrs)
    //         }
    //         case SVType.Func:
    //             return SVFunc.create(value.addr, cleanSource(value.source, pathStore));
    //         case SVType.None:
    //             return SVNone.create(value.addr, cleanSource(value.source, pathStore));
    //         case SVType.NotImpl:
    //             return SVNotImpl.create(value.addr, cleanSource(value.source, pathStore));
    //         case SVType.Undef:
    //             return SVUndef.create(value.addr, cleanSource(value.source, pathStore));
    //         case SVType.Error:
    //             return SVError.create(value.reason, value.level, cleanSource(value.source, pathStore));
    //         default:
    //             return value;
    //     }
    // }

    export function cleanConstraint(ctr: Constraint, pathStore: FilePathStore): Constraint {
        switch (ctr.type) {
            case ConstraintType.ExpBool:
                return {
                    ...ctr,
                    exp: cleanSymExp(ctr.exp, pathStore),
                    source: cleanSource(ctr.source, pathStore),
                } as CtrExpBool;
            case ConstraintType.Equal:
                return {
                    ...ctr,
                    left: cleanSymExp(ctr.left, pathStore),
                    right: cleanSymExp(ctr.right, pathStore),
                    source: cleanSource(ctr.source, pathStore),
                } as CtrEq;
            case ConstraintType.NotEqual:
                return {
                    ...ctr,
                    left: cleanSymExp(ctr.left, pathStore),
                    right: cleanSymExp(ctr.right, pathStore),
                    source: cleanSource(ctr.source, pathStore),
                } as CtrNeq;
            case ConstraintType.And:
                return {
                    ...ctr,
                    left: cleanConstraint(ctr.left, pathStore),
                    right: cleanConstraint(ctr.right, pathStore),
                    source: cleanSource(ctr.source, pathStore),
                } as CtrAnd;
            case ConstraintType.Or:
                return {
                    ...ctr,
                    left: cleanConstraint(ctr.left, pathStore),
                    right: cleanConstraint(ctr.right, pathStore),
                    source: cleanSource(ctr.source, pathStore),
                } as CtrOr;
            case ConstraintType.Not:
                return {
                    ...ctr,
                    constraint: cleanConstraint(ctr.constraint, pathStore),
                    source: cleanSource(ctr.source, pathStore),
                } as CtrNot;
            case ConstraintType.LessThan:
                return {
                    ...ctr,
                    left: cleanSymExp(ctr.left, pathStore),
                    right: cleanSymExp(ctr.right, pathStore),
                    source: cleanSource(ctr.source, pathStore),
                } as CtrLt;
            case ConstraintType.LessThanOrEqual:
                return {
                    ...ctr,
                    left: cleanSymExp(ctr.left, pathStore),
                    right: cleanSymExp(ctr.right, pathStore),
                    source: cleanSource(ctr.source, pathStore),
                } as CtrLte;
            case ConstraintType.Forall:
                return {
                    ...ctr,
                    symbol: cleanSymbol(ctr.symbol, pathStore),
                    range: [cleanSymExp(ctr.range[0], pathStore), cleanSymExp(ctr.range[1], pathStore)],
                    constraint: cleanConstraint(ctr.constraint, pathStore),
                    source: cleanSource(ctr.source, pathStore),
                } as CtrForall;
            case ConstraintType.Broadcastable:
                return {
                    ...ctr,
                    left: cleanSymExp(ctr.left, pathStore),
                    right: cleanSymExp(ctr.right, pathStore),
                    source: cleanSource(ctr.source, pathStore),
                } as CtrBroad;
            case ConstraintType.Fail:
                return {
                    ...ctr,
                    source: cleanSource(ctr.source, pathStore),
                } as CtrFail;
        }
    }

    export function cleanSymExp(exp: number | ExpNum, pathStore: FilePathStore): number | ExpNum;
    export function cleanSymExp(exp: boolean | ExpBool, pathStore: FilePathStore): boolean | ExpBool;
    export function cleanSymExp(exp: string | ExpString, pathStore: FilePathStore): string | ExpString;
    export function cleanSymExp(exp: ExpShape, pathStore: FilePathStore): ExpShape;
    export function cleanSymExp(exp: SymExp, pathStore: FilePathStore): SymExp;
    export function cleanSymExp(
        exp: number | boolean | string | SymExp,
        pathStore: FilePathStore
    ): number | boolean | string | SymExp {
        if (typeof exp === 'object') {
            // force cast
            const retVal: any = { ...exp };

            // recursively remove source
            Object.entries(retVal).forEach(([key, value]) => {
                if (key === 'source') {
                    retVal.source = cleanSource(retVal.source, pathStore);
                }
                if (Array.isArray(value)) {
                    retVal[key] = value.map((v) => cleanSymExp(v, pathStore));
                } else if (typeof value === 'object') {
                    retVal[key] = cleanSymExp(value as SymExp, pathStore);
                }
            });

            return retVal;
        }

        return exp;
    }

    export function cleanSymbol(symbol: SymInt, pathStore: FilePathStore): SymInt;
    export function cleanSymbol(symbol: SymVal, pathStore: FilePathStore): SymVal {
        const source = cleanSource(symbol.source, pathStore);
        return { ...symbol, source };
    }

    export function cleanSource(source: CodeSource | undefined, pathStore: FilePathStore): CodeRange | undefined {
        if (!source) return source;
        if ('fileId' in source) return source;

        let moduleNode = source;
        while (moduleNode.nodeType !== ParseNodeType.Module) {
            moduleNode = moduleNode.parent!;
        }

        const fileInfo = getFileInfo(moduleNode)!;

        const filePath = fileInfo.filePath;
        const lines = fileInfo.lines;
        const start = convertOffsetToPosition(source.start, lines);
        const end = convertOffsetToPosition(source.start + source.length, lines);
        const fileId = pathStore.addPath(filePath);

        return { fileId, range: { start, end } };
    }
}
