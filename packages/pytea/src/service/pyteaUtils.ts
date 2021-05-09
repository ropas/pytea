/*
 * pyteaUtils.ts
 * Copyright (c) Seoul National University.
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo (mersshs@gmail.com)
 *
 * Utility functions for PyTea service.
 */
import axios from 'axios';
import { CommandLineOptions } from 'command-line-args';
import * as fs from 'fs';
import * as path from 'path';
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
import { ParseNode, ParseNodeType } from 'pyright-internal/parser/parseNodes';

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
import { CodeRange, CodeSource, ShValue, SVObject, SVSize, SVType } from '../backend/sharpValues';
import { ExpBool, ExpNum, ExpShape, ExpString, SymExp, SymInt, SymVal } from '../backend/symExpressions';
import { TorchIRFrontend } from '../frontend/torchFrontend';
import { ThStmt } from '../frontend/torchStatements';
import { FilePathStore } from './executionPaths';
import { PyteaOptions } from './pyteaOptions';

export const enum PyZ3RPCResultType {
    Unreachable = 0,
    Valid = 1,
    Sat = 2,
    Unsat = 3,
    DontKnow = 4,
    Timeout = -1,
}

export interface PyZ3RPCResult {
    type: PyZ3RPCResultType;
    extras?: {
        conflict?: number; // first index of conflicted constraint (type == Invalid)
        undecide?: number; // first index of undecidable constraint (type == Undecidable)
    };
}
export interface PyZ3RPCRespond {
    jsonrpc: 2.0;
    id: number;
    result: PyZ3RPCResult[];
    log: string;
}

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

    // found config by directory, but entryPath is not set
    if (isDir && configPath && fs.existsSync(configPath) && !options.entryPath) {
        return `'entryPath' is not set from '${configPath}'`;
    }

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

export type CustomObjectPrinter = [ShValue, (value: SVObject) => string];
// if value is address, return fetchAddr(value, heap)
// if that object has attr 'shape' and that is SVSize, return `Tensor ${value.size}`
// also reduce some classes with 'customObjectPrinter', list of pairs of class mro and printer.
export function reducedToString(
    value: ShValue,
    heap: ShHeap,
    customObjectPrinter?: CustomObjectPrinter[],
    attrMax?: number
): string {
    attrMax = attrMax ?? 8;

    const obj = fetchAddr(value, heap);
    if (obj) {
        if (obj.type === SVType.Object) {
            if (obj instanceof SVSize) {
                return `SVSize(${SymExp.toString(obj.shape)})`;
            }

            const shape = fetchAddr(obj.getAttr('shape'), heap);
            if (shape && shape instanceof SVSize) {
                return `tensor: ${SymExp.toString(shape.shape)}`;
            }

            const mro = fetchAddr(obj.getAttr('__mro__'), heap);
            if (customObjectPrinter && mro) {
                for (const [pmro, printer] of customObjectPrinter) {
                    if (pmro.equals(mro)) {
                        return printer(obj);
                    }
                }
            }

            const attrStr =
                obj.attrs.count() > attrMax ? `<${obj.attrs.count()} attrs>` : `${ShValue.toStringStrMap(obj.attrs)}`;
            const indStr =
                obj.indices.count() > attrMax
                    ? `<${obj.indices.count()} indexed values>`
                    : `${ShValue.toStringNumMap(obj.indices)}`;
            const kvStr =
                obj.keyValues.count() > attrMax
                    ? `<${obj.keyValues.count()} keyed values>`
                    : `${ShValue.toStringStrMap(obj.keyValues)}`;
            const shapeStr = obj.shape ? `, ${ExpShape.toString(obj.shape)}` : '';
            return `[${obj.addr.addr}]{ ${attrStr}, ${indStr}, ${kvStr}${shapeStr} }`;
        }

        return obj.toString();
    } else {
        return value.toString();
    }
}

// make ParseNode to CodeRange
export namespace CodeSourcePositioner {
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

export function formatParseNodeRange(node: ParseNode): string {
    let moduleNode = node;
    while (moduleNode.nodeType !== ParseNodeType.Module) {
        moduleNode = moduleNode.parent!;
    }

    const fileInfo = getFileInfo(moduleNode)!;

    const filePath = fileInfo.filePath;
    const lines = fileInfo.lines;
    const start = convertOffsetToPosition(node.start, lines);
    const end = convertOffsetToPosition(node.start + node.length, lines);

    const location = `${start.line + 1}:${start.character}`;
    return `[${location} - ${end.line + 1}:${end.character}] (${filePath}:${location})`;
}

export function formatCodeSource(node?: CodeSource, pathStore?: FilePathStore): string {
    if (!node) return 'internal';

    // check ParseNode or not
    if (pathStore) {
        const range = pathStore.toCodeRange(node);
        if (!range) return 'internal';

        const filePath = pathStore.getPath(range.fileId);
        const { start, end } = range.range;
        const location = `${start.line + 1}:${start.character}`;
        return `[${location} - ${end.line + 1}:${end.character}] (${filePath}:${location})`;
    }

    if (!('fileId' in node)) {
        return formatParseNodeRange(node);
    } else {
        const { start, end } = node.range;
        return `[${start.line + 1}:${start.character} - ${end.line + 1}:${end.character}] (file ${node.fileId})`;
    }
}

export async function postJsonRpc(address: string, id: number, method: string, params: any) {
    const respond = await axios.post(
        address,
        {
            jsonrpc: '2.0',
            id,
            method,
            params,
        },
        {
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
            },
        }
    );

    return respond.data.result;
}
