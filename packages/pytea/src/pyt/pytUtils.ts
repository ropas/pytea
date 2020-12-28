/*
 * pytUtils.ts
 * Copyright (c) Seoul National University.
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo
 *
 * Utility functions for PyTea service.
 */
import * as fs from 'fs';
import * as path from 'path';
import * as util from 'util';

import { AnalyzerService } from 'pyright-internal/analyzer/service';
import { ConfigOptions } from 'pyright-internal/common/configOptions';
import { ConsoleInterface } from 'pyright-internal/common/console';
import { combinePaths, getPathComponents, normalizePath } from 'pyright-internal/common/pathUtils';
import { ParseNode } from 'pyright-internal/parser/parseNodes';

import { TorchIRFrontend } from '../frontend/torchFrontend';
import { ThStmt } from '../frontend/torchStatements';
import { defaultOptions, PytOptions, PytOptionsPart } from './pytOptions';

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

export function makeOptionParts(entryName: string): PytOptionsPart | string {
    const cwd = path.normalize(process.cwd());
    const entryPath = normalizePath(combinePaths(cwd, entryName));
    let isDir = false;

    if (!fs.existsSync(entryPath)) {
        return `entry file ${entryPath} does not exists.`;
    } else if (fs.lstatSync(entryPath).isDirectory()) {
        // console.log(`setting project path: ${entryPath}`);
        isDir = true;
    } else {
        // console.log(`setting entry path: ${entryPath}`);
    }

    const dirPath = isDir ? entryPath : path.dirname(entryPath);
    const configPath = path.join(dirPath, 'pyteaconfig.json');

    if (!fs.existsSync(configPath)) {
        return `config json ${configPath} does not exists.`;
    }

    const pytOptions: PytOptionsPart = isDir ? { configPath } : { configPath, entryPath };

    return pytOptions;
}

// make paths in options absolute
export function refineOptions(options: PytOptionsPart): PytOptions {
    let opt: PytOptionsPart = { ...defaultOptions, ...options };
    const configPath = opt.configPath;

    let entryPath = opt.entryPath;
    let basePath = '';

    if (configPath && fs.existsSync(configPath)) {
        basePath = path.dirname(configPath);
        try {
            const configJSON: Partial<PytOptions> = JSON.parse(fs.readFileSync(configPath).toString());
            opt = { ...opt, ...configJSON };
        } catch (e) {
            throw `${configPath} is not a valid JSON file`;
        }
    } else if (entryPath && fs.existsSync(entryPath)) {
        basePath = path.dirname(entryPath);
    }

    entryPath = opt.entryPath;

    if (!basePath) {
        throw `either configPath${configPath ? `(${configPath})` : ''} or entryPath${
            entryPath ? `(${entryPath})` : ''
        } is not found`;
    }

    if (!opt.pytLibPath) {
        // throw 'pytLibPath is not set';
        opt.pytLibPath = path.join(__dirname, 'pylib');
    }

    if (entryPath && !path.isAbsolute(entryPath)) opt.entryPath = path.join(basePath, entryPath);
    if (!path.isAbsolute(opt.pytLibPath)) opt.pytLibPath = path.join(basePath, opt.pytLibPath);

    if (opt.entryPath && !fs.existsSync(opt.entryPath)) {
        throw `cannot find entryPath ${opt.entryPath}`;
    } else if (!fs.existsSync(opt.pytLibPath)) {
        throw `cannot find pytLibPath ${opt.pytLibPath}`;
    }

    return opt as PytOptions;
}

// return every .py filenames
// e.g.) ['LibCall.py', 'torch/__init__.py', ...]
export function getTorchLibFileNames(baseDirPath: string, configOptions: ConfigOptions): string[] {
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

    iterDir(baseDirPath, '');

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

// return module qualPath => ThStmt
// e.g.) "torch.functional" => <some statement>
export function getTorchStmtsFromDir(service: AnalyzerService, dirPath: string): Map<string, ThStmt> {
    // Always enable "test mode".
    const parser = new TorchIRFrontend();
    const configOptions = service.getConfigOptions();

    const libFileNames = getTorchLibFileNames(dirPath, configOptions);
    const libFilePaths = libFileNames.map((fn) => path.resolve(dirPath, fn));

    const program = service.backgroundAnalysisProgram.program;
    program.setTrackedFiles(libFilePaths);

    while (program.analyze()) {
        // Continue to call analyze until it completes. Since we're not
        // specifying a timeout, it should complete the first time.
    }

    // analyze single pytorch entry file
    const libMap: Map<string, ThStmt> = new Map();
    for (const fpId in libFilePaths) {
        const fp = libFilePaths[fpId];
        const fn = libFileNames[fpId];

        const sourceFile = program.getSourceFile(fp);
        if (!sourceFile) {
            console.log(`Source file not found for ${fp}`);
            continue;
        }

        let stmt: ThStmt | undefined;
        try {
            const parseResult = service.getParseResult(fp);

            if (parseResult?.parseTree) {
                stmt = parser.parse(parseResult.parseTree);
            }
        } catch (e) {
            console.log(`Frontend parse failed: ${fp}\n${e}`);
            continue;
        }

        if (!stmt) {
            console.log(`library script parse error: ${fp}`);
        } else {
            libMap.set(filePathToQualId(fn), stmt);
        }
    }

    return libMap;
}

export function nodePosToString(node?: ParseNode): string {
    if (!node) {
        return 'internal';
    }

    return `char index [${node.start}:${node.start + node.length}]`;
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
