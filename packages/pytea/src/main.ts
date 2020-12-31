/*
 * pytea.ts (modified from pyright.ts v1.1.95)
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 * Author: Eric Traut
 *
 * Command-line entry point for pytea analyzer.
 */

// Add the start timer at the very top of the file, before we import other modules.

/* eslint-disable */
import commandLineArgs, { CommandLineOptions, OptionDefinition } from 'command-line-args';
/* eslint-enable */
import * as process from 'process';

import { AnalyzerService } from 'pyright-internal/analyzer/service';
import { CommandLineOptions as PyrightCommandLineOptions } from 'pyright-internal/common/commandLineOptions';
import { NullConsole, StandardConsole } from 'pyright-internal/common/console';

import { createFromRealFileSystem } from 'pyright-internal/common/fileSystem';
import { combinePaths, normalizePath } from 'pyright-internal/common/pathUtils';

import { PyteaService } from './service/pyteaService';
import { makeOptionParts } from './service/pyteaUtils';

const toolName = 'pytea';

enum ExitStatus {
    NoErrors = 0,
    ErrorsReported = 1,
    FatalError = 2,
    ConfigFileParseError = 3,
}

function processArgs(): CommandLineOptions | undefined {
    const optionDefinitions: OptionDefinition[] = [
        { name: 'createstub', type: String },
        { name: 'dependencies', type: Boolean },
        { name: 'files', type: String, multiple: true, defaultOption: true },
        { name: 'help', alias: 'h', type: Boolean },
        { name: 'lib', type: Boolean },
        { name: 'outputjson', type: Boolean },
        { name: 'project', alias: 'p', type: String },
        { name: 'stats' },
        { name: 'typeshed-path', alias: 't', type: String },
        { name: 'venv-path', alias: 'v', type: String },
        { name: 'verifytypes', type: String },
        { name: 'verbose', type: Boolean },
        { name: 'version', type: Boolean },
        { name: 'watch', alias: 'w', type: Boolean },
    ];

    let args: CommandLineOptions;

    try {
        args = commandLineArgs(optionDefinitions);
    } catch (err) {
        const argErr: { name: string; optionName: string } = err;
        if (argErr && argErr.optionName) {
            console.error(`Unexpected option ${argErr.optionName}.\n${toolName} --help for usage`);
            return;
        }

        console.error(`Unexpected error\n${toolName} --help for usage`);
        return;
    }

    if (args.help !== undefined) {
        printUsage();
        return;
    }

    if (args.version !== undefined) {
        printVersion();
        return;
    }

    if (args.outputjson) {
        const incompatibleArgs = ['watch', 'stats', 'verbose', 'createstub', 'dependencies'];
        for (const arg of incompatibleArgs) {
            if (args[arg] !== undefined) {
                console.error(`'outputjson' option cannot be used with '${arg}' option`);
                return;
            }
        }
    }

    if (args['verifytypes'] !== undefined) {
        const incompatibleArgs = ['watch', 'stats', 'createstub', 'dependencies'];
        for (const arg of incompatibleArgs) {
            if (args[arg] !== undefined) {
                console.error(`'verifytypes' option cannot be used with '${arg}' option`);
                return;
            }
        }
    }

    if (args.createstub) {
        const incompatibleArgs = ['watch', 'stats', 'verifytypes', 'dependencies'];
        for (const arg of incompatibleArgs) {
            if (args[arg] !== undefined) {
                console.error(`'createstub' option cannot be used with '${arg}' option`);
                return;
            }
        }
    }

    return args;
}

export function getPyteaService(entryPath: string): PyteaService | undefined {
    if (!entryPath) {
        printUsage();
        return;
    }

    // const entryName = './test/scratch.py';
    const pyteaOptions = makeOptionParts(entryPath);
    if (typeof pyteaOptions === 'string') {
        console.error(pyteaOptions);
        return;
    }

    const logger = new StandardConsole();
    const pyteaService = new PyteaService(pyteaOptions, logger, true);

    // options not set, has error.
    if (!pyteaService.options) {
        return;
    }

    return pyteaService;
}

export function runPytea(entryPath: string): PyteaService | undefined {
    const service = getPyteaService(entryPath);
    service?.startAnalyzer([entryPath]);

    return service;
}

function runMain(args: CommandLineOptions) {
    const options = new PyrightCommandLineOptions(process.cwd(), false);

    // Assume any relative paths are relative to the working directory.
    if (args.files && Array.isArray(args.files)) {
        options.fileSpecs = args.files;
        options.fileSpecs = options.fileSpecs.map((f) => combinePaths(process.cwd(), f));
    } else {
        console.error('entry path is not set.');
        process.exit(ExitStatus.FatalError);
    }

    if (args.project) {
        options.configFilePath = combinePaths(process.cwd(), normalizePath(args.project));
    }

    if (args['venv-path']) {
        options.venvPath = combinePaths(process.cwd(), normalizePath(args['venv-path']));
    }

    if (args['typeshed-path']) {
        options.typeshedPath = combinePaths(process.cwd(), normalizePath(args['typeshed-path']));
    }

    if (args.createstub) {
        options.typeStubTargetImportName = args.createstub;
    }

    if (args.verbose) {
        options.verboseOutput = true;
    }
    if (args.lib) {
        options.useLibraryCodeForTypes = true;
    }
    options.checkOnlyOpenFiles = false;

    // ignore original pyright output.
    // const output = args.outputjson ? new NullConsole() : undefined;
    const output = new NullConsole();
    const realFileSystem = createFromRealFileSystem(output);
    const entryPath = options.fileSpecs[0] as string | undefined;

    const watch = args.watch !== undefined;
    options.watchForSourceChanges = watch;

    const pyrightService = new AnalyzerService('<default>', realFileSystem, output);
    let pyteaService: PyteaService | undefined;

    pyrightService.setCompletionCallback((results) => {
        if (results.fatalErrorOccurred) {
            process.exit(ExitStatus.FatalError);
        }

        if (results.configParseErrorOccurred) {
            process.exit(ExitStatus.ConfigFileParseError);
        }

        if (!pyteaService && entryPath) {
            pyteaService = getPyteaService(entryPath);
            pyteaService?.setAnalyzerService(pyrightService);
        }

        if (pyteaService && pyteaService.options) {
            const entryPath = pyteaService.options.entryPath;

            // this triggers project folder parsing.
            pyteaService.parseEntry(entryPath);

            if (!pyteaService.validate()) {
                console.error('pytea service got error');
                process.exit(ExitStatus.FatalError);
            } else {
                // de pytea job
                try {
                    pyteaService.checkWithLog();
                } catch (e) {
                    console.error(e);
                    process.exit(ExitStatus.FatalError);
                }
            }
        } else {
            console.error('pytea service is not initialized');
        }

        if (!watch) {
            process.exit(ExitStatus.NoErrors);
            // process.exit(errorCount > 0 ? ExitStatus.ErrorsReported : ExitStatus.NoErrors);
        } else if (!args.outputjson) {
            console.error('Watching for file changes...');
        }
    });

    // This will trigger the analyzer.
    pyrightService.setOptions(options);

    const brokenPromise = new Promise(() => {
        // Do nothing.
    });
    brokenPromise.then().catch();
}

function printUsage() {
    console.log(
        'Usage: ' +
            toolName +
            ' [options] files...\n' +
            '  Options:\n' +
            '  --createstub IMPORT              Create type stub file(s) for import\n' +
            '  --dependencies                   Emit import dependency information\n' +
            '  -h,--help                        Show this help message\n' +
            '  --lib                            Use library code to infer types when stubs are missing\n' +
            '  --outputjson                     Output results in JSON format\n' +
            '  -p,--project FILE OR DIRECTORY   Use the configuration file at this location\n' +
            '  --stats                          Print detailed performance stats\n' +
            '  -t,--typeshed-path DIRECTORY     Use typeshed type stubs at this location\n' +
            '  -v,--venv-path DIRECTORY         Directory that contains virtual environments\n' +
            '  --verbose                        Emit verbose diagnostics\n' +
            '  --version                        Print PyTEA version\n' +
            '  -w,--watch                       Continue to run and watch for changes\n'
    );
}

function getVersionString() {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const version = require('package.json').version;
    return version.toString();
}

function printVersion() {
    console.log(`${toolName} ${getVersionString()}`);
}

export function main() {
    if (process.env.NODE_ENV === 'production') {
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        require('source-map-support').install();
    }

    const args = processArgs();
    if (args) runMain(args);
}
