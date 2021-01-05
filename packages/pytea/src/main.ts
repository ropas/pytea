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

function parsePyrightArgs(): CommandLineOptions | undefined {
    const optionDefinitions: OptionDefinition[] = [
        { name: 'file', type: String, defaultOption: true },
        { name: 'help', alias: 'h', type: Boolean },
        { name: 'extract-ir', alias: 'p', type: Boolean },
        { name: 'lib-path', alias: 'l', type: String },
        { name: 'config-path', type: String },
        { name: 'python-args', alias: 'a', type: String },
        { name: 'log-level', type: String },
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

    return args;
}

export function getPyteaService(args: CommandLineOptions): PyteaService | undefined {
    const entryPath = args.file;

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

// export function runPytea(entryPath: string): PyteaService | undefined {
//     const service = getPyteaService(entryPath);
//     service?.startAnalyzer([entryPath]);

//     return service;
// }

function runMain(args: CommandLineOptions) {
    const options = new PyrightCommandLineOptions(process.cwd(), false);

    // Assume any relative paths are relative to the working directory.
    if (args.file) {
        args.file = combinePaths(process.cwd(), args.file);
        options.fileSpecs = [args.file];
    } else {
        console.error('Python script path is not set.');
        process.exit(ExitStatus.FatalError);
    }

    if (args.project) {
        options.configFilePath = combinePaths(process.cwd(), normalizePath(args.project));
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
            pyteaService = getPyteaService(args);
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
                // do pytea job
                try {
                    const result = pyteaService.analyze();
                    if (result) pyteaService.printLog(result);
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
            ' [options] file\n' +
            '  Options:\n' +
            '  -h,--help                        Show this help message\n' +
            '  -e,--extract-ir                  Run only Frontend and Extract\n' +
            '                                       internal representations of Python scripts\n' +
            '  -a,--python-args                 command line arguments for main Python script\n' +
            '  -l,--lib-path                    Path to PyTea Python library implementations\n' +
            '  --config-path                    Path to pyteaconfig.json\n' +
            '  --log-level                      Verbosity of log (none, result-only, reduced, full)\n' +
            '  --verbose                        Emit Pyright verbose diagnostics\n' +
            '  --version                        Print PyTea version\n' +
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

    const args = parsePyrightArgs();
    if (args) runMain(args);
}
