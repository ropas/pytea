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
import { combinePaths } from 'pyright-internal/common/pathUtils';
import { PyrightFileSystem } from 'pyright-internal/pyrightFileSystem';
import { createFromRealFileSystem } from 'pyright-internal/common/realFileSystem';

import { defaultOptions } from './service/pyteaOptions';
import { PyteaService } from './service/pyteaService';
import { buildPyteaOption, exportConstraintSet } from './service/pyteaUtils';

const toolName = 'pytea';

enum ExitStatus {
    NoErrors = 0,
    ErrorsReported = 1,
    FatalError = 2,
    ConfigFileParseError = 3,
}

function parsePyrightArgs(): CommandLineOptions | undefined {
    const optionDefinitions: OptionDefinition[] = [
        { name: 'file', type: String, defaultOption: true, defaultValue: '' },
        { name: 'help', alias: 'h', type: Boolean },
        { name: 'extractIR', alias: 'e', type: Boolean },
        { name: 'libPath', alias: 'l', type: String },
        { name: 'configPath', type: String, defaultValue: '' },
        { name: 'resultPath', type: String },
        { name: 'pythonArgs', alias: 'a', type: String },
        { name: 'logLevel', type: String },
        { name: 'verbose', type: Boolean, defaultValue: false },
        { name: 'version', type: Boolean },
        { name: 'z3', type: Boolean, defaultValue: false },
        { name: 'timeout', type: Number },
        { name: 'maxPath', type: Number },
    ];

    let args: CommandLineOptions;

    try {
        args = commandLineArgs(optionDefinitions);
    } catch (err) {
        const argErr: { optionName: string } = err as { optionName: string };
        if (argErr && argErr.optionName) {
            console.error(`Unexpected option ${argErr.optionName}.\n${toolName} --help for usage`);
            return;
        }

        console.error(`Unexpected error while parsing command line options.\n${toolName} --help for usage`);
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

    if (args.resultPath === undefined) {
        if (args.extractIR) {
            args.resultPath = 'out_ir.lisp';
        } else {
            args.resultPath = 'constraint.json';
        }
    }

    return args;
}

export function getPyteaService(args: CommandLineOptions, service?: AnalyzerService): PyteaService | undefined {
    const entryPath = args.file;

    if (!entryPath) {
        printUsage();
        return;
    }

    const pyteaOptions = buildPyteaOption(args, undefined, defaultOptions);

    if (typeof pyteaOptions === 'string') {
        console.error(pyteaOptions);
        process.exit(-1);
    }

    const logger = new StandardConsole();
    const pyteaService = new PyteaService(service, pyteaOptions, logger, true);

    return pyteaService;
}

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

    options.checkOnlyOpenFiles = false;

    // ignore original pyright output.
    const output = new NullConsole();
    const fileSystem = new PyrightFileSystem(createFromRealFileSystem(output));

    const runZ3 = args.z3;
    options.watchForSourceChanges = false;

    const pyrightService = new AnalyzerService('<default>', fileSystem, output);
    let pyteaService: PyteaService | undefined;

    const resultPath = combinePaths(process.cwd(), args.resultPath);

    pyrightService.setCompletionCallback(async (results) => {
        if (results.fatalErrorOccurred) {
            process.exit(ExitStatus.FatalError);
        }

        if (results.configParseErrorOccurred) {
            process.exit(ExitStatus.ConfigFileParseError);
        }

        if (!pyteaService) {
            pyteaService = getPyteaService(args, pyrightService);
        }

        if (pyteaService && pyteaService.options) {
            const entryPath = pyteaService.options.entryPath;

            // this triggers translation of project folder.
            const errMsg = pyteaService.translateAll(entryPath);
            if (errMsg) {
                console.error(errMsg);
                process.exit(ExitStatus.FatalError);
            }

            if (!pyteaService.validate()) {
                console.error('pytea service got error');
                process.exit(ExitStatus.FatalError);
            } else {
                // do pytea job
                if (pyteaService.options.extractIR) {
                    pyteaService.extractIR(resultPath);
                    console.log(`IR has extracted to ${resultPath}`);
                    process.exit(ExitStatus.NoErrors);
                }

                try {
                    const result = await pyteaService.analyze();
                    if (pyteaService && result) {
                        if (runZ3) {
                            const z3Result = await pyteaService.runZ3Py(result);
                            pyteaService.printLogWithZ3(result, z3Result);
                        } else {
                            pyteaService!.printLog(result);
                            exportConstraintSet(result, resultPath);
                        }
                    }
                } catch (e) {
                    console.error(e);
                    process.exit(ExitStatus.FatalError);
                }
            }
        } else {
            console.error('pytea service is not initialized');
        }

        process.exit(ExitStatus.NoErrors);
    });

    // This will trigger the pyright analyzer.
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
            '  -h,--help                       Show this help message\n' +
            '  -e,--extractIR                  Run the parser only and extract\n' +
            '                                      internal representations of Python scripts\n' +
            '  -l,--libPath                    Path to the PyTea Python library implementations\n' +
            '  --configPath                    Path to pyteaconfig.json\n' +
            '  --resultPath                    Path to save the result (constraint json or extracted IR)\n' +
            '  --logLevel                      Verbosity of log (none, result-only, reduced, full)\n' +
            '  --z3                            Interact with local Z3Py server (need to run pyteaserver.py first)\n' +
            '  --z3Port                        Port to Z3Py server (default: 17851)\n' +
            '  --extract                       Extract PyTea IR in S-Expression format\n' +
            '  --timeout                       Set timeout in miliseconds\n' +
            '  --verbose                       Emit verbose Pyright (parser) diagnostics\n' +
            '  --version                       Print version\n'
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
