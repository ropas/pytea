/*
 * pyteaOptions.ts
 * Copyright (c) Seoul National University
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo (starvessel@naver.com)
 *
 * options for PyTea analyzer.
 */

export type PyCmdArgs = { [option: string]: boolean | number | string };
export interface PyteaOptions {
    // Absolute path to pyteaconfig.json
    configPath: string;

    // Path of PyTea implementation of Python builtins.
    // Absolute or relative to configPath
    pyteaLibPath: string;

    // Python entry point. Absolute or relative to configPath
    entryPath: string;

    // Python command line arguments.
    pythonCmdArgs: PyCmdArgs;

    // 'dest' of argparse.add_subparsers(...)
    pythonSubcommand: string;

    // Debug log level (default: reduced)
    logLevel: 'none' | 'result-only' | 'reduced' | 'full';

    // Check and discard trivial constraints (default: true)
    immediateConstraintCheck: boolean;

    // Ignore assert statements of Python. (default: false)
    ignoreAssert: boolean;

    // Extract internal representation to file (TorchIR)
    extractIR: boolean;

    // Explicit range of random variables.
    // The range of random varaible which name starts with prefix will be altered to this.
    // null means unbounded, range is inclusive.
    variableRange: { [prefix: string]: null | number | [number | null, number | null] };
}

export type PyteaOptionsPart = Partial<PyteaOptions>;

export const defaultOptions: PyteaOptionsPart = {
    pythonCmdArgs: {},
    pythonSubcommand: '',
    logLevel: 'reduced',
    immediateConstraintCheck: true,
    ignoreAssert: false,
    extractIR: false,
    variableRange: {},
};
