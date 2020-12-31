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
    // arguments is given as key "$defaults"
    pythonCmdArgs: PyCmdArgs;

    // Debug log level (default: reduced)
    logLevel: 'none' | 'result_only' | 'reduced' | 'full';

    // Check and discard trivial constraints (default: true)
    immediateConstraintCheck: boolean;

    // Ignore assert statements of Python. (default: false)
    ignoreAssert: boolean;

    // Print internal representation (TorchIR)
    printIR: boolean;
}

export type PyteaOptionsPart = Partial<PyteaOptions>;

export const defaultOptions: PyteaOptionsPart = {
    pythonCmdArgs: {},
    logLevel: 'reduced',
    immediateConstraintCheck: true,
    ignoreAssert: false,
    printIR: false,
};
