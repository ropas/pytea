/*
 * pytOptions.ts
 * Copyright (c) Seoul National University
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo
 *
 * options for PyTea analyzer.
 */

import { CommandLineOptions } from '../common/commandLineOptions';

export interface PytOptions {
    // Absolute path to pyrightconfig.json
    configPath: string;

    // Path of PyTea implementation of Python builtins.
    // Absolute or relative to configPath
    pytLibPath: string;

    // Python entry point. Absolute or relative to configPath
    entryPath: string;

    // Python command line arguments.
    // arguments is given as key "$defaults"
    pythonCmdArgs: { [option: string]: boolean | number | string };

    // Debug log level (default: reduced)
    logLevel: 'none' | 'result_only' | 'reduced' | 'full';

    // Check and discard trivial constraints (default: true)
    immediateConstraintCheck: boolean;

    // Ignore assert statements of Python. (default: false)
    ignoreAssert: boolean;

    // Print internal representation (TorchIR)
    printIR: boolean;
}

export type PytOptionsPart = Partial<PytOptions>;

export const defaultOptions: PytOptionsPart = {
    logLevel: 'reduced',
    immediateConstraintCheck: true,
    ignoreAssert: false,
    printIR: false,
};
