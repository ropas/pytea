/*
 * pyteaOptions.ts
 * Copyright (c) Seoul National University
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo (starvessel@naver.com)
 *
 * options for PyTea analyzer.
 */

export type PyCmdArgs = { [option: string]: boolean | number | string };
export type PyteaLogLevel = 'none' | 'result-only' | 'reduced' | 'full';
export interface PyteaOptions {
    // Absolute path to pyteaconfig.json
    configPath: string;

    // Path to PyTea implementation of Python builtins.
    // Absolute or relative to configPath.
    pyteaLibPath: string;

    // Python entry point. Absolute or relative to configPath.
    entryPath: string;

    // Python command line arguments. Injected to Python argparse methods.
    pythonCmdArgs: PyCmdArgs;

    // 'dest' value of argparse.add_subparsers.
    pythonSubcommand: string;

    // Severity of analysis result. (default: reduced)
    logLevel: PyteaLogLevel;

    // Discard trivial constraints. (default: true)
    immediateConstraintCheck: boolean;

    // Ignore assert statements of Python. (default: false)
    ignoreAssert: boolean;

    // Extract Internal Representation of entry script. (TorchIR)
    extractIR: boolean;

    // Explicit range of random variables.
    // Key should be the prefix of specific random variable.
    // If the name of random variable is 'PILImgC_I3', the key should be "PILImgC".
    // (the suffix '_I3' means third immediate random variable)
    // The range of random varaible which name starts with prefix will be altered to this.
    // Range is always inclusive. null means the range is unbounded (or half-bounded).
    variableRange: { [prefix: string]: null | number | [number | null, number | null] };

    // Assign random concrete value to some random variable by seed (if set).
    // Key should be the prefix of specific random variable (see above 'variableRange')
    // null means the seed will be set from runtime of analyzer.
    // If variableRange is not set to this prefix, the default range will be
    // [1, 10000] (int) or [0.0, 1.0] (float).
    variableSeed: { [prefix: string]: number | null };

    // Iterate torch DataLoader only once. (default: true)
    // If it is set to false, large dataset will give its all data (e.g. MNIST gives 60000 items)
    boxDataLoader: boolean;

    // Analyzer timeout in millisecond. undefined means no timeout (default: no timeout)
    timeout: number | null | undefined;

    // Set max path count, throw runtime error if path count exceeds it (default: 1000)
    maxPath: number | null | undefined;

    // Port to Python Z3 server (default: 17851)
    z3Port: number;
}

export const defaultOptions: PyteaOptions = {
    configPath: '',
    pyteaLibPath: '',
    entryPath: '',
    pythonCmdArgs: {},
    pythonSubcommand: '',
    logLevel: 'reduced',
    immediateConstraintCheck: true,
    ignoreAssert: false,
    extractIR: false,
    variableRange: {},
    variableSeed: {},
    z3Port: 17851,
    timeout: undefined,
    maxPath: undefined,
    boxDataLoader: true,
};
