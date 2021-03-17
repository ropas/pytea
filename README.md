# PyTEA: Python Tensor Error Analyzer

## Requirements

- `node.js >= 12.x`
- `python >= 3.8`
  - `z3-solver >= 4.8`

## Usage

### Build and Analyze

```bash
# install dependencies
npm run install:all
pip install z3-solver

# build node frontend
npm run build

# Analyze file
python bin/pytea.py path/to/source.py
```

### Debug

```bash
cd packages/pytea
npm run webpack

# run frontend only (run without z3)
node index.js path/to/source.py --logLevel=reduced
```

We add two debug options from VSCode Debug panel.
- `Pytea CLI`: Build and run pytea frontend using `packages/pytea/pytest/basics/pyteaconfig.json`. That config file should define `entryPath` option.
- `Pytea CLI scratch`: Scratchpad debugging. Does not build pytea config, but analyze `packages/pytea/pytest/basics/scratch.py`. User should build pytea package before to run in.

## Config (pyteaconfig.json)

We use `pyteaconfig.json` to handle options for Pytea, following `pyrightconfig.json` of Pyright. See `packages/pytea/src/service/pyteaOptions.ts` for more details.

Place `pyteaconfig.ts` to the directory that contains the target Python file, or run with options like below.

```bash
python bin/pytea.py --node-arguments="--configPath=path/to/pyrightconfig.json" path/to/source.py

# or use below
node bin/index.js --configPath=path/to/pyrightconfig.json path/to/source.py
 ```

```typescript
export type PyCmdArgs = { [option: string]: boolean | number | string };
export type PyteaLogLevel = 'none' | 'result-only' | 'reduced' | 'full';
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
    logLevel: PyteaLogLevel;

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

    // Pass analysis result to Python Z3 server (default: false)
    runZ3: boolean;

    // Port to Python Z3 server
    z3Port: number;

    // Analyzer timeout in millisecond. undefined means no timeout (default: undefined)
    timeout?: number;

    // Set max path count, throw runtime error if path count exceeds it (default: undefined)
    maxPath?: number;
}
```
