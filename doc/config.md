# Config (pyteaconfig.json)

PyTea reads `pyteaconfig.json` to handle analysis options of Pytea, like `pyrightconfig.json` of Pyright.

The user should place `pyteaconfig.json` in which contains the target Python file. If the config file does not exist, default parameters will be used.

The user can also run PyTea with explicit command-line options like below:

```bash
# set explicit path of pyteaconfig.json
python bin/pytea.py --config=path/to/pyteaconfig.json path/to/source.py

# or use below
node bin/index.js --configPath=path/to/pyteaconfig.json path/to/source.py
```

The main paper of PyTea calls `bin/index.js` (Constraint Generator) as **Online analyzer**, and `bin/pytea.py` (Z3Py) as **Offline analyzer**. From this document, we will call them **JS script** and **Python script**, respectively.

## Notable Configurations

### `pythonCmdArgs: { [option: string]: boolean | number | string }`:

These options will be injected to `ArgumentParser.add_argument(...)` method call of Python [argparse](https://docs.python.org/3/library/argparse.html) module.

If the `required` option of `add_argument` is set but `pythonCmdArgs` does not have an option on it, the analyzer will raise the error immediately.

```python
# "pythonCmdArgs": { "batch_size": 32, "device": "cuda", "epoch": 1 }

parser = arparser.ArgumentParser()
parser.add_argument('batch_size', type=int, default=16)
parser.add_argument('device', required=True)
parser.add_argument('epoch', type=int, default=30)

args = parser.parse_args()
# args => { { "batch_size" => 32, "device": "cuda", "epoch": 1, ... }, {}, {} }
```

### `logLevel: "none" | "result-only" | "reduced" | "full"`

This option determines how detailed Pytea will print the analysis result.

Python script has an alias of this option. Use `-l` or `--log`, like `--log=1` (`logLevel` will be set to `"result-only"`).

#### `none`

Online analyzer will execute the analysis silently. Print only the result of offline analyzer.

```bash
> python bin/pytea.py --log=0 path/to/source.py

analyzer starts!
------------- z3 result -------------
<OVERALL: total 1 paths>
  Valid paths (no constraint error): 1
```

#### `result-only`

Print the final state of each path when the online analyzer is finished. If the analyzer found error, it will print the conflicted constraint and the source position of that constraint. It also prints the call stack in order to determine which user code is exactly failed.

It also prints **Warnings**. Warnings contains **Unimplemented function call and module import**, so if you have met some false warnings about unimplemented API that we have not implemented, please make issue and notify us.

#### `reduced`

In addition to the above, print every value assingned to the variables in the environment (i.e., reduced environment and heap) at the stopped position. It also prints every collected constraint from each path.

Each constraint is divided by three classes.

- **Hard Constraint**
  - **Initial Constraint** (Dark Gray): Initialization condition of each symbolic variable. Produced from a for loop (index range) or any unknown tensor initialization which is marked that each dimension should be non-zero.
  - **Path Constraint** (Yellow): Branch condition (e.g., `if ...` in Python syntax).
- **Soft Constraint** (White): Safety conditions required by each tensor-related API. This kind of constraint should not be violated under the previous Hard constraints (i.e. under all possible input values).

#### `full`

In addition to the above, print raw dump of the PyTea environment object. It also prints the PyTea IR (internal representation) translation of the entry file. You may not use this option.

## Full Configurations

For the up-to-date configuration, see [pyteaOptions.ts](../packages/pytea/src/service/pyteaOptions.ts)

```typescript
export type PyCmdArgs = { [option: string]: boolean | number | string };
export type PyteaLogLevel = "none" | "result-only" | "reduced" | "full";
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

  // Severity of analysis result (default: reduced)
  logLevel: PyteaLogLevel;

  // Turn on online checker.
  // Check and discard trivial constraints (default: true)
  immediateConstraintCheck: boolean;

  // Ignore assert statements of Python. (default: false)
  ignoreAssert: boolean;

  // Extract internal representation to external file (TorchIR)
  extractIR: boolean;

  // Explicit range of random variables.
  // Key should be the prefix of specific random variable.
  // If the name of random variable is 'PILImgC_I3', the key should be "PILImgC".
  // (the suffix '_I3' means third immediate random variable)
  // The range of random varaible which name starts with prefix will be altered to this.
  // Range is always inclusive. null means the range is unbounded (or half-bounded).
  variableRange: {
    [prefix: string]: null | number | [number | null, number | null];
  };

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
  timeout: number | undefined;

  // Set max path count, throw runtime error if path count exceeds it (default: 1000)
  maxPath: number | undefined;

  // Port to Python Z3 server (default: 17851)
  z3Port: number;
}

export const defaultOptions: PyteaOptions = {
  configPath: "",
  pyteaLibPath: "",
  entryPath: "",
  pythonCmdArgs: {},
  pythonSubcommand: "",
  logLevel: "reduced",
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
```
