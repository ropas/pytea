# Config (pyteaconfig.json)

We use `pyteaconfig.json` to handle options for Pytea, following `pyrightconfig.json` of Pyright.

Place `pyteaconfig.ts` to the directory that contains the target Python file, or run with options like below.

```bash
# set explicit pyteaconfig.json
python bin/pytea.py --config=path/to/pyteaconfig.json path/to/source.py

# or use below
node bin/index.js --configPath=path/to/pyteaconfig.json path/to/source.py
```

We will call `bin/index.js` (Constraint Generator) as **Node side**, and `bin/pytea.py` (Z3Py) as **Z3 side**.

## Notable Configurations

### `pythonCmdArgs: { [option: string]: boolean | number | string }`:

This options will be injected to `ArgumentParser.add_argument(...)` method from Python [argparse](https://docs.python.org/3/library/argparse.html) module.

If `required` option of `add_argument` is on but `pythonCmdArgs` does not have it, the analyzer will raise error immediately.

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

This option determines how detailed Pytea will print the analysis result to the user.

`-l, --log` option of `pytea.py` will give an alias of this Node option. This option of Z3 side will be given as integer number (0 to 3), like `--log=2`.

#### `none`

Node side runs silently. Print final results of Z3 side only.

```bash
> python bin/pytea.py --log=0 path/to/source.py

analyzer starts!
------------- z3 result -------------
<OVERALL: total 1 paths>
  Valid paths (no constraint error): 1
```

#### `result-only`

Print final states of paths of Node sides. If the analyzer found error, it prints the broken constraint and the source position of that constraint. It also prints function call stacks, so users can determine which exact user code is failed.

It also prints **Warnings**. Warnings contains **Unimplemented function call and module import**, so if you have met some false warnings about unimplemented API that we have not implemented, please make issue and notify us.

#### `reduced`

In addition to the above, print every value assingned to variables in the environment at the stopped position (i.e., dumps reduced environment and heap), and every collected constraint from each path.

Constraints are divided by three classes.

- **Hard Constraint** (Dark Gray): Initialization condition of each symbolic variable. Produced from for loop (index range) or unknown tensor initialization (marks each dimension should be non-zero).
- **Path Constraint** (Yellow): Branch condition (e.g., `if ...` in Python syntax).
- **Soft Constraint** (White): Conditions required by each API. It can be violated, that means, if this constriant is violated, there is an error.

#### `full`

In addition to the above, print every raw dump of environments and python. It also prints PyTea IR (internal representation) translation result of the entry file. You may not want this option.

## Full Configurations

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
