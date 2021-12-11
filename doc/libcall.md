# `LibCall` APIs

`LibCall` expression extends the semantics of PyTea IR in order to express the versatile and dynamic behaviors of Python. It directly communicates with PyTea online checker, hence it can explicitly handle the context and constraint generation.

`LibCall` is divided into two classes: 'Internal LibCalls' and 'Explicit LibCalls'.

## Internal LibCalls

Internal LibCalls are generated only when in translation stage which parses Python code and translates into PyTea IR. It extends PyTea IR to handle several Python features such as default parameter, and PEPs such as [PEP 570](https://www.python.org/dev/peps/pep-0570/)(Positional-only parameters) (though the current version does not support it, but the future version of PyTea might be able to handle it by appending internal libcalls).

### `import`

```python
# Python
import a.b.c
from a.b.c import d

# Translated to PyTea IR
LibCall("import", qualPath="a.b.c")
LibCall("import", qualPath="a.b.c", assignTo="d")
```

To make an importing stage simple, qualified import (i.e. `import a.b.c`) behaviors like `import a; import a.b; import a.b.c` from the online checker.

* CAVEAT: `import` in a block statement (i.e. if, function body, etc.) is currently not supported. Every import statement must be placed at the header of Python script.

### `genList`, `genDict`

```python
# Python
x = [1, 2]
y = [i + 1 for i in x]

# Translated to PyTea IR
x = LibCall("genList", param$0=1, param$1=2)
def $Imm0_LCFun():
    $Imm0_LCList = LibCall("genList")
    for i in x:
        $Imm0_LCList.append(i + 1)
y = $Imm0_LCFun()
```

* CAVEAT: list unpacking in a list expression (`y = [4, *x]`) is currently not supported, and also raise parse error from the PyTea frontend. (this will be fixed ASAP)

### `setDefault`

Set default parameter of Python function.

### `callKV`

Set keyworded argument of Python function call.

### `objectClass`

Returns Python built-in `object` class.

### `exportGlobal`

Assign the given value to its module object. Dynamic semantics of PyTea IR does not allow implicit export of global variable, so this LibCall is used to implement it.

### `raise`

Raise an exception. Currently, PyTea does not support exception handling, so this LibCall unconditionally halts the analysis.

### `explicit`

See below.

## Explicit LibCalls

Explicit LibCalls are used to configure behaviors of Python library APIs. It also directly connects Python code and TypeScript code, hence it behaves like TypeScript FFI in Python script. There are hundreds of explicit LibCalls, so we will show the exemplary pattern of how to implement and use it.

The basic `torch.mm` API is implemented as follows:

```python
# pytea/pylib/torch/functional.py
def mm(input, mat2, out=None):
    if not (input.dtype == mat2.dtype):
        raise TypeError("Tensor dtype mismatch")
    dtype = input.dtype
    tensor = LibCall.torch.mm(input, mat2) # LibCall("explicit", $func="torch.mm", input, mat2)
    tensor.dtype = dtype
    LibCall.torch.copyOut(tensor, out)
    return tensor
```

From the above example, `LibCall.torch.mm` calls the foreign TS function:

```typescript
// pytea/src/pylibImplements/torch/index.ts
export function mm(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
    const params = ctx.retVal.params;

    // ...

    const heap = ctx.heap;
    const [leftAddr, rightAddr] = params; // input, mat2

    const leftSize = fetchSize(leftAddr, heap);
    const rightSize = fetchSize(rightAddr, heap);

    // ...

    // ctx.require pushes the given constraint on its context.
    // if the online checker finds that the given constraint is always false on the
    // current context, it immediately raise an error with a given error message.
    return ctx
        .require(
            [ctx.genEq(2, leftRank, source), ctx.genEq(2, rightRank, source)],
            `from 'LibCall.torch.mm': input must be 2-D tensors`,
            source
        )
        .require(
            ctx.genEq(ExpNum.index(leftShape, 1, source), ExpNum.index(rightShape, 0, source), source),
            `from 'LibCall.torch.mm': dimension mismatch`,
            source
        )
        .flatMap((ctx) => {
            // the context is a monadic object, so it composes the next continuation with
            // `flatMap` method.

            // make a tensor with a calculated shape.
            const newShape = ExpShape.fromConst(
                2,
                [ExpNum.index(leftShape, 0, source), ExpNum.index(rightShape, 1, source)],
                source
            );
            return genTensor(ctx, newShape, source);
        });
    }
```