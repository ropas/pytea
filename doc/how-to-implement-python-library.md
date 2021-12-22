# How to implement Python Library API and LibCalls

Every path below comes with a prefix `packages/pytea`.

## Source structure

- `src/pytea.ts`: PyTea entry point
- `src/service`: Service modules (language server / analysis manager)
  - `pyteaService.ts`: Manage Python scripts, setting, logging, and import resolutions
- `src/frontend`: Translate Python script into PyTea IR
  - `torchFrontend.ts`: Main translation engine
  - `torchStatements.ts`: Definition of PyTea IR
- `src/backend`: Run online analysis and collect constraints
  - `torchBackend.ts`: Main analysis engine
  - `context.ts`: Implementation of `Context` and `ContextSet`. Look after the interfaces of those classes.
  - `constraintSet.ts`: Constraint set of a single path. Most of the methods can be accessed from a `Context`, so you should not use those methods in this class directly.
  - `constraintType.ts`: Definitions of constraints
  - `constraintSolver.ts`: Online constraint checker (Simple Linear CAS)
  - `expUtils.ts`: Simplify symbolic expressions and extract information from it
  - `sharpValues.ts`: Values of PyTea IR Static semeantics
  - `sharpEnvironments.ts`: Heap and Enviroment of PyTea IR
  - `symExpressions.ts`: Symbolic variables and expressions
  - `range.ts`: Abstract range domain
- `src/pylibImplements`: Implementations of PyTea LibCall (Semantics of PyTorch and 3rd-party libraries)
  - `index.ts`: Register `libCallMap`
  - `libcall.ts`: Implemntation of special LibCalls except `explicit` (e.g., `import`, `callKV`, `exportGlobal`). Those are mainly for special Python semantics like `f(*args, **kwargs)` (variadic/keyword parameter).
- `pylib`: Implementation of Python builtin and 3rd-party libraries


## Overall workflow

1. Implement PyTorch (or other library) API by referring to `pylib` and [supported Python syntax](supported-python-syntax.md) on directory `pylib`. For some well-formed high-level modules like [torchvision/models/resnet.py](../packages/pytea/pylib/torchvision/models/resnet.py), you can completely copy and paste the original source.
2. Replace expressions which require explicit constraints by appropriate `LibCall` methods, like
`LibCall.torch.matmul(self, other)`, or `LibCall.guard.require_lt(0, x)`. Each `LibCall.XXX.YYY(...)` will invoke the function `YYY(...)` from `src/pylibImplements/XXX/index.ts` or `src/pylibImplements/XXX.ts`.
3. Suppose that we use a function `LibCall.foo.bar(x, y)` from Python script. You should implement `bar(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue>` on `src/pylibImplements/backend/foo/index.ts`. If there is no directory or file `foo`, create it and register `libCallImpls` at `src/pylibImplements/backend/index.ts`.
4. Register `foo` at `libCallImpls`. `libCallImpls` is usually placed in the end of the file.

### Note

- PyTea internal values (Context, Env, Heap, Value, Constraint) are implemented with [Immutable.js](https://immutable-js.com/), the efficient persistent data structure library for JS.
  - Thus, you cannot fix the value in-place. Every constraint have to be created by context method such as `ctx.genLte(..., ...)`
- Integer ranges are 0-based, inclusive lower bound and exclusive upper bound, which is analagous to Python range. `ExpShape.slice(shape, start, end)` is equal to `shape[start:end]`.

## Structure of Context and LibCall implementations
### `Context<T>` and `ContextSet<T>`

`Context<T>` holds every information (e.g. `ShEnv`, `ShHeap`) to execute a single Python statement. `ContextSet<T>` is a set of `Context`s, which holds both successful paths and failed paths.

`Context<T>` has several `setXXX` chaining methods. The user should combine result value and environments to it and construct the next context.

`ContextSet` is a monadic structure; the user can combine `map` and `flatMap` methods to control the execution path, and inject constraints by using `require` method.

### Properties of `Context<T>`

```typescript
interface ContextProps<T> {
  failId: number; // context id
  env: ShEnv; // Id -> Addr
  heap: ShHeap; // Addr -> Value
  ctrSet: ConstraintSet; // Hard \/ Soft \/ Path constraints
  retVal: T; // return value of the last expression or statement

  // the properties below are internal values; the user should not directly modify these values.
  // SVFunc is a type of Python function, string is a name of LibCall.
  callStack: List<[SVFunc | string, CodeSource | undefined]>;
  logs: List<ShValue>; // error logs (appended by warnXXX, failXXX, etc...)
  imported: ShEnv; // qualified path -> Addr.
  relPath: string; // relative path of current file.

  // if it is set, this context is regarded as a failed path.
  failed?: SVError;
}
```
### `ContextMethods<T>`

To implement semantics of Python or PyTorch API, user should use a bunch of methods in `ContextMethods<T>`.

Beware that `Context<T>` is **immutable**. Each method builds a new `Context` or `ContextSet`, and the source `Context` will not be modified.

In a context of `LibCall` implementation, the below methods are usually used or required:

- `toSet`, `toSetWith`: Make a single context to context set. `toSetWith` sets a return value.
- `setXXX`: Setter function of `Context`. This method is **not** an in-place method.
- `getAttrDeep`: Get an attribute with `__getattr__` method. If the attribute is already assinged, `__getattr__` will be ignored just like the Python semantics. It might call any nondeterministic function, so the type of the return value is `ContextSet`.
- `warn`, `warnWithMsg`: Write warning log and set new `SVError` type value to `ctx.retVal`.
- `fail`, `failWithMsg`: Immediately stops current path.
- `warnTensorWithMsg`: Write warning log, but set a symbolic unknown-ranked tensor to `ctx.retVal`. If the shape of the return value cannot be statically known, this method should be called.
- `genSymXXX`: Generate new symbolic variable.
- `genIntGte`, `genFlaotGte`: Return new constrainted symbolic variable. the result context holds that variable in `ctx.retVal`.
- `genConstRankedShape`: Return new ranked tensor. `partialDims` is a partial map which defines each dimension of result tensor.
- `genRankedShape`: Return new ranked tensor with unconstrainted symbolic dimensions.
- `genXXX` (e.g. `genEq`): Constructors of constraints
- `parseSize`: if the `iterable` is a list or tuple of integer, PyTea regards those values as dimenstions, and returns new `ExpShape` using those dimensions. If the parsing is failed, it will return string type of error message.
- `shXXX`: Basic shape operations regarding constrints. If the constraints are violated, this context will be added to failed path in the result `ContextSet`.
- `require`: Soft constraint setter.
- `guarantee`: Hard constraint setter.
- `ifThenElse`: Return two paths (true/false paths) with a given constraint. If the given constraint can be immediately known to be constant true or false, the other side of result set will be empty set.
- `getCachedRange`: Return a conservative range of a given `ExpNum` by regarding current `ctrSet`.
- `checkImmediate`: Discriminate that a given constraint is constant true or false, or undecidable by current `ctrSet`.
### Major utility functions

- `src/backend/backUtils.ts`
  - `fetchAddr(value, heap)`: Recursively dereference the address if `value` is `SVAddr` type.
  - `sanitizeAddr(value, heap)`: Recursively dereference the address, but stops when the dereferenced value is `SVObject` type.
- `src/pylibImplements/backend/utils.ts`:
  - `genTensor(ctx, shape, source?)`: return `torch.Tensor` object with a given shape.
  - `fetchSize(mayAddr, heap)`: if `mayAddr` is a pointer of a `SVSize` value, return that `SVSize` object.

### `ShValue`, `SVObject` and `SVSize`

`ShValue` consists of Python objects and primitive values which are expressed in symbolic expression, such as `a + 3 * b`.

Literal values such as `SVInt`, `SVFloat`, `SVString`, and `SVBool` have JS `number`, `string`, `boolean`, or `SymExp` object. `ctx.getCachedRange` can calculate current conservative (i.e. lower and upper bound) range of each value.

`SVObject` consists of three internal maps and its pointer (i.e. address). If `SVObject` is 'shapable', which can be regarded as shaped value such as Tensor, ndarray, or nested array, PyTea will assign `shape` property to it. From this situations, `SVObject` value can be casted to `SVSize` type. `SVSize` type can be treated as Python tuple or `torch.Size`.

Three maps in `SVObject` represent 'array' (e.g. 'a[1]'), 'attributes' (e.g. `a.x`), and 'dictionary' (e.g. `a['key']`]). Mathematically, `SVObject` is a disjoint union of three maps, `Attr -> Value`, `Int -> Value`, and `String -> Value`.

Unlike the original Python, there is no hash function of a plain object in PyTea. Therefore, we cannot use a plain object as a key of a dictionary. (e.g. 'x = object(); a[x] = 3' is prohibited.)

To make new ShValue, the user should use any derived `ShValue.create` functions such as `SVInt.create`.
