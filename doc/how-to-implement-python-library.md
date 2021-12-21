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

- `SymExp`, `SymVal`, `Constraint`를 제외한 모든 값은 immutable.js를 사용하여 구현되었다. 즉, assign을 통한 프로퍼티의 직접 변경이 불가능하다. (persistent data structure)
  - 위 3개는 (가능하더라도) 프로퍼티의 직접 수정을 하지 말아야 한다. `ExpNum.create`, `ctx.genLte` 등의 유틸 함수들을 활용하여 값을 제작하여야 한다.
- 기존 `Context`를 context(heap+env+constraint set) 자체인 `Context<T>`와 success, failed path의 조합인 `ContextSet<T>`로 분리.
- static_semantics.pdf 와 약간 다른 expression 구현. (Shape에서 SetDim과 Symbolic integer/float의 구분이 추가됨)
- 모든 integer range는 0-based, exclusive이다. 즉, Python의 range 범위와 동일하다.
  - ExpShape.slice(shape, start, end)는 Python의 `shape[start:end]`과 같다. 즉, start, dim 모두 0부터 시작하는 index이며, start는 포함하고 end는 포함하지 않는 range이다.

## Implementation of Context<T> and ContextSet<T>

`Context<T>`는 코드 상의 어떤 한 지점에서 코드의 한 라인을 실행시키기 위한 모든 정보를 가지고 있는 자료구조이다. `ContextSet<T>`는 `Context`의 집합으로, 성공 path들의 리스트와 실패 path의 리스트를 모두 들고 있다.

Context로는 Environment와 Heap을 참조하여 constraint와 return value를 만들어낼 수 있고, setXXX 함수 등으로 다음 context를 만들어 낼 수 있다.

ContextSet은 직접 조작이 불가능하며 map, flatMap을 조합하여 코드의 실행을 구현할 수 있고, require, guarantee 문을 통해 constraint를 주입할 수 있다.

**`Context<T>`의 프로퍼티 설명**

```typescript
interface ContextProps<T> {
  env: ShEnv; // Id -> Addr
  heap: ShHeap; // Addr -> Value
  ctrSet: ConstraintSet; // Constraint 집합
  retVal: T; // 어떤 expression 또는 statement를 evaluate 한 후에 결정되는 return value.

  // 이하는 전부 ContextMethods 및 TorchBackend에서 자동으로 추적해주는 내부 변수이다. LibCall 구현에서 직접 조작은 할 필요 없다.
  callStack: List<[SVFunc | string, ParseNode | undefined]>; // function call stack
  logs: List<ShValue>; // warn, fail 등의 함수를 통해 모은 로그들의 리스트
  imported: ShEnv; // import된 module들의 heap에서의 위치. (한번 import한 모듈은 다시 import 안하므로 계속 유지해주어야 함)
  relPath: string; // 현재 context가 어떤 경로의 스크립트에서 실행되고 있는지 추적

  // fail 또는 failWithMsg를 부르거나 require문에서 실패할 시 자동으로 결정됨
  // 이 값이 set되었다면 ContextSet에서 더 이상 map/flatMap을 통해 진행할 수 없게 된다.
  failed?: SVError;
  failId: number;
}
```

함수를 구현할 때 맨 마지막에 해야할 것은 ctx.setRetVal을 사용하여 함수의 return value를 설정해 주는 것이다. `Context<T>` 및 `ContextSet<T>`의 T 타입은 이 return value의 타입이다.

LibCall implementation을 비롯한 Backend의 각 함수의 리턴타입은 `ContextSet<T>`인데 현재 함수가 사용할 수 있는 값은 Context 타입의 ctx밖에 없을 때가 있다. 이 때는 `ctx.toSet()`이나 `ctx.toSetWith(retVal)`로 ContextSet 타입으로 만들어 주면 된다.

**`ContextMethods<T>`의 설명**

ContextMethods는 Context가 할 수 있는 일들을 모아놓은 interface로, Context에 구현된 함수들을 보기 좋게 한 곳에 몰아놓은 것이다.
LibCall 구현에 사용 가능한 함수는 다음과 같다.

- `toSet`, `toSetWith`: 단일 Context를 ContextSet으로 변환
- `setXXX`: ContextProps의 setter 함수. in-place setter가 아님에 유의할 것
- `getAttrDeep`: object의 attribute를 (존재한다면) `__getattr__` 함수를 고려해서 가져온다. (내부적으로 functionCall을 부르기 때문에 return value가 ContextSet이다.)
- `warn`, `warnWithMsg`: 로그를 남기면서 SVError 타입을 retVal로 설정.
  - Not implemented API 등이 사용될 때 SVError 타입의 값을 반환하므로 LibCall의 parameter로 언제든지 SVError를 받을 수 있다. 이 때 warning을 propagate하면 된다.
- `fail`, `failWithMsg`: 즉시 path를 fail시킨다. shape error 등을 즉시 판단할 수 있을 때 사용하면 된다.
- `warnTensorWithMsg`: 일부 PyTorch API는 잘못된 (주로 propagate된 SVError 타입의) argument를 받더라도 임의 dimension의 Tensor 타입의 값을 반환하는 것이 안전할 때가 있다. 이 때 `warn`대신 `warnTensorWithMsg`를 사용하면 에러 로그를 남기면서도 새로운 symbolic Tensor를 반환할 수 있다.
- `genSymXXX`: 새로운 symbolic variable을 제작한다.
- `genConstRankedShape`: `rank` 랭크를 가지는 새로운 텐서를 반환한다. partialDims에 일부 확정적인 dimension을 담을 수 있다.
- `genRankedShape`: `rank` 랭크를 가지며 dimension은 전부 새로운 symbolic variable인 새로운 텐서를 반환한다.
- `genEq` 등의 `genXXX`: Constraint의 constructor function들
- `parseSize`: `iterable`이 정수 리스트 또는 튜플이면 (indices에 SVInt들이 있다면) 이를 dimension들로 간주하고 파싱한 ExpShape을 반환한다. 만약 오류가 있다면 오류 메시지(string)을 반환한다.
- `shXXX`: 기초 shape 연산. constraint에서 오류가 생긴다면 자동으로 fail path로 넘어간다.
- `require`: soft constraint setter. 틀리면 즉시 이 자리에서 오류를 반환해야 한다.
- `guarantee`: hard constraint setter. 새로운 symbolic variable이 dimension이므로 0보다 크다 라는 constraint 같은것을 주입할 때 이를 사용한다.
- `ifThenElse`: constraint를 입력하면 if/else 분기 각각을 탄 새로운 ContextSet 2개를 출력. 만약 한 path만 탄다면 다른 한 쪽은 그냥 빈 ContextSet이 된다.
- `getCachedRange`: 현재까지의 ctrSet을 기반으로 특정 number 또는 ExpNum의 range 추정 값을 반환. 다음 단락 참고
- `checkImmediate`: 현재까지의 ctrSet 기반으로 `constraint`가 true인지 false인지, 또는 판단 불가능한 undefined지를 반환.

**`ContextSet<T>`의 주요 함수**

- `map`, `flatMap`: Monad 구조를 사용하여 functional하게 imperative language를 구현할 수 있게 하는 핵심 함수들. 내부 Context들을 변형해야 한다면 map을, 내부에서 암묵적으로 function call 또는 expression evaluation 등이 사용된다면 flatMap을 사용하면 된다.
- `return`: `retVal`을 특정 값으로 세팅한다. 함수의 return value를 세팅하는 함수
- `join`: path가 나뉜 두 ContextSet을 모은다. `ifThenElse`로 나뉜 두 분기를 한데 모을 때 사용.
- `fail`, `require`, `guarantee`, `ifThenElse`: `Context<T>`와 같음

**그 외 주요 유틸리티 함수**

- `src/backend/backUtils.ts`
  - `fetchAddr(value, heap)`: value가 address 타입이라면 heap을 따라가 address가 가리키는 value를 반환. `***ptr`같이 address가 연쇄적으로 있는 값이라면 address를 전부 따라간 최종 값을 반환
  - `sanitizeAddr(value, heap)`: 기본적으로 `fetchAddr`과 비슷하나 만약 도달한 값이 SVObject 타입일 경우 그 SVObject를 가리키는 SVAddr 타입의 값을 반환. (만약 value가 처음부터 SVObject 타입이면 그 값을 그대로 반환)
  - `isTruthy(ctx, value, source)`: value가 Truthy인지 Fasly인지를 반환. 판단 불가라면 value가 Truthy임을 나타내는 Constraint를 반환.
- `src/pylibImplements/backend/utils.ts`:
  - `simplifyExp(ctrSet, exp)`: SymExp 타입의 값을 가능한 계산하여 축약한다. 사칙연산의 결합/교환/분배 법칙이 구현되어 있다.
  - `genTensor(ctx, shape, source?)`: shape을 기반으로 한 Tensor 타입의 object를 반환한다. 기본적으로 Python에서 `torch.Tensor(*shape)`을 부르는 것과 동일하나, 내부적으로 해야하는 일이 많기 때문에 새로운 Tensor를 만들 때는 이 함수를 사용하여야 한다.
  - `fetchSize(mayAddr, heap)`: 만약 mayAddr이 SVAddr 타입의 값이라 가리키는 값이 SVSize라면 그 SVSize 값을 반환한다.

### getCachedRange의 사용

TypeScript에 구현한 in-place SMT는 conservative range를 기반으로 구현되어 있다.

`getCachedRange`는 현재까지의 constraint set를 기반으로 특정 ExpNum의 range를 판단할 수 있으며, 리턴 타입은 NumRange 또는 (constraint가 깨졌다면) undefined를 반환한다.

NumRange 타입은 닫힌 구간, 열린 구간, 반만 닫힌 구간을 구현할 수 있다. 또한 사칙연산과 대소 비교가 간단하게 구현되어 있으며, 자세한 내용은 NumRange class를 참조할 것.

만약 `__getitem__` 같이 LibCall 구현 도중에 정확한 numeric value를 알아야 하는 경우 이 함수를 사용할 수 있으며, 만약 정확한 value가 아니라 range가 나왔다면 warn또는 warnTensorWithMsg를 사용하여 현재 LibCall이 정확한 값을 반환할 수 없다는 것을 표현할 수 있다.

### ShValue 및 SVObject, SVSize의 구현

`ShValue`는 이전의 `ThValue`를 계승하는 Python의 primitive value들의 구현이다. immutable.js 기반으로 구현되어 있으므로 값의 직접 변경은 불가능하며, 기존 값을 기반으로 새로운 값을 만들어낼 수만 있다.

`SVInt`, `SVFloat`, `SVString`, `SVBool` 등의 literal value는 내부에 number, string, boolean 또는 SymExp 타입의 value가 들어있다. 실제 사용시에 typeof value를 통해 SymExp인지 아닌지 체크하고 SymExp 타입인 경우 만약 정확한 값을 알아내야 한다면 ctx.getCachedRange를 통해 Exp값의 범위가 어느 정도인지 알아내야 한다.

가장 중요하고 많이 사용하는 `SVObject` 타입은 총 3개의 내부 값으로 이루어져 있다. 수학적으로는 `Attr + Int + String -> Value` 지만 이를 3개 map의 disjoint union으로 보아 `Attr -> Value` (attrs), `Int -> Value` (indices), `String -> Value` (keyValues)로 분리 시킨 것이다. 각각 `a.x`와 같은 attribute, `a[1]`과 같은 indexed value, `a["X"]`와 같은 key-value map을 표현한다.

`SVSize`는 `SVObject`을 상속받은 타입으로, torch.Size 클래스의 instance를 의미한다. torch.Size 클래스는 python의 tuple 타입을 상속하므로 SVSize도 기본적으로는 tuple object와 똑같이 다룰 수 있으나 내부적으로 shape 변수가 있어서 현재 shape이 무엇인지를 알 수 있다.

새로운 ShValue를 만들 때는 `SVInt.create` 등의 함수를 사용한다. (interface이므로 new를 사용해서 만들 수 없다.) 항상 source를 넣어주는 것을 잊지 말자.

### Constraint의 추가

Constraint는 3 종류로 나뉜다.

- hard constraint: guarantee 함수로 넣을 수 있는, z3의 assumption이 되는 기초 constraint. 새로 만든 dimension이 0 이상임을 보장하는 것과 같은 곳에서 사용된다.
- soft constraint: require 함수로 넣을 수 있는, z3이 sat/unsat인지 판단해야 하는 constraint. 틀리면 이것이 틀렸다고 unsat을 반환해야 한다.
- path constraint: ifThenElse로 세팅되는 constraint. 틀리면 처음부터 잘못된 분기를 탔다는 것을 인지하고 따로 로그 없이 제거되어야 한다.

`ctx.genXXX` 함수를 통해 constraint를 만들고 `guarantee, require, ifThenElse`로 constraint를 주입하면 된다.

## Implementation of LibCall

LibCall의 파라미터로는 `Context<ExplicitParams>`를 받는데, 이 안에는 단순히 `ShValue[]` 타입의 `ctx.retVal.params` 값이 있을 뿐이다. 이 안에서 타입을 적절히 판단해서 Tensor와 사이즈를 뽑아내려면 적절한 boilerplate가 필요하다. (나중에 이를 단순화시킬 수 있겠으나 현재는 생으로 해야한다.)

`pylibImplements/backend/torch/index.ts`의 `matmul` 함수를 분석하면서 전체적인 구조가 어떻게 진행되는지를 확인하자.

```typescript
// implementation of torch.matmul
export function matmul(
  ctx: Context<LCBase.ExplicitParams>, // 단일한 Context와 파라미터 리스트 (ExplicitParams)를 받는다.
  source?: ParseNode // 실제 코드의 어떤 곳에서 call 되었는지 추적하는 parameter. 다른 함수를 부를 때 가능한 한 이 source를 꼬박꼬박 넣어줘야 한다.
): ContextSet<ShValue> {
  // 리턴값은 path들의 조합인 ContextSet이어야 한다.

  const params = ctx.retVal.params; // LibCall.torch.matmul(x, y)를 불렀다면 이 params에 [x, y] 라는 리스트가 들어간다.

  if (params.length !== 2) {
    // parameter가 2개인 것을 확인해주어야 한다.
    // matmul 함수는 Tensor 타입을 리턴한다는 것을 아므로 ctx.warn 대신
    // warning log를 추가하면서 임의 rank의 텐서를 리턴하는 warnTensor WithMsg 함수를 사용한다.
    return ctx.warnTensorWithMsg(
      `from 'LibCall.torch.matmul': got insufficient number of argument: ${params.length}`,
      source
    );
  }

  const heap = ctx.heap;

  // LibCall 타입은 LibCall.torch.matmul(*args, **kwargs)와 같이 *이 붙은 parameter를 활용할 수 없다. 반드시 f(args, kwargs)와 같이 그대로 넣어줘야 한다.
  const [leftAddr, rightAddr] = params;

  // params가 받은 값은 pointer 값인 SVAddr 타입일 수 있다. 이를 heap을 따라 추적하여 SVSize value를 찾아내는 fetchSize 함수를 사용하여 size를 알아낸다.
  // 만약 도달한 값이 SVSize 타입이 아닐 경우 string 타입의 error log를 반환한다.
  const leftSize: SVSize = fetchSize(leftAddr, heap);
  const rightSize: SVSize = fetchSize(rightAddr, heap);

  // matmul의 left, right 값이 다른 함수콜로부터 propagate된 SVError 타입일 수 있다.
  // 이 때도 잘 처리해주기 위해 ctx.fail 대신 ctx.warnTensorWithMsg를 불러서 무작정 path를 끊어버리지 않도록 한다.
  if (typeof leftSize === "string") {
    return ctx.warnTensorWithMsg(
      `from 'LibCall.torch.matmul': ${leftSize}`,
      source
    );
  } else if (typeof rightSize === "string") {
    return ctx.warnTensorWithMsg(
      `from 'LibCall.torch.matmul': ${rightSize}`,
      source
    );
  }

  const leftShape = leftSize.shape;
  const rightShape = rightSize.shape;
  const leftRank = leftSize.rank();
  const rightRank = rightSize.rank();

  // 먼저 matmul rank의 requirement를 서술한다. ctx.require 함수를 통해 matmul의 shape requirement를 서술한다.
  return ctx
    .require(
      [ctx.genLte(1, leftRank, source), ctx.genLte(1, rightRank, source)], // rank는 모두 1 이상이어야 한다.
      `from 'LibCall.torch.matmul': cannot do matmul with scalar tensor`, // 이 조건이 틀렸을 경우에 대한 error message를 서술한다.
      source // genXXX, require, guarantee 등에 계속 넣어줘야 코드 상에서 어디에서 틀렸는지 판단할 수 있다.
    )
    .flatMap((ctx) => {
      // context가 다음으로 해야할 행동을 flatMap을 통해 기술한다.
      // `ctx.require(constraint).flatMap((ctx) => ...)`  에서 ctx는 constraint가 적용된 새로운 context이다.
      // 만약 constraint가 잘못되었다면 flatMap은 내부적으로 아예 실행되지 않게 된다.

      // torch.matmul는 left, right 값이 각각 rank가 1일때 또는 2 이상일 때를 고려하여 4개의 path로 나뉘게 된다.
      // 아래는 먼저 left의 rank가 1일때의 constraint를 만들고 이를 ifThenElse에 넘겨 left의 rank가 1 또는 2 이상일 때의 2개의 path를 만들게 된다.
      const isLeftRankOne = ctx.genEq(1, leftRank, source);
      const [leftOnePath, leftTwoPath] = ctx.ifThenElse(isLeftRankOne, source);

      // left의 rank가 1인 path.
      const leftPath = leftOnePath.flatMap((ctx) => {
        const isRightRankOne = ctx.genEq(1, rightRank, source);
        const [rightOnePath, rightTwoPath] = ctx.ifThenElse(
          isRightRankOne,
          source
        );

        // left와 right의 rank가 모두 1인 path.
        const lr11 = rightOnePath
          .flatMap((ctx) => {
            const sameDim = ctx.genEq(
              // 둘 다 rank가 1이라면 두 shape의 유일한 dimension이 서로 같아야 하며, 결과 Tensor는 scala tensor (rank-0) 이어야 한다.
              ExpNum.index(leftShape, 0), // shape의 index는 static-semantics.pdf의 문서와는 다르게 0-based exclusive 이다.
              ExpNum.index(rightShape, 0),
              source
            );

            // ctx.require문을 통해 위 constraint를 주입한다.
            return ctx.require(
              [sameDim],
              `from 'LibCall.torch.matmul': dimension mismatch between rank-1 tensors`,
              source
            );
          })
          .flatMap((ctx) =>
            // 새로운 Tensor는 아래처럼 genTensor를 통해 만든다.
            genTensor(ctx, ExpShape.fromConst(0, [], source), source)
          );

        // left의 rank가 1이고 right의 rank가 2 이상인 path.
        const rightAxis = ExpNum.bop(NumBopType.Sub, rightRank, 2);
        const lr12 = rightTwoPath
          .flatMap((ctx) => {
            const sameDim = ctx.genEq(
              ExpNum.index(leftShape, 0),
              ExpNum.index(rightShape, rightAxis),
              source
            );
            return ctx.require(
              [sameDim],
              `from 'LibCall.torch.matmul: dimension mismatch between rank-1 @ rank-n`
            );
          })
          .flatMap((ctx) => ctx.shReduce(rightShape, rightAxis, source)) // 왼쪽이 rank가 1일 경우, require문이 맞다면 right shape의 마지막에서 2번째 dim이 제거되어야 한다. 이를 shReduce로 표현할 수 있다.
          .flatMap((ctx) => genTensor(ctx, ctx.retVal, source));

        return lr11.join(lr12);
      });

      // 이하는 위와 비슷하다.
      const rightPath = leftTwoPath.flatMap((ctx) => {
        const isRightRankOne = ctx.genEq(1, rightRank, source);
        const [rightOnePath, rightTwoPath] = ctx.ifThenElse(
          isRightRankOne,
          source
        );

        const leftAxis = ExpNum.bop(NumBopType.Sub, leftRank, 1, source);
        const lr21 = rightOnePath
          .flatMap((ctx) => {
            const sameDim = ctx.genEq(
              ExpNum.index(leftShape, leftAxis),
              ExpNum.index(rightShape, 0),
              source
            );
            return ctx.require(
              [sameDim],
              `from 'LibCall.torch.matmul': dimension mismatch between rank-n @ rank-1`,
              source
            );
          })
          .flatMap((ctx) => ctx.shReduce(leftShape, leftAxis, source))
          .flatMap((ctx) => genTensor(ctx, ctx.retVal, source));

        // shMatmul 함수는 broadcasting을 고려한 matmul의 구현이다. 사용하려면 두 shape 모두 rank가 2 이상이여야 한다.
        const lr22 = rightTwoPath
          .flatMap((ctx) => ctx.shMatmul(leftShape, rightShape, source))
          .flatMap((ctx) => genTensor(ctx, ctx.retVal, source));

        return lr21.join(lr22);
      });

      // 나뉜 분기는 join을 통해 다시 합쳐준다.
      return leftPath.join(rightPath);
    });
}
```

