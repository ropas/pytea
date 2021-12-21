# Python/Node communication protocol (JSON-RPC)

WIP

## example

### ping

#### request (node -> python, 100ms timeout)
```js
{
    "jsonrpc": "2.0",
    "method": "ping",
    "id": 0,
    "params": ["14214"] // random number
}
```

#### respond (python -> node)
```js
{
    "jsonrpc": "2.0",
    "id": 0, // number
    "result": [
        "14214" // echo
    ]
}
```

### z3 solve request (node -> python)
```js
{
    "jsonrpc": "2.0",
    "method": "solve",
    "id": 0, // number
    "params": [
        {
            "ctrPool": [{ ... }, ... ], // list of constraints
            "hardCtr": [1, 3, 4, ...],  // indices of hard constriants
            "softCtr": [0, 5, 7, ...],  // indices of soft constriants
            "pathCtr": [2, 6, 8, ...]   // indices of path constriants
        },
        ...
    ] // constraint set of each paths
}
```

#### send result (python -> node)
```js
{
    "jsonrpc": "2.0",
    "id": 0, // number
    "result": [
        { "type": 1 }, // type: PyZ3RPCResult
    ], // results for each paths
    "log": "<some extra messages>"
}
```

## Result type

```typescript
enum PyZ3RPCResultType {
    Unreachable = 0,
    Valid = 1,
    Sat = 2,
    Unsat = 3,
    DontKnow = 4,
    Timeout = -1,
}

interface PyZ3RPCResult {
    type: PyZ3RPCResultType,
    extras: {
        conflict?: number // first index of conflicted constraint (type == Invalid)
        undecide?: number // first index of undecidable constraint (type == Undecidable)
    }
}
```