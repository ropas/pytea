# Python/Node communication protocol (JSON-RPC)


## example

### ping

#### request (node -> python, 100ms timeout)
```json
{
    "jsonrpc": "2.0",
    "method": "ping",
    "id": 0,
    "params": ["14214"] // random number
}
```

#### respond (python -> node)
```json
{
    "jsonrpc": "2.0",
    "id": 0, // number
    "result": [
        "14214" // echo
    ]
}
```

### z3 solve request (node -> python)
```json
{
    "jsonrpc": "2.0",
    "method": "solve",
    "id": 0, // number
    "params": [
        {
            ctrPool: [{ ... }, ... ], // list of constraints
            hardCtr: [1, 3, 4, ...],  // indices of hard constriants
            softCtr: [0, 5, 7, ...],  // indices of soft constriants
            pathCtr: [2, 6, 8, ...]   // indices of path constriants
        },
        ...
    ] // constraint set of each paths
}
```

#### send result (python -> node)
```json
{
    "jsonrpc": "2.0",
    "id": 0, // number
    "result": [
        { type: "valid" }, // type: PathResult
    ] // results for each paths
}
```

## Result type

```typescript
enum PathResultType {
    Nonexistent = 0,
    Valid = 1,
    MayInvalid = 2,
    Invalid = 3,
    Undecidable = 4,
}

interface PathResult {
    type: PathResultType,
    extras: {
        counterexample?: string // if type == MayInvalid
        conflict?: number // index of conflicted constraint (type == Invalid)
    }
}
```