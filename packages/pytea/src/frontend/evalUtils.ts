/*
 * frontUtils.ts
 * Copyright (c) Seoul National University.
 * Licensed under the MIT license.
 * Author: Woo Sung Song
 *
 * Utility functions for evaluate function on torchInterpreter.ts.
 */

import { ParseNode } from 'pyright-internal/parser/parseNodes';

import { ThEnv, ThHeap } from './torchEnvironments';
import { TorchInterpreter } from './torchInterpreter';
import { TEBopType, TEUopType } from './torchStatements';
import {
    ThContFlag,
    ThValue,
    TVAddr,
    TVBool,
    TVError,
    TVFloat,
    TVFunc,
    TVInt,
    TVNone,
    TVNotImpl,
    TVObject,
    TVString,
    TVType,
} from './torchValues';

export namespace OperatorUtils {
    export function isElementType(type: TVType): boolean {
        return type === TVType.Bool || type === TVType.Int || type === TVType.Float;
    }

    function elementTypeFetchValue(element: ThValue): number | boolean {
        if (element as TVBool) {
            return (element as TVBool).value;
        } else if (element as TVInt) {
            return (element as TVInt).value;
        } else if (element as TVFloat) {
            return (element as TVFloat).value;
        } else {
            // cannot reach here if element is elementType.
            return 0;
        }
    }

    export function evalElementTypeBinOp(left: ThValue, right: ThValue, bop: TEBopType, source?: ParseNode): ThValue {
        const [leftValue, rightValue] = [Number(elementTypeFetchValue(left)), Number(elementTypeFetchValue(right))];

        let resultType: TVType;
        let resultValue: number | boolean;

        switch (bop) {
            case TEBopType.Add:
                resultType = elementTypeUpperBoundOfTypes([left.type, right.type, TVType.Int]);
                resultValue = leftValue + rightValue;
                break;
            case TEBopType.Sub:
                resultType = elementTypeUpperBoundOfTypes([left.type, right.type, TVType.Int]);
                resultValue = leftValue - rightValue;
                break;
            case TEBopType.Mul:
                resultType = elementTypeUpperBoundOfTypes([left.type, right.type, TVType.Int]);
                resultValue = leftValue * rightValue;
                break;
            case TEBopType.FloorDiv:
                resultType = elementTypeUpperBoundOfTypes([left.type, right.type, TVType.Int]);
                resultValue = Math.floor(leftValue / rightValue);
                break;
            case TEBopType.Mod:
                resultType = elementTypeUpperBoundOfTypes([left.type, right.type, TVType.Int]);
                resultValue = leftValue % rightValue;
                break;
            case TEBopType.TrueDiv:
                resultType = elementTypeUpperBoundOfTypes([left.type, right.type, TVType.Float]);
                resultValue = leftValue / rightValue;
                break;
            case TEBopType.Lt:
                resultType = TVType.Bool;
                resultValue = leftValue < rightValue;
                break;
            case TEBopType.Lte:
                resultType = TVType.Bool;
                resultValue = leftValue <= rightValue;
                break;
            case TEBopType.Eq:
                resultType = TVType.Bool;
                resultValue = leftValue === rightValue;
                break;
            case TEBopType.Neq:
                resultType = TVType.Bool;
                resultValue = leftValue !== rightValue;
                break;
            case TEBopType.And:
                if (leftValue === 0) {
                    resultType = left.type;
                    resultValue = false;
                } else {
                    resultType = right.type;
                    resultValue = rightValue;
                }
                break;
            case TEBopType.Or:
                if (leftValue === 0) {
                    resultType = right.type;
                    resultValue = rightValue;
                } else {
                    resultType = left.type;
                    resultValue = leftValue;
                }
                break;
            case TEBopType.Is:
                resultType = TVType.Bool;
                resultValue = left.type === right.type && leftValue === rightValue;
                break;
            case TEBopType.IsNot:
                resultType = TVType.Bool;
                resultValue = left.type !== right.type || leftValue !== rightValue;
                break;
            case TEBopType.In:
            case TEBopType.NotIn:
            default:
                return TVError.create('PyTea Interpreter: Element type is not iterable', source);
        }

        if (resultType === TVType.Bool) {
            return TVBool.create(Boolean(resultValue), source);
        } else if (resultType === TVType.Int) {
            return TVInt.create(Number(resultValue), source);
        } else if (resultType === TVType.Float) {
            return TVFloat.create(Number(resultValue), source);
        } else {
            return TVError.create('PyTea Interpreter: Cannot reach here', source);
        }
    }

    export function evalStringTypeBinOp(left: string, right: string, bop: TEBopType, source?: ParseNode): ThValue {
        let resultType: TVType;
        let resultValue: string | boolean = false;

        switch (bop) {
            case TEBopType.Add:
                resultType = TVType.String;
                resultValue = left + right;
                break;
            case TEBopType.Sub:
            case TEBopType.FloorDiv:
            case TEBopType.TrueDiv:
            case TEBopType.Mod:
                resultType = TVType.Error;
                break;
            case TEBopType.Mul:
                resultType = TVType.NotImpl;
                break;
            case TEBopType.Lt:
                resultType = TVType.Bool;
                resultValue = left < right;
                break;
            case TEBopType.Lte:
                resultType = TVType.Bool;
                resultValue = left <= right;
                break;
            case TEBopType.Eq:
                resultType = TVType.Bool;
                resultValue = left === right;
                break;
            case TEBopType.Neq:
                resultType = TVType.Bool;
                resultValue = left !== right;
                break;
            case TEBopType.And:
                resultType = TVType.String;
                resultValue = left === '' ? left : right;
                break;
            case TEBopType.Or:
                resultType = TVType.String;
                resultValue = left === '' ? right : left;
                break;
            case TEBopType.Is:
                resultType = TVType.Bool;
                resultValue = left === right;
                break;
            case TEBopType.IsNot:
                resultType = TVType.Bool;
                resultValue = left !== right;
                break;
            case TEBopType.In:
            case TEBopType.NotIn:
            default:
                return TVError.create('PyTea Interpreter: Element type is not iterable', source);
        }

        if (resultType === TVType.Bool) {
            return TVBool.create(Boolean(resultValue), source);
        } else if (resultType === TVType.String) {
            return TVString.create(String(resultValue), source);
        } else if (resultType === TVType.NotImpl) {
            return TVNotImpl.create('PyTea Interpreter: Not implemented string operator.', source);
        } else {
            return TVError.create('PyTea Interpreter: Not supported type of operator.', source);
        }
    }

    export function evalElementTypeUnaryOp(base: ThValue, uop: TEUopType, source?: ParseNode): ThValue {
        const value = Number(elementTypeFetchValue(base));

        let resultType: TVType;
        let resultValue: number | boolean;

        switch (uop) {
            case TEUopType.Neg:
                resultType = elementTypeUpperBoundOfTypes([base.type, TVType.Int]);
                resultValue = -Number(elementTypeFetchValue(base));
                break;
            case TEUopType.Not:
                resultType = TVType.Bool;
                resultValue = !value;
                break;
            default:
                return TVError.create('PyTea Interpreter: Cannot reach here.', source);
        }

        if (resultType === TVType.Bool) {
            return TVBool.create(Boolean(resultValue), source);
        } else if (resultType === TVType.Int) {
            return TVInt.create(Number(resultValue), source);
        } else if (resultType === TVType.Float) {
            return TVFloat.create(Number(resultValue), source);
        } else {
            return TVError.create('PyTea Interpreter: Cannot reach here', source);
        }
    }

    export function binOpString(isLeft: boolean, bop: TEBopType): string | undefined {
        switch (bop) {
            case TEBopType.Add:
                return isLeft ? '__add__' : '__radd__';
            case TEBopType.Sub:
                return isLeft ? '__sub__' : '__rsub__';
            case TEBopType.Mul:
                return isLeft ? '__mul__' : '__rmul__';
            case TEBopType.FloorDiv:
                return isLeft ? '__floordiv__' : '__rfloordiv__';
            case TEBopType.Mod:
                return isLeft ? '__mod__' : '__rmod__';
            case TEBopType.TrueDiv:
                return isLeft ? '__truediv__' : '__rtruediv__';
            case TEBopType.Lt:
                return isLeft ? '__lt__' : '__gt__';
            case TEBopType.Lte:
                return isLeft ? '__le__' : '__ge__';
            case TEBopType.Eq:
                // will be exceptionally treated
                return '__eq__';
            case TEBopType.Neq:
                // will be exceptionally treated
                return '__ne__';
            case TEBopType.And:
                return isLeft ? '__and__' : '__rand__';
            case TEBopType.Or:
                return isLeft ? '__or__' : '__ror__';
            case TEBopType.Is:
                // this part should be modified later.
                // it has a hole when `1 is True' (python outputs False).
                // this is different to __eq__.
                return '__eq__';
            case TEBopType.IsNot:
                // similarily, this part should be modified later.
                return '__neq__';
            case TEBopType.In:
                return isLeft ? '__contains__' : undefined;
            case TEBopType.NotIn:
                // will be negated from the result
                return isLeft ? '__contains__' : undefined;
            default:
                return undefined;
        }
    }

    function elementTypeUpperBoundOfBinOp(leftType: TVType, rightType: TVType): TVType {
        if (leftType === TVType.Bool) {
            switch (rightType) {
                case TVType.Bool:
                    return TVType.Bool;
                case TVType.Int:
                    return TVType.Int;
                case TVType.Float:
                    return TVType.Float;
                default:
                    return TVType.Undef;
            }
        } else if (leftType === TVType.Int) {
            switch (rightType) {
                case TVType.Bool:
                    return TVType.Int;
                case TVType.Int:
                    return TVType.Int;
                case TVType.Float:
                    return TVType.Float;
                default:
                    return TVType.Undef;
            }
        } else if (leftType === TVType.Float) {
            switch (rightType) {
                case TVType.Bool:
                    return TVType.Float;
                case TVType.Int:
                    return TVType.Float;
                case TVType.Float:
                    return TVType.Float;
                default:
                    return TVType.Undef;
            }
        } else {
            return TVType.Undef;
        }
    }

    function elementTypeUpperBoundOfTypes(types: TVType[]): TVType {
        const result = TVType.Bool;
        return types.reduce((result, type) => elementTypeUpperBoundOfBinOp(result, type), result);
    }
}

// process Call(f, args)
export function functionCall(
    env: ThEnv,
    heap: ThHeap,
    f: TVFunc,
    args: ThValue[],
    source?: ParseNode,
    kwargs?: { [paramName: string]: ThValue }
): [ThValue, ThHeap] {
    if (!f.funcEnv) {
        // how do I handle this error?
        return [TVError.create('PyTea Interpreter: The environment of a function is not defined.', source), heap];
    }

    let newHeap = heap;

    // process sigma_f[fname->l][params->l']
    // process H[l->f][l'->args]
    const [fAddr, nextHeap1] = newHeap.allocNew(f, source);
    let fEnv = f.funcEnv.setId(f.name, fAddr);
    newHeap = nextHeap1;

    const paramAddrs: Map<string, TVAddr> = new Map();
    f.params.forEach((param) => {
        const [addr, h] = newHeap.malloc();
        newHeap = h;
        fEnv = fEnv.setId(param, addr);
        paramAddrs.set(param, addr);
    });

    // assigning defaults
    f.defaults.forEach((v, k) => {
        if (paramAddrs.has(k)) {
            newHeap = newHeap.setVal(paramAddrs.get(k)!, v);
        }
    });

    // assigning pos args and varargs
    // TODO: set varargTuple as tuple
    const posargsLen = f.params.count() - (f.kwargsParam ? 1 : 0) - (f.varargsParam ? 1 : 0);
    let varargTuple = f.varargsParam ? TVObject.create() : undefined;
    args.forEach((v, i) => {
        if (i < posargsLen) {
            const paramName = f.params.get(i)!;
            newHeap = newHeap.setVal(paramAddrs.get(paramName)!, v);
        } else if (varargTuple) {
            varargTuple = varargTuple.setIndice(i - posargsLen, v);
        }
    });
    if (varargTuple) {
        newHeap = newHeap.setVal(paramAddrs.get(f.varargsParam!)!, varargTuple);
    }

    // assigning kwargs
    // TODO: set kwargDict as dict
    let kwargDict = f.kwargsParam ? TVObject.create() : undefined;
    if (kwargs) {
        Object.entries(kwargs).forEach(([k, v]) => {
            if (paramAddrs.has(k)) {
                newHeap = newHeap.setVal(paramAddrs.get(k)!, v);
            } else if (kwargDict) {
                kwargDict = kwargDict.setKeyVal(k, v);
            }
        });
    }
    if (kwargDict) {
        newHeap = newHeap.setVal(paramAddrs.get(f.kwargsParam!)!, kwargDict);
    }

    // run the function body
    const [value, nextHeap2] = TorchInterpreter.run(fEnv, newHeap, f.funcBody);
    newHeap = nextHeap2;

    // // TODO: check safety.
    // // free malloced paramAddrs
    // paramAddrs.forEach((addr) => {
    //     newHeap = newHeap.free(addr);
    // });
    // newHeap = newHeap.free(fAddr);

    if (value === ThContFlag.Brk || value === ThContFlag.Cnt || value === ThContFlag.Run) {
        // this is right. a function without any return method actually returns None.
        return [TVNone.create(source), newHeap];
    } else {
        return [value, newHeap];
    }
}

export function getItemByIndex(env: ThEnv, heap: ThHeap, obj: TVObject, idx: number): [ThValue, ThHeap] {
    if (obj.getAttr('__getitem__') !== undefined) {
        return functionCall(env, heap, obj.getAttr('__getitem__') as TVFunc, [TVInt.create(idx)]);
    }
    // TODO: keyValues
    // else if ()...
    else {
        const value = obj.getIndice(idx);
        if (value === undefined) {
            return [TVError.create('PyTea Interpreter: Unavailable indexing'), heap];
        }
        return [value, heap];
    }
}

// process evaluate(E) => (H[l]) if H[l] not in {object, undefined}, otherwise (l).
export function endPoint([value, heap]: [ThValue, ThHeap]): [ThValue, ThHeap] {
    if (value.type !== TVType.Addr) {
        return [value, heap];
    }

    const fetch = heap.getVal(value);
    if (fetch === undefined) {
        return [TVError.create('PyTea Interpreter: Try to access non-existing variable.', value.source), heap];
    }
    if (fetch.type === TVType.Addr) {
        // defensive implementation
        return endPoint([fetch, heap]);
    } else if (fetch.type === TVType.Object) {
        return [value, heap];
    } else {
        return [fetch, heap];
    }
}

// follow (potential) address `value` through the heap until value is not an address,
export function fetchAddr(value: ThValue, heap: ThHeap): ThValue | undefined {
    let retVal: ThValue | undefined = value;
    while (retVal && retVal.type === TVType.Addr) {
        retVal = heap.getVal(retVal);
    }

    return retVal;
}
