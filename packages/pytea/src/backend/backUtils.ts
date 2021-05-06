/*
 * backUtils.ts
 * Copyright (c) Seoul National University.
 * Licensed under She MIT license.
 * AuShor: Ho Young Jhoo (mersshs@gmail.com)
 *
 * Utility functions for She interpreter.
 */
import { ParseNode } from 'pyright-internal/parser/parseNodes';

import { TEBinOp, TEBopType, TEUopType } from '../frontend/torchStatements';
import { FilePathStore } from '../service/executionPaths';
import { ConstraintSet } from './constraintSet';
import { Constraint, ConstraintType } from './constraintType';
import { Context, ContextSet } from './context';
import { simplifyBool, simplifyShape } from './expUtils';
import { ShEnv, ShHeap } from './sharpEnvironments';
import {
    CodeSource,
    ShValue,
    SVAddr,
    SVBool,
    SVError,
    SVErrorLevel,
    SVFloat,
    SVInt,
    SVLiteral,
    SVNumber,
    SVNumeric,
    SVNumericType,
    SVSize,
    SVString,
    SVType,
} from './sharpValues';
import { BoolOpType, ExpBool, ExpNum, ExpShape, ExpString, NumBopType, NumUopType } from './symExpressions';
import { TorchBackend } from './torchBackend';

/**
 * Normalize index based on PyShon semantics Shat supports negative index.
 * If `index` is less than 0, return absolute index following lengSh of `arr`
 * return -1 if `index` is out of range.
 */
export function absIndex(arr: string, index: number): number;
export function absIndex<T>(arr: T[], index: number): number;
export function absIndex<T>(arr: T[] | string, index: number): number {
    const len = arr.length;
    return absIndexByLen(len, index);
}
export function absIndexByLen(len: number, index: number): number {
    if (-len <= index && index < 0) {
        return len + index;
    } else if (0 <= index && index < len) {
        return index;
    }
    return -1;
}

// if value is address of address of ... some value, return that value.
export function fetchAddr(value: ShValue | undefined, heap: ShHeap): ShValue | undefined {
    if (value?.type === SVType.Addr) {
        return fetchAddr(heap.getVal(value), heap);
    }

    return value;
}

// if retVal is address of address of ... some value, return that value otherwise the value is object
// if the endpoint is object, return the address of object.
// if retVal is non-address, just return it.
export function sanitizeAddr(retVal: ShValue | undefined, heap: ShHeap): ShValue | undefined {
    if (!retVal || retVal.type !== SVType.Addr) {
        return retVal;
    }

    const fetch = heap.getVal(retVal);

    if (!fetch) {
        return;
    } else if (fetch.type === SVType.Object) {
        // defensive implementation
        return retVal;
    } else {
        return fetch;
    }
}
export function sanitizeAddrCtx(ctx: Context<ShValue>): Context<ShValue> {
    const { heap, retVal } = ctx;

    const fetch = sanitizeAddr(retVal, heap);
    if (fetch === undefined) {
        return ctx.warnWithMsg(`accessed to undefined address: ${retVal.toString()}`, retVal.source);
    }

    return ctx.setRetVal(fetch);
}

export function sanitizeAddrSet(ctxSet: ContextSet<ShValue>): ContextSet<ShValue> {
    return ctxSet.map(sanitizeAddrCtx);
}

export function sanitizeSource(obj: Object, pathStore?: FilePathStore): typeof obj {
    // TODO: map Map
    if (typeof obj !== 'object') {
        return obj;
    }

    if (Array.isArray(obj)) {
        return obj.map((o) => sanitizeSource(o, pathStore));
    }

    const ret: any = {};

    Object.entries(obj).forEach(([k, v]) => {
        if (k === 'source') {
            if (typeof v === 'object') {
                if ('nodeType' in v) {
                    return pathStore?.toCodeRange(v as ParseNode);
                } else if ('fileId' in v) {
                    return v;
                }
            }
            return;
        } else if (Array.isArray(v)) {
            ret[k] = v.map((o) => sanitizeSource(o, pathStore));
        } else {
            ret[k] = sanitizeSource(v, pathStore);
        }
    });

    return ret as typeof obj;
}

// track __mro__ and make address list
// assume that env has type objects like int, float, ...
// WARNING: currently we assume that __mro__ is linked-list-like tuple.
// TODO: resolve multi-inheritance
export function trackMro(value: ShValue, heap: ShHeap, env: ShEnv): (number | undefined)[] {
    // TODO: set list and tuple.
    const list: number[] = [];

    const objectAddr = env.getId('object')?.addr;
    const obj = fetchAddr(value, heap);
    if (!obj) return list;

    switch (obj.type) {
        case SVType.Bool:
            return [(sanitizeAddr(env.getId('bool'), heap) as SVAddr | undefined)?.addr, objectAddr];
        case SVType.Int:
            return [(sanitizeAddr(env.getId('int'), heap) as SVAddr | undefined)?.addr, objectAddr];
        case SVType.Float:
            return [(sanitizeAddr(env.getId('float'), heap) as SVAddr | undefined)?.addr, objectAddr];
        case SVType.String:
            return [(sanitizeAddr(env.getId('str'), heap) as SVAddr | undefined)?.addr, objectAddr];
        case SVType.Object:
            break;
        default:
            return [objectAddr];
    }

    let mro = fetchAddr(obj.getAttr('__mro__'), heap);
    if (mro?.type !== SVType.Object) return [objectAddr];

    let clsAddr: ShValue | undefined = sanitizeAddr(mro.getIndice(0), heap);
    while (clsAddr?.type === SVType.Addr) {
        list.push(clsAddr.addr);
        mro = fetchAddr(clsAddr, heap);
        if (mro?.type !== SVType.Object) break;
        mro = fetchAddr(mro.getAttr('__mro__'), heap);
        if (mro?.type !== SVType.Object) break;
        clsAddr = sanitizeAddr(mro.getIndice(1), heap);
    }

    return list.length > 0 ? list : [objectAddr];
}

// compare two value points same address
export function isSameAddr(val1: ShValue, val2: ShValue, heap: ShHeap): boolean {
    let addr1: ShValue | undefined = val1;
    let addr2: ShValue | undefined = val2;

    while (addr1?.type === SVType.Addr) {
        val1 = addr1;
        addr1 = heap.getVal(addr1);
    }

    while (addr2?.type === SVType.Addr) {
        val2 = addr2;
        addr2 = heap.getVal(addr2);
    }

    return val1.type === SVType.Addr && val2.type === SVType.Addr && val1.addr === val2.addr;
}

// if one cannot say value is truthy or falsy, return a constraint if cannot determine it.
export function isTruthy<T>(ctx: Context<T>, value: ShValue, source: CodeSource | undefined): boolean | Constraint {
    const { heap, ctrSet } = ctx;

    switch (value.type) {
        case SVType.Addr: {
            const obj = heap.getValRecur(value);
            if (!obj) {
                // TODO: really?
                return false;
            }
            return isTruthy(ctx, obj, source);
        }
        case SVType.Bool:
            if (value.value === true || value.value === false) {
                return value.value;
            } else {
                const simpl = simplifyBool(ctx.ctrSet, value.value);
                if (simpl.opType === BoolOpType.Const) {
                    return simpl.value;
                }

                const checked = ctrSet.checkImmediate(simpl);
                if (checked !== undefined) {
                    return checked;
                }
                return ctrSet.genFromBool(value.value, source);
            }
        case SVType.Int:
        case SVType.Float:
            if (typeof value.value === 'number') {
                return value.value !== 0;
            } else {
                const range = ctrSet.getCachedRange(value.value);
                if (!range) return ctrSet.genNumCompare(ConstraintType.LessThan, 0, value.value, source);

                if (range.isTruthy()) return true;
                else if (range.isFalsy()) return false;
                else return ctrSet.genNumCompare(ConstraintType.LessThan, 0, value.value, source);
            }
        case SVType.Object: {
            const length = value.getAttr('$length');
            if (length !== undefined) {
                return isTruthy(ctx, length, source);
            }
            return true;
        }
        case SVType.String:
            if (typeof value.value === 'string') {
                return value.value !== '';
            } else {
                // TODO: cache string.
                return ctrSet.genEquality(
                    ConstraintType.NotEqual,
                    value.value,
                    ExpString.fromConst('', source),
                    source
                );
            }
        case SVType.None:
            return false;
        case SVType.NotImpl:
            return true;
        default:
            // TODO: really?
            return true;
    }
}

export namespace SymOpUtils {
    export function isNumeric(value: ShValue): value is SVNumeric {
        const type = value.type;
        return type === SVType.Bool || type === SVType.Int || type === SVType.Float;
    }

    export function isConstant(type: SVLiteral): boolean {
        const valueType = typeof type.value;
        return valueType === 'boolean' || valueType === 'string' || valueType === 'number';
    }

    // left.value and right.value should be non-symbolic
    export function binOpLiteral(
        left: SVNumeric,
        right: SVNumeric,
        bop: TEBopType,
        source: CodeSource | undefined
    ): ShValue {
        const leftVal = left.value as number | boolean;
        const rightVal = right.value as number | boolean;
        const leftNum = +leftVal;
        const rightNum = +rightVal;

        let resultType: SVType;
        let resultValue: number | boolean;

        switch (bop) {
            case TEBopType.Add:
                resultType = elementTypeUpperBoundOfTypes(left.type, right.type, SVType.Int);
                resultValue = leftNum + rightNum;
                break;
            case TEBopType.Sub:
                resultType = elementTypeUpperBoundOfTypes(left.type, right.type, SVType.Int);
                resultValue = leftNum - rightNum;
                break;
            case TEBopType.Mul:
                resultType = elementTypeUpperBoundOfTypes(left.type, right.type, SVType.Int);
                resultValue = leftNum * rightNum;
                break;
            case TEBopType.Pow:
                resultType = elementTypeUpperBoundOfTypes(left.type, right.type, SVType.Int);
                resultValue = leftNum ** rightNum;
                break;
            case TEBopType.Mod:
                resultType = elementTypeUpperBoundOfTypes(left.type, right.type, SVType.Int);
                resultValue = leftNum % rightNum;
                break;
            case TEBopType.FloorDiv:
                resultType = elementTypeUpperBoundOfTypes(left.type, right.type, SVType.Int);
                resultValue = Math.floor(leftNum / rightNum);
                break;
            case TEBopType.TrueDiv:
                resultType = elementTypeUpperBoundOfTypes(left.type, right.type, SVType.Float);
                resultValue = leftNum / rightNum;
                break;
            case TEBopType.Lt:
                resultType = SVType.Bool;
                resultValue = leftNum < rightNum;
                break;
            case TEBopType.Lte:
                resultType = SVType.Bool;
                resultValue = leftNum <= rightNum;
                break;
            case TEBopType.Eq:
                resultType = SVType.Bool;
                resultValue = leftNum === rightNum;
                break;
            case TEBopType.Neq:
                resultType = SVType.Bool;
                resultValue = leftNum !== rightNum;
                break;
            case TEBopType.Is:
                resultType = SVType.Bool;
                resultValue = left.type === right.type && leftVal === rightVal;
                break;
            case TEBopType.IsNot:
                resultType = SVType.Bool;
                resultValue = left.type !== right.type || leftVal !== rightVal;
                break;
            case TEBopType.In:
            case TEBopType.NotIn:
            default:
                return SVError.create('value is not iterable', SVErrorLevel.Warning, source);
        }

        if (resultType === SVType.Bool) {
            return SVBool.create(Boolean(resultValue), source);
        } else if (resultType === SVType.Int) {
            return SVInt.create(Number(resultValue), source);
        } else {
            return SVFloat.create(Number(resultValue), source);
        }
    }

    // cast boolean to number or number to boolean before use it.
    export function binOpNum(left: SVNumber, right: SVNumber, bop: TEBopType, source: CodeSource | undefined): ShValue {
        let resultType: SVType;
        let resultValue: ExpNum | ExpBool;

        if (isConstant(left) && isConstant(right)) {
            return binOpLiteral(left, right, bop, source);
        }

        const leftVal = left.value;
        const rightVal = right.value;
        const leftExp = typeof leftVal === 'number' ? ExpNum.fromConst(leftVal, source) : leftVal;
        const rightExp = typeof rightVal === 'number' ? ExpNum.fromConst(rightVal, source) : rightVal;

        switch (bop) {
            case TEBopType.Add:
                resultType = elementTypeUpperBoundOfTypes(left.type, right.type, SVType.Int);
                resultValue = ExpNum.bop(NumBopType.Add, leftVal, rightVal, source);
                break;
            case TEBopType.Sub:
                resultType = elementTypeUpperBoundOfTypes(left.type, right.type, SVType.Int);
                resultValue = ExpNum.bop(NumBopType.Sub, leftVal, rightVal, source);
                break;
            case TEBopType.Mul:
                resultType = elementTypeUpperBoundOfTypes(left.type, right.type, SVType.Int);
                resultValue = ExpNum.bop(NumBopType.Mul, leftVal, rightVal, source);
                break;
            case TEBopType.Pow:
                resultType = elementTypeUpperBoundOfTypes(left.type, right.type, SVType.Int);
                return SVError.create(`symbolic pow will be executed from _evalBinOp`, SVErrorLevel.Warning, source);
            case TEBopType.FloorDiv:
                resultType = elementTypeUpperBoundOfTypes(left.type, right.type, SVType.Int);
                resultValue = ExpNum.bop(NumBopType.FloorDiv, leftVal, rightVal, source);
                break;
            case TEBopType.TrueDiv:
                resultType = elementTypeUpperBoundOfTypes(left.type, right.type, SVType.Float);
                resultValue = ExpNum.bop(NumBopType.TrueDiv, leftVal, rightVal, source);
                break;
            case TEBopType.Mod:
                resultType = elementTypeUpperBoundOfTypes(left.type, right.type, SVType.Int);
                resultValue = ExpNum.bop(NumBopType.Mod, leftVal, rightVal, source);
                break;
            case TEBopType.Lt:
                resultType = SVType.Bool;
                resultValue = ExpBool.lt(leftVal, rightVal, source);
                break;
            case TEBopType.Lte:
                resultType = SVType.Bool;
                resultValue = ExpBool.lte(leftVal, rightVal, source);
                break;
            case TEBopType.Eq:
                resultType = SVType.Bool;
                resultValue = ExpBool.eq(leftExp, rightExp, source);
                break;
            case TEBopType.Neq:
                resultType = SVType.Bool;
                resultValue = ExpBool.neq(leftExp, rightExp, source);
                break;
            case TEBopType.Is:
                resultType = SVType.Bool;
                resultValue =
                    left.type === right.type ? ExpBool.eq(leftExp, rightExp, source) : ExpBool.fromConst(false, source);
                break;
            case TEBopType.IsNot:
                resultType = SVType.Bool;
                resultValue =
                    left.type === right.type ? ExpBool.neq(leftExp, rightExp, source) : ExpBool.fromConst(true, source);
                break;
            case TEBopType.In:
            case TEBopType.NotIn:
            default:
                return SVError.create(
                    `invalid operation for numeric values: got (${TEBinOp.toStringBop(bop)})`,
                    SVErrorLevel.Warning,
                    source
                );
        }

        if (resultType === SVType.Bool) {
            return SVBool.create(resultValue as ExpBool, source);
        } else if (resultType === SVType.Int) {
            return SVInt.create(resultValue as ExpNum, source);
        } else {
            return SVFloat.create(resultValue as ExpNum, source);
        }
    }

    export function binOpStrLiteral(
        ctrSet: ConstraintSet,
        left: string,
        right: string,
        bop: TEBopType,
        source: CodeSource | undefined
    ): ShValue | undefined {
        switch (bop) {
            case TEBopType.Add:
                return SVString.create(left + right, source);
            case TEBopType.Lt:
                return SVBool.create(left < right, source);
            case TEBopType.Lte:
                return SVBool.create(left <= right, source);
            case TEBopType.Eq:
                return SVBool.create(left === right, source);
            case TEBopType.Neq:
                return SVBool.create(left !== right, source);
            case TEBopType.Is:
                return SVBool.create(left === right, source);
            case TEBopType.IsNot:
                return SVBool.create(left !== right, source);
            case TEBopType.In:
                return SVBool.create(right.search(left) !== -1, source);
            case TEBopType.NotIn:
                return SVBool.create(right.search(left) === -1, source);
            case TEBopType.Mod:
                // TODO: format string
                return SVString.create(ExpString.fromSymbol(ctrSet.genSymString('str_format', source)), source);
            default:
                return;
        }
    }

    // return undefined if unsupported operation.
    export function binOpStr(
        ctrSet: ConstraintSet,
        left: SVString,
        right: SVString,
        bop: TEBopType,
        source?: CodeSource | undefined
    ): ShValue | undefined {
        if (typeof left.value === 'string' && typeof right.value === 'string') {
            return binOpStrLiteral(ctrSet, left.value, right.value, bop, source);
        }

        const leftExp = typeof left.value === 'string' ? ExpString.fromConst(left.value, source) : left.value;
        const rightExp = typeof right.value === 'string' ? ExpString.fromConst(right.value, source) : right.value;
        let symName: string | undefined;

        switch (bop) {
            case TEBopType.Add:
                return SVString.create(ExpString.concat(leftExp, rightExp, source), source);
            case TEBopType.Eq:
            case TEBopType.Is:
                return SVBool.create(ExpBool.eq(leftExp, rightExp, source), source);
            case TEBopType.Neq:
            case TEBopType.IsNot:
                return SVBool.create(ExpBool.neq(leftExp, rightExp, source), source);
            case TEBopType.Mod:
                // TODO: format string
                return SVString.create(ExpString.fromSymbol(ctrSet.genSymString('str_format', source)), source);
            case TEBopType.Lt:
                symName = 'bop_lt';
                break;
            case TEBopType.Lte:
                symName = 'bop_lte';
                break;
            case TEBopType.In:
                symName = 'bop_lt';
                break;
            case TEBopType.NotIn:
                symName = 'bop_lt';
                break;
            default:
                return;
        }

        // not implementable in our resolver.
        if (symName) return SVBool.create(ExpBool.fromSymbol(ctrSet.genSymBool(symName, source)), source);

        return;
    }

    // return undefined if unsupported operation
    export function binOpStrNum(
        ctrSet: ConstraintSet,
        left: SVString,
        right: SVNumeric,
        bop: TEBopType,
        source?: CodeSource | undefined
    ): ShValue | undefined {
        switch (bop) {
            case TEBopType.Mul: {
                if (right.type !== SVType.Int) return;

                const numRng = ctrSet.getCachedRange(right.value);
                if (numRng?.isConst() !== true || numRng.start < 0) {
                    return SVString.create(ExpString.fromSymbol(ctrSet.genSymString('str_mul', source)), source);
                }

                const repeatN = numRng.start;
                if (typeof left.value === 'string') {
                    return SVString.create(left.value.repeat(repeatN), source);
                } else {
                    // TODO: concat n times.
                    switch (repeatN) {
                        case 0:
                            return SVString.create('', source);
                        case 1:
                            return left;
                        case 2:
                            return SVString.create(ExpString.concat(left.value, left.value, source), source);
                        default:
                            return SVString.create(
                                ExpString.fromSymbol(ctrSet.genSymString('str_mul', source)),
                                source
                            );
                    }
                }
            }
            case TEBopType.Eq:
            case TEBopType.Is:
                return SVBool.create(false, source);
            case TEBopType.Neq:
            case TEBopType.IsNot:
                return SVBool.create(true, source);
            default:
                return;
        }
    }

    export function unaryOp(base: SVNumeric, uop: TEUopType, source: CodeSource | undefined): ShValue {
        let resultType: SVNumericType;
        let result: number | boolean | ExpNum | ExpBool;

        switch (uop) {
            case TEUopType.Neg:
                resultType = base.type === SVType.Float ? SVType.Float : SVType.Int;
                if (isConstant(base)) {
                    result = -base.value;
                } else if (base.type === SVType.Bool) {
                    // already casted. does not happen.
                    return SVError.warn('do bool2int cast before calling unaryOp Neg', source);
                } else {
                    // Int/Float
                    result = ExpNum.uop(NumUopType.Neg, base.value, source);
                }
                break;
            case TEUopType.Not:
                resultType = SVType.Bool;
                if (isConstant(base)) {
                    result = !base.value;
                } else if (base.type === SVType.Bool) {
                    result = ExpBool.not(base.value as ExpBool, source);
                } else {
                    return SVError.warn('do bool2int cast before calling unaryOp Not', source);
                }
                break;
        }

        switch (resultType) {
            case SVType.Bool:
                return SVBool.create(result as boolean | ExpBool, source);
            case SVType.Int:
                return SVInt.create(result as number | ExpNum, source);
            case SVType.Float:
                return SVFloat.create(result as number | ExpNum, source);
        }
    }

    function elementTypeUpperBoundOfBinOp(leftType: SVType, rightType: SVType): SVType {
        if (leftType === SVType.Bool) {
            switch (rightType) {
                case SVType.Bool:
                    return SVType.Bool;
                case SVType.Int:
                    return SVType.Int;
                case SVType.Float:
                    return SVType.Float;
                default:
                    return SVType.Undef;
            }
        } else if (leftType === SVType.Int) {
            switch (rightType) {
                case SVType.Bool:
                    return SVType.Int;
                case SVType.Int:
                    return SVType.Int;
                case SVType.Float:
                    return SVType.Float;
                default:
                    return SVType.Undef;
            }
        } else if (leftType === SVType.Float) {
            switch (rightType) {
                case SVType.Bool:
                    return SVType.Float;
                case SVType.Int:
                    return SVType.Float;
                case SVType.Float:
                    return SVType.Float;
                default:
                    return SVType.Undef;
            }
        } else {
            return SVType.Undef;
        }
    }

    export function elementTypeUpperBoundOfTypes(...types: SVType[]): SVType {
        const result = SVType.Bool;
        return types.reduce((result, type) => elementTypeUpperBoundOfBinOp(result, type), result);
    }

    export const operatorMap: { [operator: number]: [string, string] } = {
        [TEBopType.Add]: ['__add__', '__radd__'],
        [TEBopType.Sub]: ['__sub__', '__rsub__'],
        [TEBopType.Mul]: ['__mul__', '__rmul__'],
        [TEBopType.FloorDiv]: ['__floordiv__', '__rfloordiv__'],
        [TEBopType.TrueDiv]: ['__truediv__', '__rtruediv__'],
        [TEBopType.Mod]: ['__mod__', '__rmod__'],
        [TEBopType.Pow]: ['__pow__', '__rpow__'],
        [TEBopType.And]: ['__and__', '__rand__'],
        [TEBopType.Or]: ['__or__', '__ror__'],
        [TEBopType.Lt]: ['__lt__', '__gt__'],
        [TEBopType.Lte]: ['__le__', '__ge__'],
        [TEBopType.Eq]: ['__eq__', '__eq__'],
        [TEBopType.Neq]: ['__ne__', '__ne__'],
        [TEBopType.Is]: ['__eq__', '__eq__'],
        [TEBopType.IsNot]: ['__ne__', '__ne__'], // front will map 'A is not B' to 'not (A is B)'
        [TEBopType.In]: ['__contains__', '__contains__'],
        [TEBopType.NotIn]: ['__contains__', '__contains__'], // front will map 'A not in B' to 'not (A in B)'
    };
}

export function genTorchSize<T>(ctx: Context<T>, shape: ExpShape, source: CodeSource | undefined): ContextSet<ShValue> {
    shape = simplifyShape(ctx.ctrSet, shape);
    const size = SVSize.createSize(ctx, shape, source);

    return TorchBackend.libClassInit(ctx, 'torch.Size', [size.retVal], source);
}

export function genTensor<T>(ctx: Context<T>, shape: ExpShape, source: CodeSource | undefined): ContextSet<ShValue> {
    shape = simplifyShape(ctx.ctrSet, shape);
    const size = SVSize.createSize(ctx, shape, source);

    return TorchBackend.libClassInit(ctx, 'torch.Tensor', [size.retVal], source);
}

export function isSize(value: ShValue | undefined): value is SVSize {
    return value !== undefined && value instanceof SVSize;
}

// return either size of tensor in `mayAddr` or return error message
export function fetchSize(value: ShValue | undefined, heap: ShHeap): SVSize | string {
    const obj = fetchAddr(value, heap);
    if (obj?.type !== SVType.Object) {
        return `value is not sized type. got ${ShValue.toStringType(obj?.type)}`;
    }

    const size = fetchAddr(obj.getAttr('shape'), heap);
    if (!isSize(size)) {
        return `attribute 'shape' is not a Size object`;
    }

    return size;
}
