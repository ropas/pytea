/*
 * context.ts
 * Copyright (c) Seoul National University.
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo (mersshs@gmail.com)
 *
 * Context for backend processing.
 * Collection of Environment, Heap, and Constraint set.
 */
import { List, Map, Record, Set } from 'immutable';

import { getFileInfo } from 'pyright-internal/analyzer/analyzerNodeInfo';
import { ParseNode, ParseNodeType } from 'pyright-internal/parser/parseNodes';

import { fetchAddr, genTensor } from './backUtils';
import { ConstraintSet } from './constraintSet';
import {
    Constraint,
    ConstraintType,
    CtrAnd,
    CtrBroad,
    CtrEq,
    CtrFail,
    CtrForall,
    CtrLt,
    CtrLte,
    CtrNeq,
    CtrNot,
    CtrOr,
    ctrToStr,
    extractSymbols,
} from './constraintType';
import { isStructuallyEq, simplifyExp, simplifyShape } from './expUtils';
import { NumRange } from './range';
import { ShEnv, ShHeap } from './sharpEnvironments';
import {
    CodeSource,
    ShContFlag,
    ShValue,
    SVAddr,
    SVBool,
    SVError,
    SVErrorLevel,
    SVFloat,
    SVFunc,
    SVInt,
    SVNone,
    SVObject,
    SVSize,
    SVString,
    SVType,
} from './sharpValues';
import {
    ExpBool,
    ExpNum,
    ExpNumSymbol,
    ExpShape,
    ExpShapeConst,
    NumBopType,
    SymBool,
    SymExp,
    SymFloat,
    SymInt,
    SymShape,
    SymString,
} from './symExpressions';
import { TorchBackend } from './torchBackend';

export type CtxExpr = ContextSet<ShValue>;
export type CtxStmt = ContextSet<ShValue | ShContFlag>;

let _failId = 0;
function getFailedId(): number {
    return ++_failId;
}
const LOG_IGNORE = /pylib.*(tensor|functional|builtins|linear|module|loss|math).py$/;
function checkIgnorePath(path: string) {
    return LOG_IGNORE.test(path);
}

interface ContextProps<T> {
    failId: number;
    env: ShEnv;
    heap: ShHeap;
    ctrSet: ConstraintSet;
    retVal: T;

    // the properties below are internal values; the user should not directly modify these values.
    // SVFunc is a type of Python function, string is a name of LibCall.
    callStack: List<[SVFunc | string, CodeSource | undefined]>;
    logs: List<ShValue>;
    imported: ShEnv; // qualPath (relative to project root or site-packages) to address.
    relPath: string; // relative path to entry file. starts with entry file name.

    // if it is set, this context is regarded as a failed path.
    failed?: SVError;
}

interface ContextMethods<T> {
    // make set
    toSet(): ContextSet<T>;
    toSetWith<A>(retVal: A): ContextSet<A>;

    // property setters.
    setEnv(env: ShEnv): Context<T>;
    setHeap(heap: ShHeap): Context<T>;
    setCtrSet(ctrSet: ConstraintSet): Context<T>;
    setRetVal<A>(retVal: A): Context<A>;
    setRelPath(relPath: string): Context<T>;
    setImported(imported: ShEnv): Context<T>;

    getAttrDeep(value: ShValue, attr: string, source: CodeSource | undefined): ContextSet<ShValue>;
    getIndiceDeep(value: ShValue, index: number, source: CodeSource | undefined): ContextSet<ShValue>;
    getKeyValDeep(value: ShValue, key: string, source: CodeSource | undefined): ContextSet<ShValue>;

    addLog(message: string, source: CodeSource | undefined): Context<T>;
    addLogValue(log: ShValue): Context<T>;
    pushCallStack(stack: [SVFunc | string, ParseNode | undefined]): Context<T>;
    popCallStack(): Context<T>;

    // these methods does not cut off paths. just log warnings
    warn(warning: SVError): Context<SVError>;
    warnWithMsg(message: string, source: CodeSource | undefined): Context<SVError>;
    warnSize(warning: SVError): Context<SVSize>;
    warnSizeWithMsg(message: string, source: CodeSource | undefined): Context<SVSize>;
    warnTensor(warning: SVError): ContextSet<ShValue>;
    warnTensorWithMsg(message: string, source: CodeSource | undefined): ContextSet<ShValue>;

    // these two methods cut off paths
    fail(error: SVError): Context<SVError>;
    failWithMsg(message: string, source: CodeSource | undefined): Context<SVError>;

    // primitive collections
    genList(values: ShValue[], source: CodeSource | undefined): [SVObject, SVAddr, Context<T>];
    genTuple(values: ShValue[], source: CodeSource | undefined): [SVObject, SVAddr, Context<T>];

    // symbolic variable generator
    genSymInt(name: string, source: CodeSource | undefined): SymInt;
    genSymFloat(name: string, source: CodeSource | undefined): SymFloat;
    genSymBool(name: string, source: CodeSource | undefined): SymBool;
    genSymString(name: string, source: CodeSource | undefined): SymString;
    genSymShape(name: string, rank: ExpNum, source: CodeSource | undefined): SymShape;

    // return ExpNumSymbol that is greater than `value`;
    genIntGte(name: string, value: number | ExpNum, source: CodeSource | undefined): Context<ExpNumSymbol>;
    genFloatGte(name: string, value: number | ExpNum, source: CodeSource | undefined): Context<ExpNumSymbol>;

    // generate constant-ranked shape. all the dimensions is new symbolic number gte 0. should check rank >= 0 before call it.
    // partialDims is additional constant part of dimensions
    genConstRankedShape(
        rank: number,
        source: CodeSource | undefined,
        partialDims?: Map<number, ExpNum>
    ): ContextSet<ExpShapeConst>;

    // generate ranked shape. all the dimensions is new symbolic number gte 0.
    // partialDims is additional constant part of dimensions
    // if rank has upper-bound, return slice of upper-bound-ranked shape. (for optimization)
    // in production, ranked has force-upper bound of some number (e.g. 6)
    genRankedShape(rank: number | ExpNum, source: CodeSource | undefined): ContextSet<ExpShape>;

    // constraint generator
    genBool(pred: ExpBool | boolean, source: CodeSource | undefined): Constraint;
    genEq(
        left: SymExp | number | string | boolean,
        right: SymExp | number | string | boolean,
        source: CodeSource | undefined
    ): CtrEq;
    genNeq(left: SymExp | number | string, right: SymExp | number | string, source: CodeSource | undefined): CtrNeq;
    genLt(left: ExpNum | number, right: ExpNum | number, source: CodeSource | undefined): CtrLt;
    genLte(left: ExpNum | number, right: ExpNum | number, source: CodeSource | undefined): CtrLte;
    genAnd(left: Constraint, right: Constraint, source: CodeSource | undefined): CtrAnd;
    genOr(left: Constraint, right: Constraint, source: CodeSource | undefined): CtrOr;
    genNot(constraint: Constraint, source: CodeSource | undefined): CtrNot;
    genForall(
        symbol: SymInt,
        range: [number | ExpNum, number | ExpNum],
        constraint: Constraint,
        source: CodeSource | undefined
    ): CtrForall;
    genBroadcastable(left: ExpShape, right: ExpShape, source: CodeSource | undefined): CtrBroad;
    genFail(reason: string, source: CodeSource | undefined): CtrFail;

    // shape operations

    // make `sizable` to shape, if `sizable` may be iterable object of integer.
    // if parse error is not critical, return parse error messages
    parseSize(iterable: ShValue, source: CodeSource | undefined): ContextSet<ExpShape | string>;

    // matmul between shapes which rank is greater than 1
    // if rank is greater than 2, it follows the matrix multiplication broadcasting rule of numpy.
    shMatmul(left: ExpShape, right: ExpShape, source: CodeSource | undefined): ContextSet<ExpShape>;

    // return broadcasted shape
    shBroadcast(left: ExpShape, right: ExpShape, source: CodeSource | undefined): ContextSet<ExpShape>;

    // return shape without `axis`-th. `axis` is 0-based
    shReduce(shape: ExpShape, axis: ExpNum | number, source: CodeSource | undefined): ContextSet<ExpShape>;

    // repeat `shape` by `count` times and stack along `axis`. `axis` is 0-based
    shRepeat(
        shape: ExpShape,
        axis: ExpNum | number,
        count: ExpNum | number,
        source: CodeSource | undefined
    ): ContextSet<ExpShape>;

    // constraint injector
    // failed immediately if `ctr` is false. (soft-constraint)
    require(ctr: Constraint | Constraint[], failMsg: string, source: CodeSource | undefined): ContextSet<T | SVError>;

    // add admitted constraints (hard-constraint)
    guarantee(ctr: Constraint | Constraint[]): Context<T>;

    // return both if-path and else-path.
    ifThenElse(ctr: Constraint, source: CodeSource | undefined): [ContextSet<T>, ContextSet<T>];

    // immediate linear SMT : generates conservative range.
    getCachedRange(num: number | ExpNum): NumRange | undefined; // return conservative range
    checkImmediate(constraint: Constraint): boolean | undefined; // return true if always true, false if always false, undefined if don't know

    // check if this ctx has path constraints.
    hasPathCtr(): boolean;

    // check this path is timeouted
    isTimedOut(): boolean;
}

export interface ContextSet<T> {
    // to distinguish context and contextset
    env: false;
    setCtxList<A>(ctxList: List<Context<A>>): ContextSet<A>;

    filter(tester: (ctx: Context<T>) => boolean): ContextSet<T>;
    map<A>(mapper: (ctx: Context<T>) => Context<A>): ContextSet<A>;
    flatMap<A>(mapper: (ctx: Context<T>) => ContextSet<A>): ContextSet<A>;

    return<A>(retVal: A): ContextSet<A>;
    fail(errMsg: string, source: CodeSource | undefined): ContextSet<SVError>;
    join<A>(ctxSet: ContextSet<A>): ContextSet<A | T>;

    require(ctr: Constraint | Constraint[], failMsg: string, source: CodeSource | undefined): ContextSet<T | SVError>;
    guarantee(ctr: Constraint | Constraint[]): ContextSet<T>;
    ifThenElse(ctr: Constraint, source: CodeSource | undefined): [ContextSet<T>, ContextSet<T>];

    getList(): List<Context<T>>;
    getFailed(): List<Context<SVError>>;
    getStopped(): List<Context<SVError>>;
    getRunningCount(): number;

    isEmpty(): boolean;
    addLog(message: string, source: CodeSource | undefined): ContextSet<T>;
    addLogValue(value: ShValue): ContextSet<T>;
    prunePureFunctionCall(oldCtx: Context<unknown>, symIdMax: number): ContextSet<T>;
}

const contextDefaults: ContextProps<unknown> = {
    failId: -1,
    env: new ShEnv(),
    heap: new ShHeap(),
    ctrSet: new ConstraintSet(),
    retVal: undefined,

    callStack: List(),
    logs: List(),
    imported: new ShEnv(),
    relPath: '.',

    failed: undefined,
};

export class Context<T> extends Record(contextDefaults) implements ContextProps<T>, ContextMethods<T> {
    retVal!: T;

    constructor(values?: Partial<ContextProps<T>>) {
        values ? super(values) : super();
    }

    setEnv(env: ShEnv): Context<T> {
        return this.set('env', env);
    }

    setHeap(heap: ShHeap): Context<T> {
        return this.set('heap', heap);
    }

    setCtrSet(ctrSet: ConstraintSet): Context<T> {
        return this.set('ctrSet', ctrSet);
    }

    setRetVal<A>(retVal: A): Context<A> {
        // WARNING: use unknown type hack due to the TProps type parameter of Record
        //          follows type of contextDefaults (ContextProps<unknown>)
        return ((this as unknown) as Context<A>).set('retVal', retVal);
    }

    setRelPath(relPath: string): Context<T> {
        return this.set('relPath', relPath);
    }

    setImported(imported: ShEnv): Context<T> {
        return this.set('imported', imported);
    }

    getAttrDeep(value: ShValue, attr: string, source: CodeSource | undefined): ContextSet<ShValue> {
        return TorchBackend.getAttrDeep(this, value, attr, source);
    }

    getIndiceDeep(value: ShValue, index: number | ExpNum, source: CodeSource | undefined): ContextSet<ShValue> {
        return TorchBackend.getIndiceDeep(this, value, index, source);
    }

    getKeyValDeep(value: ShValue, key: string, source: CodeSource | undefined): ContextSet<ShValue> {
        return TorchBackend.getKeyValDeep(this, value, key, source);
    }

    // these two methods does not cut off paths. just log warnings
    warn(warning: SVError): Context<SVError> {
        return this.setRetVal(warning).addLogValue(warning);
    }
    warnWithMsg(message: string, source: CodeSource | undefined): Context<SVError> {
        source = this._replaceBuiltinSource(source);
        const warning = SVError.create(message, SVErrorLevel.Warning, source);
        return this.setRetVal(warning).addLogValue(warning);
    }

    // return fully symbolic shaped tensor with warning log.
    warnTensor(warning: SVError): ContextSet<ShValue> {
        const ctx = this.addLogValue(warning);
        const source = warning.source;
        const rank = ctx.genSymInt('WarnTempRank', source);
        const shape = ctx.genSymShape('WarnTempShape', ExpNum.fromSymbol(rank), source);
        return genTensor(ctx, ExpShape.fromSymbol(shape), source);
    }
    // return fully symbolic shaped tensor with warning log.
    warnTensorWithMsg(message: string, source: CodeSource | undefined): ContextSet<ShValue> {
        const warning = SVError.create(message, SVErrorLevel.Warning, source);
        return this.warnTensor(warning);
    }

    // return fully symbolic shape with warning log.
    warnSize(warning: SVError): Context<SVSize> {
        const ctx = this.addLogValue(warning);
        const source = warning.source;
        const rank = ctx.genSymInt('WarnTempRank', source);
        const shape = ctx.genSymShape('WarnTempShape', ExpNum.fromSymbol(rank), source);
        return SVSize.createSize(ctx, ExpShape.fromSymbol(shape), source);
    }
    // return fully symbolic Size with warning log.
    warnSizeWithMsg(message: string, source: CodeSource | undefined): Context<SVSize> {
        const warning = SVError.create(message, SVErrorLevel.Warning, source);
        return this.warnSize(warning);
    }

    // these two methods cut off paths
    fail(error: SVError): Context<SVError> {
        return this.set('failed', error).set('failId', getFailedId()).addLogValue(error).setRetVal(error);
    }
    failWithMsg(message: string, source: CodeSource | undefined): Context<SVError> {
        source = this._replaceBuiltinSource(source);
        return this.fail(SVError.create(message, SVErrorLevel.Error, source));
    }

    addLog(message: string, source: CodeSource | undefined): Context<T> {
        source = this._replaceBuiltinSource(source);
        return this.set('logs', this.logs.push(SVError.create(message, SVErrorLevel.Log, source)));
    }

    addLogValue(log: ShValue): Context<T> {
        return this.set('logs', this.logs.push(log));
    }

    pushCallStack(stack: [SVFunc | string, CodeSource | undefined]): Context<T> {
        return this.set('callStack', this.callStack.push(stack));
    }

    popCallStack(): Context<T> {
        return this.set('callStack', this.callStack.pop());
    }

    toSet(): ContextSet<T> {
        return ContextSetImpl.fromCtx(this);
    }

    toSetWith<A>(retVal: A): ContextSet<A> {
        return ContextSetImpl.fromCtx(this.setRetVal(retVal));
    }

    static getEmptySet(): ContextSet<never> {
        return new ContextSetImpl(List());
    }

    // shift every addresses to negative.
    asDefault(): Context<T> {
        const offset = -this.heap.addrMax - 1;
        return this.setEnv(this.env.addOffset(offset)).setHeap(this.heap.addOffset(offset));
    }

    // primitive collections generator
    genList(values: ShValue[], source: CodeSource | undefined): [SVObject, SVAddr, Context<T>] {
        const { heap, env } = this;
        const [list, listAddr, heap2] = SVObject.create(heap, source);
        const listMro = (fetchAddr(heap.getVal(env.getId('list')!)!, heap) as SVObject).getAttr('__mro__')!;
        let listVal = list;
        values.forEach((v, i) => {
            listVal = listVal.setIndice(i, v);
        });
        listVal = listVal.setAttr('$length', SVInt.create(values.length, source)).setAttr('__mro__', listMro);

        return [list, listAddr, this.setHeap(heap2.setVal(listAddr, listVal))];
    }

    genTuple(values: ShValue[], source: CodeSource | undefined): [SVObject, SVAddr, Context<T>] {
        const { heap, env } = this;
        const [tuple, tupleAddr, heap2] = SVObject.create(heap, source);
        const tupleMro = (fetchAddr(heap.getVal(env.getId('tuple')!)!, heap) as SVObject).getAttr('__mro__')!;
        let listVal = tuple;
        values.forEach((v, i) => {
            listVal = listVal.setIndice(i, v);
        });
        listVal = listVal.setAttr('$length', SVInt.create(values.length, source)).setAttr('__mro__', tupleMro);

        return [tuple, tupleAddr, this.setHeap(heap2.setVal(tupleAddr, listVal))];
    }

    // symbolic variable generator
    genSymInt(name: string, source: CodeSource | undefined): SymInt {
        return this.ctrSet.genSymInt(name, source);
    }
    genSymFloat(name: string, source: CodeSource | undefined): SymFloat {
        return this.ctrSet.genSymFloat(name, source);
    }
    genSymBool(name: string, source: CodeSource | undefined): SymBool {
        return this.ctrSet.genSymBool(name, source);
    }
    genSymString(name: string, source: CodeSource | undefined): SymString {
        return this.ctrSet.genSymString(name, source);
    }
    genSymShape(name: string, rank: ExpNum, source: CodeSource | undefined): SymShape {
        return this.ctrSet.genSymShape(name, rank, source);
    }

    // return ExpNumSymbol that is equal to `value`;
    genIntEq(name: string, value: number | ExpNum, source: CodeSource | undefined): Context<ExpNumSymbol> {
        source = this._replaceBuiltinSource(source);
        const [num, ctrSet] = this.ctrSet.genSymIntEq(name, value, source);
        const exp = ExpNum.fromSymbol(num);
        return this.setCtrSet(ctrSet).setRetVal(exp);
    }

    // return ExpNumSymbol that is greater than `value`;
    genIntGte(name: string, value: number | ExpNum, source: CodeSource | undefined): Context<ExpNumSymbol> {
        const num = this.genSymInt(name, source);
        const exp = ExpNum.fromSymbol(num);
        const pos = this.genLte(value, exp, source);
        return this.guarantee(pos).setRetVal(exp);
    }

    genFloatGte(name: string, value: number | ExpNum, source: CodeSource | undefined): Context<ExpNumSymbol> {
        const num = this.genSymFloat(name, source);
        const exp = ExpNum.fromSymbol(num);
        const pos = this.genLte(value, exp, source);
        return this.guarantee(pos).setRetVal(exp);
    }

    // generate constant-ranked shape. all the dimensions is new symbolic number gte.
    // partialDims is additional constant part of dimensions
    genConstRankedShape(
        rank: number,
        source: CodeSource | undefined,
        partialDims?: Map<number, ExpNum>
    ): ContextSet<ExpShapeConst> {
        source = this._replaceBuiltinSource(source);
        if (rank < 0)
            return (this.failWithMsg(
                `from 'genRankedShape': got negative rank ${rank}`,
                source
            ).toSet() as ContextSet<unknown>) as ContextSet<ExpShapeConst>;

        const dims: ExpNum[] = [];
        let ctx: Context<any> = this;

        for (let i = 0; i < rank; i++) {
            const dimPart = partialDims?.get(i);
            if (dimPart) {
                dims.push(dimPart);
                continue;
            }
            const minDimPart = partialDims?.get(-rank + i);
            if (minDimPart) {
                dims.push(minDimPart);
                continue;
            }

            const temp = ctx.genIntGte(`tempDim${i}`, 0, source);
            const dim = temp.retVal;
            dims.push(dim);
            ctx = temp;
        }

        const parts: ExpNum[] = partialDims ? [...partialDims.values()] : [];
        if (parts.length > 0) {
            return ctx
                .require(
                    parts.map((num) => ctx.genLte(0, num, source)),
                    `from 'genRankedShape': got negative partialDims`,
                    source
                )
                .return(ExpShape.fromConst(rank, dims, source));
        }

        return ctx.setRetVal(ExpShape.fromConst(rank, dims, source)).toSet();
    }

    // generate ranked shape. all the dimensions is new symbolic number gte 0.
    // partialDims is additional constant part of dimensions
    // if rank has upper-bound, return slice of upper-bound-ranked shape. (for optimization)
    // in production, ranked has force-upper bound of some number (e.g. 6)
    genRankedShape(
        rank: number | ExpNum,
        source: CodeSource | undefined,
        partialDims?: Map<number, ExpNum>
    ): ContextSet<ExpShape> {
        source = this._replaceBuiltinSource(source);
        if (typeof rank === 'number') {
            return this.genConstRankedShape(rank, source, partialDims);
        }

        const rankRng = this.getCachedRange(rank);
        if (!rankRng || rankRng.lt(0)) {
            return (this.failWithMsg(
                `from 'genRankedShape': invalid rank ${ExpNum.toString(rank)}`,
                source
            ).toSet() as ContextSet<unknown>) as ContextSet<ExpShape>;
        }

        if (rankRng.isConst()) {
            return this.genConstRankedShape(rankRng.end, source, partialDims);
        }

        const dimValues: ExpNum[] = partialDims ? [...partialDims.values()] : [];
        const ctxSet: ContextSet<any> =
            dimValues.length > 0
                ? this.require(
                      dimValues.map((num) => this.genLte(0, num, source)),
                      `from 'genRankedShape': got negative partialDims`,
                      source
                  )
                : this.toSet();

        return ctxSet
            .require(this.genLte(0, rank, source), `genRankedShape got negative rank ${rank}`, source)
            .flatMap((ctx) => {
                if (rankRng.end !== Infinity && rankRng.end <= 20) {
                    // cut too huge ranks
                    const rankMax = rankRng.end;
                    return ctx.genConstRankedShape(rankMax, source, partialDims).map((ctx) => {
                        const shape = ctx.retVal;
                        return ctx.setRetVal(ExpShape.slice(shape, 0, rank, source)) as Context<ExpShape>;
                    });
                } else {
                    const symShape = ctx.genSymShape('temp', rank, source);
                    const expShape = ExpShape.fromSymbol(symShape);
                    let ctxSet: ContextSet<any> = ctx.toSet();
                    let retShape: ExpShape = ExpShape.fromSymbol(symShape);

                    partialDims?.forEach((dim, idx) => {
                        const idxExp =
                            idx >= 0 ? ExpNum.fromConst(idx, source) : ExpNum.bop(NumBopType.Sub, rank, idx, source);
                        retShape = ExpShape.setDim(retShape, idxExp, dim, source);
                        ctxSet = ctxSet.guarantee(ctx.genEq(ExpNum.index(expShape, idxExp, source), dim, source));
                    });

                    return ctxSet.return(ExpShape.fromSymbol(symShape));
                }
            });
    }

    // constraint generator
    genBool(pred: ExpBool | boolean, source: CodeSource | undefined): Constraint {
        source = this._replaceBuiltinSource(source);
        return this.ctrSet.genEquality(
            ConstraintType.Equal,
            SymExp.fromConst(pred),
            SymExp.fromConst(true),
            source
        ) as CtrEq;
    }

    genEq(
        left: SymExp | number | string | boolean,
        right: SymExp | number | string | boolean,
        source: CodeSource | undefined
    ): CtrEq {
        source = this._replaceBuiltinSource(source);
        return this.ctrSet.genEquality(
            ConstraintType.Equal,
            SymExp.fromConst(left),
            SymExp.fromConst(right),
            source
        ) as CtrEq;
    }

    genNeq(
        left: SymExp | number | string | boolean,
        right: SymExp | number | string | boolean,
        source: CodeSource | undefined
    ): CtrNeq {
        source = this._replaceBuiltinSource(source);
        return this.ctrSet.genEquality(
            ConstraintType.NotEqual,
            SymExp.fromConst(left),
            SymExp.fromConst(right),
            source
        ) as CtrNeq;
    }

    genLt(left: ExpNum | number, right: ExpNum | number, source: CodeSource | undefined): CtrLt {
        source = this._replaceBuiltinSource(source);
        return this.ctrSet.genNumCompare(
            ConstraintType.LessThan,
            SymExp.fromConst(left) as ExpNum,
            SymExp.fromConst(right) as ExpNum,
            source
        ) as CtrLt;
    }

    genLte(left: ExpNum | number, right: ExpNum | number, source: CodeSource | undefined): CtrLte {
        source = this._replaceBuiltinSource(source);
        return this.ctrSet.genNumCompare(
            ConstraintType.LessThanOrEqual,
            SymExp.fromConst(left) as ExpNum,
            SymExp.fromConst(right) as ExpNum,
            source
        ) as CtrLte;
    }

    genAnd(left: Constraint, right: Constraint, source: CodeSource | undefined): CtrAnd {
        source = this._replaceBuiltinSource(source);
        return this.ctrSet.genAnd(left, right, source);
    }

    genOr(left: Constraint, right: Constraint, source: CodeSource | undefined): CtrOr {
        source = this._replaceBuiltinSource(source);
        return this.ctrSet.genOr(left, right, source);
    }

    genNot(constraint: Constraint, source: CodeSource | undefined): CtrNot {
        source = this._replaceBuiltinSource(source);
        return this.ctrSet.genNot(constraint, source);
    }

    genBroadcastable(left: ExpShape, right: ExpShape, source: CodeSource | undefined): CtrBroad {
        source = this._replaceBuiltinSource(source);
        return this.ctrSet.genBroad(left, right, source);
    }

    genForall(
        symbol: SymInt,
        range: [number | ExpNum, number | ExpNum],
        constraint: Constraint,
        source: CodeSource | undefined
    ): CtrForall {
        source = this._replaceBuiltinSource(source);
        return this.ctrSet.genForall(symbol, range, constraint, source);
    }

    genFail(reason: string, source: CodeSource | undefined): CtrFail {
        source = this._replaceBuiltinSource(source);
        return this.ctrSet.genFail(reason, source);
    }

    // make `sizable` to shape, if `sizable` may be iterable object of integer.
    // if parse error is not critical, return parse error messages
    parseSize(iterable: ShValue, source: CodeSource | undefined): ContextSet<ExpShape | string> {
        const sizeObj = fetchAddr(iterable, this.heap);
        if (sizeObj?.type !== SVType.Object) {
            return this.toSetWith('value is not iterable; cannot parse to size.');
        }

        if (sizeObj instanceof SVSize) {
            return this.toSetWith(sizeObj.shape);
        }

        const rankValue = fetchAddr(sizeObj.getAttr('$length'), this.heap);
        if (rankValue?.type !== SVType.Int) {
            return this.toSetWith('value is not iterable; cannot parse to size.');
        }

        return this.genRankedShape(rankValue.value, source, sizeObj.extractIndexedNumber(this.heap));
    }

    // shape operations
    shBroadcast(left: ExpShape, right: ExpShape, source: CodeSource | undefined): ContextSet<ExpShape> {
        return this.require([this.genBroadcastable(left, right, source)], 'shape is not broadcastable', source).return(
            simplifyShape(this.ctrSet, ExpShape.broadcast(left, right, source))
        );
    }

    shReduce(shape: ExpShape, axis: ExpNum | number, source: CodeSource | undefined): ContextSet<ExpShape> {
        const left = ExpShape.slice(shape, undefined, axis, source);
        const right = ExpShape.slice(shape, ExpNum.bop(NumBopType.Add, axis, 1, source), undefined, source);
        let newShape: ExpShape = ExpShape.concat(left, right, source);
        newShape = simplifyShape(this.ctrSet, newShape);
        return this.setRetVal(newShape).toSet();
    }

    shMatmul(left: ExpShape, right: ExpShape, source: CodeSource | undefined): ContextSet<ExpShape> {
        const ctx = this;
        const leftRank = ExpShape.getRank(left);
        const rightRank = ExpShape.getRank(right);
        const leftMDim = ExpNum.index(left, ExpNum.bop(NumBopType.Sub, leftRank, 1, source), source);
        const rightMDim = ExpNum.index(right, ExpNum.bop(NumBopType.Sub, rightRank, 2, source), source);

        return ctx
            .require(
                [ctx.genLte(2, leftRank, source), ctx.genLte(2, rightRank, source)],
                `from 'matmul': rank should be greater than 1`,
                source
            )
            .flatMap((ctx) =>
                ctx.require([ctx.genEq(leftMDim, rightMDim, source)], `from 'matmul': dimension mismatch`, source)
            )
            .flatMap((ctx) =>
                ctx.shBroadcast(
                    ExpShape.slice(left, 0, ExpNum.bop(NumBopType.Sub, leftRank, 2, source), source),
                    ExpShape.slice(right, 0, ExpNum.bop(NumBopType.Sub, rightRank, 2, source), source),
                    source
                )
            )
            .map((ctx) => {
                const broadShape = ctx.retVal;
                const leftLDim = ExpNum.index(left, ExpNum.bop(NumBopType.Sub, leftRank, 2, source), source);
                const rightRDim = ExpNum.index(right, ExpNum.bop(NumBopType.Sub, rightRank, 1, source), source);

                const matmulShape = ExpShape.fromConst(2, [leftLDim, rightRDim], source);
                return ctx.setRetVal(simplifyShape(this.ctrSet, ExpShape.concat(broadShape, matmulShape, source)));
            });
    }

    shRepeat(
        shape: ExpShape,
        axis: ExpNum | number,
        count: ExpNum | number,
        source: CodeSource | undefined
    ): ContextSet<ExpShape> {
        const rank = ExpShape.getRank(shape);
        const axisCtr = this.genAnd(this.genLte(0, axis, source), this.genLte(axis, rank, source), source);
        const countCtr = this.genLte(0, count, source);

        return this.require([axisCtr, countCtr], 'shRepeat constraint failed', source).map((ctx) => {
            const leftShape = ExpShape.slice(shape, undefined, axis, source);
            const rightShape = ExpShape.slice(shape, axis, undefined, source);
            const repeated = ExpShape.concat(
                ExpShape.concat(leftShape, ExpShape.fromConst(1, [count], source), source),
                rightShape,
                source
            );
            return ctx.setRetVal(simplifyShape(this.ctrSet, repeated));
        });
    }

    require(ctr: Constraint | Constraint[], failMsg: string, source: CodeSource | undefined): ContextSet<T | SVError> {
        source = this._replaceBuiltinSource(source);
        if (Array.isArray(ctr)) {
            return this.toSet().require(ctr, failMsg, source);
        } else {
            return this.toSet().require(ctr, failMsg, source);
        }
    }

    guarantee(ctr: Constraint | Constraint[]): Context<T> {
        if (Array.isArray(ctr)) {
            return this.setCtrSet(this.ctrSet.guaranteeAll(ctr));
        } else {
            return this.setCtrSet(this.ctrSet.guarantee(ctr));
        }
    }

    ifThenElse(ctr: Constraint, source: CodeSource | undefined): [ContextSet<T>, ContextSet<T>] {
        return this.toSet().ifThenElse(ctr, source);
    }

    // immediate linear SMT : generates conservative range.
    getCachedRange(num: number | ExpNum): NumRange | undefined {
        return this.ctrSet.getCachedRange(num);
    }

    checkImmediate(constraint: Constraint): boolean | undefined {
        return this.ctrSet.checkImmediate(constraint);
    }

    hasPathCtr(): boolean {
        return !this.ctrSet.pathCtr.isEmpty();
    }

    isTimedOut(): boolean {
        const reason = this.failed;
        if (reason && reason.reason.startsWith('timeout expired')) {
            return true;
        }
        return false;
    }

    // replace builtin.py source by call stack
    private _replaceBuiltinSource(source: CodeSource | undefined): CodeSource | undefined {
        if (source && !('fileId' in source)) {
            let moduleNode = source;
            while (moduleNode.nodeType !== ParseNodeType.Module) {
                moduleNode = moduleNode.parent!;
            }

            const fileInfo = getFileInfo(moduleNode)!;
            if (checkIgnorePath(fileInfo.filePath)) {
                for (let i = this.callStack.count() - 1; i >= 0; i--) {
                    const node = this.callStack.get(i)![1];
                    if (node && !('fileId' in node)) {
                        moduleNode = node;
                        while (moduleNode.nodeType !== ParseNodeType.Module) {
                            moduleNode = moduleNode.parent!;
                        }
                        const fileInfo = getFileInfo(moduleNode)!;
                        if (!checkIgnorePath(fileInfo.filePath)) {
                            return node;
                        }
                    }
                }
            }
        }

        return source;
    }
}

// mimic interval
export namespace ContextSet {
    export type CtxConstructorCallback = (ctxSet: ContextSetImpl<unknown>) => void;

    // callback, start time, call count, interval
    let _callbacks: CtxConstructorCallback[] = [];
    let _maxPaths = 0;

    export function clearCallbacks(): void {
        _callbacks = [];
    }

    export function setCallback(callback: (ctxSet: ContextSetImpl<unknown>) => void): void {
        _callbacks.push(callback);
    }

    export function checkCallbacks(ctxSet: ContextSetImpl<unknown>) {
        for (const callback of _callbacks) {
            callback(ctxSet);
        }
    }

    export function getMaxPaths(): number {
        return _maxPaths;
    }

    export function setMaxPaths(count: number) {
        _maxPaths = count;
    }
}

export class ContextSetImpl<T> implements ContextSet<T> {
    env: false;
    // running path
    ctxList: List<Context<T>>;
    // failed path
    failed: List<Context<SVError>>;
    // stopped path: failed but possibly unreachable path
    stopped: List<Context<SVError>>;

    constructor(ctxList: List<Context<T>>, failed?: List<Context<SVError>>, stopped?: List<Context<SVError>>) {
        this.env = false;
        this.ctxList = ctxList;
        this.failed = failed ? failed : List();
        this.stopped = stopped ? stopped : List();

        ContextSet.checkCallbacks(this);
    }

    static fromCtx<A>(ctx: Context<A>): ContextSet<A> {
        if (ctx.failed) {
            let failedCtx = ctx.setRetVal(ctx.failed);
            if (failedCtx.failId === -1) {
                failedCtx = failedCtx.set('failId', getFailedId());
            }
            if (failedCtx.hasPathCtr()) {
                return new ContextSetImpl(List(), List(), List([failedCtx]));
            } else {
                return new ContextSetImpl(List(), List([failedCtx]), List());
            }
        } else {
            return new ContextSetImpl(List([ctx]), List(), List());
        }
    }

    getList(): List<Context<T>> {
        return this.ctxList;
    }

    getFailed(): List<Context<SVError>> {
        return this.failed;
    }

    getStopped(): List<Context<SVError>> {
        return this.stopped;
    }

    getRunningCount(): number {
        return this.ctxList.count();
    }

    filter(tester: (ctx: Context<T>) => boolean): ContextSet<T> {
        return new ContextSetImpl(this.ctxList.filter(tester), this.failed, this.stopped);
    }

    setCtxList<A>(ctxList: List<Context<A>>): ContextSet<A> {
        const succeed = ctxList.filter((ctx) => ctx.failed === undefined);
        const failed = ctxList
            .filter((ctx) => ctx.failed !== undefined)
            .filter((ctx) => !ctx.hasPathCtr())
            .map((ctx) => {
                let failed = ctx.setRetVal(ctx.failed!);
                if (ctx.failId === -1) failed = failed.set('failId', getFailedId());
                return failed;
            });
        const stopped = ctxList
            .filter((ctx) => ctx.failed !== undefined)
            .filter((ctx) => ctx.hasPathCtr())
            .map((ctx) => {
                let stopped = ctx.setRetVal(ctx.failed!);
                if (ctx.failId === -1) stopped = stopped.set('failId', getFailedId());
                return stopped;
            });
        return new ContextSetImpl(succeed, this.failed.concat(failed), this.stopped.concat(stopped));
    }

    map<A>(mapper: (ctx: Context<T>) => Context<A>): ContextSet<A> {
        return this.setCtxList(this.ctxList.map(mapper));
    }

    flatMap<A>(mapper: (ctx: Context<T>) => ContextSet<A>): ContextSet<A> {
        const mapped = this.ctxList.map(mapper);
        const succeed = mapped.toArray().flatMap((cs) => cs.getList().toArray());
        const failed = mapped.toArray().flatMap((cs) => cs.getFailed().toArray());
        const stopped = mapped.toArray().flatMap((cs) => cs.getStopped().toArray());

        return new ContextSetImpl(List(succeed), List(failed).concat(this.failed), List(stopped).concat(this.stopped));
    }

    return<A>(retVal: A): ContextSet<A> {
        return new ContextSetImpl(
            this.ctxList.map((ctx) => ctx.setRetVal(retVal)),
            this.failed,
            this.stopped
        );
    }

    fail(errMsg: string, source: CodeSource | undefined): ContextSet<SVError> {
        return new ContextSetImpl(
            List(),
            this.ctxList
                .filter((ctx) => {
                    !ctx.hasPathCtr();
                })
                .map((ctx) => {
                    return ctx.failWithMsg(errMsg, source);
                })
                .concat(this.failed),
            this.ctxList
                .filter((ctx) => {
                    ctx.hasPathCtr();
                })
                .map((ctx) => {
                    return ctx.failWithMsg(errMsg, source);
                })
                .concat(this.stopped)
        );
    }

    join<A>(ctxSet: ContextSet<A>): ContextSet<A | T> {
        const thisList: List<Context<T | A>> = this.ctxList;
        const thatList: List<Context<T | A>> = ctxSet.getList();

        // TODO: precalculate failed set.
        const failedSet: { [id: number]: Context<SVError> } = {};
        const thatFailed = ctxSet.getFailed();
        this.failed.forEach((ctx) => (failedSet[ctx.failId] = ctx));
        thatFailed.forEach((ctx) => (failedSet[ctx.failId] = ctx));

        const stoppedSet: { [id: number]: Context<SVError> } = {};
        const thatStopped = ctxSet.getStopped();
        this.stopped.forEach((ctx) => (stoppedSet[ctx.failId] = ctx));
        thatStopped.forEach((ctx) => (stoppedSet[ctx.failId] = ctx));

        return new ContextSetImpl(
            thisList.concat(thatList),
            List(Object.values(failedSet)),
            List(Object.values(stoppedSet))
        );
    }

    require(ctr: Constraint | Constraint[], failMsg: string, source: CodeSource | undefined): ContextSet<T | SVError> {
        // immediate constraint check
        const ctrList: Constraint[] = Array.isArray(ctr)
            ? ctr.map((c) => ({ ...c, message: failMsg }))
            : [{ ...ctr, message: failMsg }];

        return this.map((ctx) => {
            const ctrSet = ctx.ctrSet.requireAll(ctrList);
            if (ctrSet.valid === false) {
                const ctrMsg = ctrList.map((ctr) => `${ctrToStr(ctr)}`).join(' /\\ \n');
                const errMsg = `${failMsg ? failMsg : 'runtime constraint mismatch'}\n  CONSTRAINTS:\n${ctrMsg}\n`;
                return ctx.setCtrSet(ctrSet).failWithMsg(errMsg, source) as Context<T | SVError>;
            } else {
                return ctx.setCtrSet(ctrSet);
            }
        });
    }

    guarantee(ctr: Constraint | Constraint[]): ContextSet<T> {
        // No immediate constraint check
        return this.map((ctx) => {
            if (Array.isArray(ctr)) {
                return ctx.setCtrSet(ctx.ctrSet.guaranteeAll(ctr));
            } else {
                return ctx.setCtrSet(ctx.ctrSet.guarantee(ctr));
            }
        });
    }

    ifThenElse(ctr: Constraint, source: CodeSource | undefined): [ContextSet<T>, ContextSet<T>] {
        const ifPath: Context<T>[] = [];
        const elsePath: Context<T>[] = [];

        this.ctxList.forEach((ctx) => {
            const ctrSet = ctx.ctrSet.addIf({ ...ctr, message: 'true path' });
            if (ctrSet.valid !== false) {
                ifPath.push(ctx.setCtrSet(ctrSet));
            }

            const elseSet = ctx.ctrSet.addIf({ ...ctx.ctrSet.genNot(ctr, source), message: 'false path' });
            if (elseSet.valid !== false) {
                elsePath.push(ctx.setCtrSet(elseSet));
            }
        });

        return [
            new ContextSetImpl(List(ifPath), this.failed, this.stopped),
            new ContextSetImpl(List(elsePath), this.failed, this.stopped),
        ];
    }

    isEmpty(): boolean {
        return this.ctxList.isEmpty();
    }

    addLog(message: string, source: CodeSource | undefined): ContextSet<T> {
        return this.map((ctx) => ctx.addLog(message, source));
    }
    addLogValue(value: ShValue): ContextSet<T> {
        return this.map((ctx) => ctx.addLogValue(value));
    }

    // check called function is (behaviourly) pure function with some path divergences.
    // this means
    // 1. path is diverged (exactly 2 paths)
    // 2. only hard and path conditions are added from this function
    //    and also those conditions use new symbols only (so that those can be garbage-collected)
    // 3. divergence is not checked. (so that this checking is not propagated to higher stacks)
    //    this check is performed by some flags in constraint set
    // 4. return value is deeply equal for every paths (except source).
    // 5. return value does not refer any new symbols
    //    (i.e. we can safely remove those path conditions)
    // 6. it does not change any value of the original heap (i.e. uses only its own stack frame)
    prunePureFunctionCall(oldCtx: Context<unknown>, symIdMax: number): ContextSet<T> {
        if (this.ctxList.count() !== 2) return this;

        const oldHeap = oldCtx.heap;
        const heapLimit = oldHeap.addrMax;

        const left = this.ctxList.get(0)!;
        const right = this.ctxList.get(1)!;

        const leftSet = left.ctrSet;
        const rightSet = right.ctrSet;
        const leftLen = leftSet.ctrPool.count();
        const rightLen = rightSet.ctrPool.count();

        // any soft constraint is appended
        const oldSoftLen = oldCtx.ctrSet.softCtr.count();
        if (leftSet.softCtr.count() !== oldSoftLen || rightSet.softCtr.count() !== oldSoftLen) {
            return this;
        }

        if (leftSet.notPrunedCtrMax >= leftLen) {
            const newList = this.ctxList.set(1, right.setCtrSet(rightSet.set('notPrunedCtrMax', rightLen)));
            return new ContextSetImpl(newList, this.getFailed(), this.getStopped());
        } else if (rightSet.notPrunedCtrMax >= rightLen) {
            const newList = this.ctxList.set(0, left.setCtrSet(leftSet.set('notPrunedCtrMax', leftLen)));
            return new ContextSetImpl(newList, this.getFailed(), this.getStopped());
        }

        const newThis: ContextSet<T> = new ContextSetImpl(
            this.ctxList
                .set(0, left.setCtrSet(leftSet.set('notPrunedCtrMax', leftLen)))
                .set(1, right.setCtrSet(rightSet.set('notPrunedCtrMax', rightLen))),
            this.failed,
            this.stopped
        );

        const oldLen = oldCtx.ctrSet.ctrPool.count();
        for (let i = oldLen; i < leftLen; i++) {
            const leftSymbols = Set(extractSymbols(leftSet.ctrPool.get(i)!));
            if (leftSymbols.find((v) => v <= symIdMax) ?? -1 >= 0) return newThis;
        }
        for (let i = oldLen; i < rightLen; i++) {
            const rightSymbols = Set(extractSymbols(rightSet.ctrPool.get(i)!));
            if (rightSymbols.find((v) => v <= symIdMax) ?? -1 >= 0) return newThis;
        }

        // return value equality and non-conditionality
        let leftRet = (left.retVal as any) as ShValue | ShContFlag;
        let rightRet = (right.retVal as any) as ShValue | ShContFlag;
        const leftHeap = left.heap;
        const rightHeap = right.heap;

        // object address equality map
        let eqMap: Map<number, number> = Map<number, number>();

        if (typeof leftRet !== 'object') leftRet = SVNone.create(undefined);
        if (typeof rightRet !== 'object') rightRet = SVNone.create(undefined);

        function eqCheck(lv?: ShValue, rv?: ShValue): boolean {
            if (lv?.type === SVType.Addr) {
                lv = fetchAddr(lv, leftHeap);
            }
            if (rv?.type === SVType.Addr) {
                rv = fetchAddr(rv, rightHeap);
            }

            if (!lv && !rv) return true;
            if (!(lv && rv)) return false;
            if (lv.type !== rv.type) return false;

            switch (lv.type) {
                case SVType.Int:
                case SVType.Float: {
                    const lval = lv.value;
                    const rval = (rv as SVInt | SVFloat).value;

                    if (typeof lval === 'number') {
                        if (typeof rval === 'number') {
                            return lval === rval;
                        }
                        return false;
                    }
                    if (typeof rval === 'number') {
                        return false;
                    }

                    const lsval = simplifyExp(leftSet, lval);
                    const rsval = simplifyExp(rightSet, rval);

                    if (SymExp.extractSymbols(lsval).findIndex((v) => v > symIdMax) >= 0) return false;
                    if (SymExp.extractSymbols(rsval).findIndex((v) => v > symIdMax) >= 0) return false;

                    return isStructuallyEq(lsval, rsval);
                }
                case SVType.String: {
                    const lval = lv.value;
                    const rval = (rv as SVString).value;

                    if (typeof lval === 'string') {
                        if (typeof rval === 'string') {
                            return lval === rval;
                        }
                        return false;
                    }
                    if (typeof rval === 'string') {
                        return false;
                    }

                    const lsval = simplifyExp(leftSet, lval);
                    const rsval = simplifyExp(rightSet, rval);

                    if (SymExp.extractSymbols(lsval).findIndex((v) => v > symIdMax) >= 0) return false;
                    if (SymExp.extractSymbols(rsval).findIndex((v) => v > symIdMax) >= 0) return false;

                    return isStructuallyEq(lsval, rsval);
                }
                case SVType.Bool: {
                    const lval = lv.value;
                    const rval = (rv as SVBool).value;

                    if (typeof lval === 'boolean') {
                        if (typeof rval === 'boolean') {
                            return lval === rval;
                        }
                        return false;
                    }
                    if (typeof rval === 'boolean') {
                        return false;
                    }

                    const lsval = simplifyExp(leftSet, lval);
                    const rsval = simplifyExp(rightSet, rval);

                    if (SymExp.extractSymbols(lsval).findIndex((v) => v > symIdMax) >= 0) return false;
                    if (SymExp.extractSymbols(rsval).findIndex((v) => v > symIdMax) >= 0) return false;

                    return isStructuallyEq(lsval, rsval);
                }
                case SVType.Object: {
                    const rval = rv as SVObject;
                    const laddr = lv.addr.addr;
                    const raddr = rval.addr.addr;

                    // check address equality below heaplimit
                    if (laddr <= heapLimit) {
                        if (raddr <= heapLimit) {
                            return laddr === raddr;
                        }
                        return false;
                    } else if (raddr <= heapLimit) {
                        return false;
                    }

                    if (eqMap.has(laddr)) {
                        if (eqMap.get(laddr) !== raddr) return false;
                        return true;
                    }

                    eqMap = eqMap.set(laddr, raddr);

                    // check recursive value equality above heaplimit
                    if (lv.shape) {
                        if (!rval.shape) return false;
                        if (!isStructuallyEq(simplifyExp(leftSet, lv.shape), simplifyExp(rightSet, rval.shape)))
                            return false;
                    }

                    if (lv.keyValues.count() !== rval.keyValues.count()) return false;
                    if (lv.indices.count() !== rval.indices.count()) return false;
                    if (lv.attrs.count() !== rval.attrs.count()) return false;

                    if (!lv.keyValues.every((lvv, lvk) => eqCheck(lvv, rval.keyValues.get(lvk)))) return false;
                    if (!lv.indices.every((lvv, lvk) => eqCheck(lvv, rval.indices.get(lvk)))) return false;
                    if (!lv.attrs.every((lvv, lvk) => eqCheck(lvv, rval.attrs.get(lvk)))) return false;

                    return true;
                }
                case SVType.Func:
                    // TODO: if closure points symbol? => check no closure exists
                    //       but list comprehension / lambda / ternary is closure!
                    // CHECK: function equality is too hard to check. alter it to referential equality
                    return lv === rv;
                case SVType.None:
                case SVType.NotImpl:
                case SVType.Undef:
                    return true;
                case SVType.Error:
                    // this should be equally propagated values.
                    return lv === rv;
                default:
                    return false;
            }
        }

        if (!eqCheck(leftRet, rightRet)) return newThis;

        // purity check (heap unchanged)
        const pureLeftMap = leftHeap.valMap.filter((_, addr) => addr <= heapLimit);
        const pureRightMap = rightHeap.valMap.filter((_, addr) => addr <= heapLimit);
        const pureMap = oldHeap.valMap.filter((_, addr) => addr <= heapLimit);

        if (!(pureLeftMap.equals(pureMap) && pureRightMap.equals(pureMap))) {
            return newThis;
        }

        // all checked. prune right and remove conditions
        return left.setCtrSet(oldCtx.ctrSet).toSet();
    }
}
