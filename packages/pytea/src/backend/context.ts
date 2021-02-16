/*
 * context.ts
 * Copyright (c) Seoul National University.
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo (mersshs@gmail.com)
 *
 * Context for backend processing.
 * Collection of Environment, Heap, and Constraint set.
 */
import { List, Map, Record } from 'immutable';

import { ParseNode } from 'pyright-internal/parser/parseNodes';

import { formatParseNode } from '../service/pyteaUtils';
import { fetchAddr } from './backUtils';
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
} from './constraintType';
import { genTensor } from './expUtils';
import { NumRange } from './range';
import { ShEnv, ShHeap } from './sharpEnvironments';
import { ShContFlag, ShValue, SVError, SVFunc, SVType } from './sharpValues';
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

interface ContextProps<T> {
    failId: number;
    env: ShEnv;
    heap: ShHeap;
    ctrSet: ConstraintSet;
    retVal: T;

    // SVFunc is python function call, string is libcall name. ParseNode is source.
    callStack: List<[SVFunc | string, ParseNode | undefined]>;
    logs: List<ShValue>;
    imported: ShEnv; // qualPath (relative to project root or site-packages) to address.
    relPath: string; // relative path to entry file. starts with entry file name.

    // if set, automatically go to failed path.
    failed?: SVError;
}

interface ContextMethods<T> {
    // log exports
    callStackToString(): string;
    logsToString(): string;

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

    getAttrDeep(value: ShValue, attr: string, source?: ParseNode): ContextSet<ShValue>;
    getIndiceDeep(value: ShValue, index: number, source?: ParseNode): ContextSet<ShValue>;
    getKeyValDeep(value: ShValue, key: string, source?: ParseNode): ContextSet<ShValue>;

    addLog(message: string, source?: ParseNode): Context<T>;
    addLogValue(log: ShValue): Context<T>;
    pushCallStack(stack: [SVFunc | string, ParseNode | undefined]): Context<T>;
    popCallStack(): Context<T>;

    // these two methods does not cut off paths. just log warnings
    warn(warning: SVError): Context<SVError>;
    warnWithMsg(message: string, source?: ParseNode): Context<SVError>;

    // these two methods cut off paths
    fail(error: SVError): Context<SVError>;
    failWithMsg(message: string, source?: ParseNode): Context<SVError>;

    // symbolic variable generator
    genSymInt(name: string, source?: ParseNode): SymInt;
    genSymFloat(name: string, source?: ParseNode): SymFloat;
    genSymBool(name: string, source?: ParseNode): SymBool;
    genSymString(name: string, source?: ParseNode): SymString;
    genSymShape(name: string, rank: ExpNum, source?: ParseNode): SymShape;

    // return ExpNumSymbol that is greater than `value`;
    genIntGte(name: string, value: number | ExpNum, source?: ParseNode): Context<ExpNumSymbol>;
    genFloatGte(name: string, value: number | ExpNum, source?: ParseNode): Context<ExpNumSymbol>;

    // generate constant-ranked shape. all the dimensions is new symbolic number gte 0. should check rank >= 0 before call it.
    // partialDims is additional constant part of dimensions
    genConstRankedShape(rank: number, source?: ParseNode, partialDims?: Map<number, ExpNum>): ContextSet<ExpShapeConst>;

    // generate ranked shape. all the dimensions is new symbolic number gte 0.
    // partialDims is additional constant part of dimensions
    // if rank has upper-bound, return slice of upper-bound-ranked shape. (for optimization)
    // in production, ranked has force-upper bound of some number (e.g. 6)
    genRankedShape(rank: number | ExpNum, source?: ParseNode): ContextSet<ExpShape>;

    // constraint generator
    genBool(pred: ExpBool | boolean, source?: ParseNode): Constraint;
    genEq(
        left: SymExp | number | string | boolean,
        right: SymExp | number | string | boolean,
        source?: ParseNode
    ): CtrEq;
    genNeq(left: SymExp | number | string, right: SymExp | number | string, source?: ParseNode): CtrNeq;
    genLt(left: ExpNum | number, right: ExpNum | number, source?: ParseNode): CtrLt;
    genLte(left: ExpNum | number, right: ExpNum | number, source?: ParseNode): CtrLte;
    genAnd(left: Constraint, right: Constraint, source?: ParseNode): CtrAnd;
    genOr(left: Constraint, right: Constraint, source?: ParseNode): CtrOr;
    genNot(constraint: Constraint, source?: ParseNode): CtrNot;
    genForall(
        symbol: SymInt,
        range: [number | ExpNum, number | ExpNum],
        constraint: Constraint,
        source?: ParseNode
    ): CtrForall;
    genBroadcastable(left: ExpShape, right: ExpShape, source?: ParseNode): CtrBroad;
    genFail(reason: string, source?: ParseNode): CtrFail;

    // shape operations

    // make `sizable` to shape, if `sizable` may be iterable object of integer.
    // if parse error is not critical, return parse error messages
    parseSize(iterable: ShValue, source?: ParseNode): ContextSet<ExpShape | string>;

    // matmul between shapes which rank is greater than 1
    // if rank is greater than 2, it follows the matrix multiplication broadcasting rule of numpy.
    shMatmul(left: ExpShape, right: ExpShape, source?: ParseNode): ContextSet<ExpShape>;

    // return broadcasted shape
    shBroadcast(left: ExpShape, right: ExpShape, source?: ParseNode): ContextSet<ExpShape>;

    // return shape without `axis`-th. `axis` is 0-based
    shReduce(shape: ExpShape, axis: ExpNum | number, source?: ParseNode): ContextSet<ExpShape>;

    // slice shape along `axis`. ranges is 0-based and exclusive
    shSubtensor(
        shape: ExpShape,
        ranges: [(ExpNum | number)?, (ExpNum | number)?],
        axis: ExpNum | number,
        source?: ParseNode
    ): ContextSet<ExpShape>;

    // reshape `shape` into `dims`
    shReshape(shape: ExpShape, dims: (ExpNum | number)[], source?: ParseNode): ContextSet<ExpShape>;

    // permute dimensions of `shape` by `axes`. `axes` is 0-based
    shPermute(shape: ExpShape, axes: (ExpNum | number)[], source?: ParseNode): ContextSet<ExpShape>;

    // concat `shapes` along `axis`. `axis` is 0-based
    shConcat(shapes: ExpShape[], axis: ExpNum | number, source?: ParseNode): ContextSet<ExpShape>;

    // make new dimension and stack `shapes` along `axis`. `axis` is 0-based
    shStack(shapes: ExpShape[], axis: ExpNum | number, source?: ParseNode): ContextSet<ExpShape>;

    // repeat `shape` by `count` times and stack along `axis`. `axis` is 0-based
    shRepeat(shape: ExpShape, axis: ExpNum | number, count: ExpNum | number, source?: ParseNode): ContextSet<ExpShape>;

    // constraint injector
    // failed immediately if `ctr` is false. (soft-constraint)
    require(ctr: Constraint | Constraint[], failMsg?: string, source?: ParseNode): ContextSet<T | SVError>;

    // add admitted constraints (hard-constraint)
    guarantee(ctr: Constraint | Constraint[]): Context<T>;

    // return both if-path and else-path.
    ifThenElse(ctr: Constraint, source?: ParseNode): [ContextSet<T>, ContextSet<T>];

    // immediate linear SMT : generates conservative range.
    getCachedRange(num: number | ExpNum): NumRange | undefined; // return conservative range
    checkImmediate(constraint: Constraint): boolean | undefined; // return true if always true, false if always false, undefined if don't know

    // Check if this ctx has path constraints.
    hasPathCtr(): boolean;
}

export interface ContextSet<T> {
    // to distinguish context and contextset
    env: false;
    setCtxList<A>(ctxList: List<Context<A>>): ContextSet<A>;

    filter(tester: (ctx: Context<T>) => boolean): ContextSet<T>;
    map<A>(mapper: (ctx: Context<T>) => Context<A>): ContextSet<A>;
    flatMap<A>(mapper: (ctx: Context<T>) => ContextSet<A>): ContextSet<A>;

    return<A>(retVal: A): ContextSet<A>;
    fail(errMsg: string, source?: ParseNode): ContextSet<SVError>;
    join<A>(ctxSet: ContextSet<A>): ContextSet<A | T>;

    require(ctr: Constraint | Constraint[], failMsg?: string, source?: ParseNode): ContextSet<T | SVError>;
    guarantee(ctr: Constraint | Constraint[]): ContextSet<T>;
    ifThenElse(ctr: Constraint, source?: ParseNode): [ContextSet<T>, ContextSet<T>];

    getList(): List<Context<T>>;
    getFailed(): List<Context<SVError>>;
    getStopped(): List<Context<SVError>>;

    isEmpty(): boolean;
    addLog(message: string, source?: ParseNode): ContextSet<T>;
    addLogValue(value: ShValue): ContextSet<T>;
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

    logsToString(): string {
        return this.logs
            .map((log) => {
                const posStr = formatParseNode(log.source);

                if (log.type === SVType.Error) {
                    return `${log.reason} - ${posStr}`;
                } else {
                    return `${log.toString()} - ${posStr}`;
                }
            })
            .join('\n');
    }

    callStackToString(): string {
        return this.callStack
            .filter(([f, _]) => {
                // filter callKV libcall
                if (typeof f === 'string') {
                    return f !== 'callKV';
                } else {
                    return f.name !== 'callKV';
                }
            })
            .map(([func, node]) => `${typeof func === 'string' ? func : func.name} - ${formatParseNode(node)}`)
            .reverse()
            .join('\n');
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

    getAttrDeep(value: ShValue, attr: string, source?: ParseNode): ContextSet<ShValue> {
        return TorchBackend.getAttrDeep(this, value, attr, source);
    }

    getIndiceDeep(value: ShValue, index: number | ExpNum, source?: ParseNode): ContextSet<ShValue> {
        return TorchBackend.getIndiceDeep(this, value, index, source);
    }

    getKeyValDeep(value: ShValue, key: string, source?: ParseNode): ContextSet<ShValue> {
        return TorchBackend.getKeyValDeep(this, value, key, source);
    }

    // these two methods does not cut off paths. just log warnings
    warn(warning: SVError): Context<SVError> {
        return this.setRetVal(warning).addLogValue(warning);
    }
    warnWithMsg(message: string, source?: ParseNode): Context<SVError> {
        const warning = SVError.create(`WARNING: ${message}`, source);
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
    warnTensorWithMsg(message: string, source?: ParseNode): ContextSet<ShValue> {
        const warning = SVError.create(`WARNING: ${message}`, source);
        return this.warnTensor(warning);
    }

    // these two methods cut off paths
    fail(error: SVError): Context<SVError> {
        return this.set('failed', error).set('failId', getFailedId()).addLogValue(error).setRetVal(error);
    }
    failWithMsg(message: string, source?: ParseNode): Context<SVError> {
        return this.fail(SVError.create(`ERROR: ${message},`, source));
    }

    addLog(message: string, source?: ParseNode): Context<T> {
        return this.set('logs', this.logs.push(SVError.create(`LOG: ${message}`, source)));
    }

    addLogValue(log: ShValue): Context<T> {
        return this.set('logs', this.logs.push(log));
    }

    pushCallStack(stack: [SVFunc | string, ParseNode | undefined]): Context<T> {
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

    // symbolic variable generator
    genSymInt(name: string, source?: ParseNode): SymInt {
        return this.ctrSet.genSymInt(name, source);
    }
    genSymFloat(name: string, source?: ParseNode): SymFloat {
        return this.ctrSet.genSymFloat(name, source);
    }
    genSymBool(name: string, source?: ParseNode): SymBool {
        return this.ctrSet.genSymBool(name, source);
    }
    genSymString(name: string, source?: ParseNode): SymString {
        return this.ctrSet.genSymString(name, source);
    }
    genSymShape(name: string, rank: ExpNum, source?: ParseNode): SymShape {
        return this.ctrSet.genSymShape(name, rank, source);
    }

    // return ExpNumSymbol that is greater than `value`;
    genIntGte(name: string, value: number | ExpNum, source?: ParseNode): Context<ExpNumSymbol> {
        const num = this.genSymInt(name, source);
        const exp = ExpNum.fromSymbol(num);
        const pos = this.genLte(value, exp, source);
        return this.guarantee(pos).setRetVal(exp);
    }

    genFloatGte(name: string, value: number | ExpNum, source?: ParseNode): Context<ExpNumSymbol> {
        const num = this.genSymFloat(name, source);
        const exp = ExpNum.fromSymbol(num);
        const pos = this.genLte(value, exp, source);
        return this.guarantee(pos).setRetVal(exp);
    }

    // generate constant-ranked shape. all the dimensions is new symbolic number gte.
    // partialDims is additional constant part of dimensions
    genConstRankedShape(
        rank: number,
        source?: ParseNode,
        partialDims?: Map<number, ExpNum>
    ): ContextSet<ExpShapeConst> {
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
    genRankedShape(rank: number | ExpNum, source?: ParseNode, partialDims?: Map<number, ExpNum>): ContextSet<ExpShape> {
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
            .require(this.genLte(0, rank), `genRankedShape got negative rank ${rank}`, source)
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
                        const idxExp = idx >= 0 ? ExpNum.fromConst(idx) : ExpNum.bop(NumBopType.Sub, rank, idx);
                        retShape = ExpShape.setDim(retShape, idxExp, dim);
                        ctxSet = ctxSet.guarantee(ctx.genEq(ExpNum.index(expShape, idxExp, source), dim, source));
                    });

                    return ctxSet.return(ExpShape.fromSymbol(symShape));
                }
            });
    }

    // constraint generator
    genBool(pred: ExpBool | boolean, source?: ParseNode): Constraint {
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
        source?: ParseNode
    ): CtrEq {
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
        source?: ParseNode
    ): CtrNeq {
        return this.ctrSet.genEquality(
            ConstraintType.NotEqual,
            SymExp.fromConst(left),
            SymExp.fromConst(right),
            source
        ) as CtrNeq;
    }

    genLt(left: ExpNum | number, right: ExpNum | number, source?: ParseNode): CtrLt {
        return this.ctrSet.genNumCompare(
            ConstraintType.LessThan,
            SymExp.fromConst(left) as ExpNum,
            SymExp.fromConst(right) as ExpNum,
            source
        ) as CtrLt;
    }

    genLte(left: ExpNum | number, right: ExpNum | number, source?: ParseNode): CtrLte {
        return this.ctrSet.genNumCompare(
            ConstraintType.LessThanOrEqual,
            SymExp.fromConst(left) as ExpNum,
            SymExp.fromConst(right) as ExpNum,
            source
        ) as CtrLte;
    }

    genAnd(left: Constraint, right: Constraint, source?: ParseNode): CtrAnd {
        return this.ctrSet.genAnd(left, right, source);
    }

    genOr(left: Constraint, right: Constraint, source?: ParseNode): CtrOr {
        return this.ctrSet.genOr(left, right, source);
    }

    genNot(constraint: Constraint, source?: ParseNode): CtrNot {
        return this.ctrSet.genNot(constraint, source);
    }

    genBroadcastable(left: ExpShape, right: ExpShape, source?: ParseNode): CtrBroad {
        return this.ctrSet.genBroad(left, right, source);
    }

    genForall(
        symbol: SymInt,
        range: [number | ExpNum, number | ExpNum],
        constraint: Constraint,
        source?: ParseNode
    ): CtrForall {
        return this.ctrSet.genForall(symbol, range, constraint, source);
    }

    genFail(reason: string, source?: ParseNode): CtrFail {
        return this.ctrSet.genFail(reason, source);
    }

    // make `sizable` to shape, if `sizable` may be iterable object of integer.
    // if parse error is not critical, return parse error messages
    parseSize(iterable: ShValue, source?: ParseNode): ContextSet<ExpShape | string> {
        const sizeObj = fetchAddr(iterable, this.heap);
        if (sizeObj?.type !== SVType.Object) {
            return this.toSetWith('value is not iterable; cannot parse to size.');
        }

        const rankValue = fetchAddr(sizeObj.getAttr('$length'), this.heap);
        if (rankValue?.type !== SVType.Int) {
            return this.toSetWith('value is not iterable; cannot parse to size.');
        }

        return this.genRankedShape(rankValue.value, source, sizeObj.extractIndexedNumber(this.heap));
    }

    // shape operations
    shBroadcast(left: ExpShape, right: ExpShape, source?: ParseNode): ContextSet<ExpShape> {
        return this.require([this.genBroadcastable(left, right, source)], 'shape is not broadcastable', source).return(
            ExpShape.broadcast(left, right, source)
        );
    }

    shReduce(shape: ExpShape, axis: ExpNum | number, source?: ParseNode): ContextSet<ExpShape> {
        // TODO: implement this.
        return this.setRetVal(shape).toSet();
    }

    shMatmul(left: ExpShape, right: ExpShape, source?: ParseNode): ContextSet<ExpShape> {
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
                return ctx.setRetVal(ExpShape.concat(broadShape, matmulShape, source));
            });
    }

    shSubtensor(
        shape: ExpShape,
        ranges: [(ExpNum | number)?, (ExpNum | number)?],
        axis: ExpNum | number,
        source?: ParseNode
    ): ContextSet<ExpShape> {
        // TODO: implement this.
        return this.setRetVal(shape).toSet();
    }

    shReshape(shape: ExpShape, dims: (ExpNum | number)[], source?: ParseNode): ContextSet<ExpShape> {
        // TODO: implement this.
        return this.setRetVal(shape).toSet();
    }

    shPermute(shape: ExpShape, axes: (ExpNum | number)[], source?: ParseNode): ContextSet<ExpShape> {
        // TODO: implement this.
        return this.setRetVal(shape).toSet();
    }

    shConcat(shapes: ExpShape[], axis: ExpNum | number, source?: ParseNode): ContextSet<ExpShape> {
        // TODO: implement this.
        return this.setRetVal(shapes[0]).toSet();
    }

    shStack(shapes: ExpShape[], axis: ExpNum | number, source?: ParseNode): ContextSet<ExpShape> {
        // TODO: implement this.
        return this.setRetVal(shapes[0]).toSet();
    }

    shRepeat(shape: ExpShape, axis: ExpNum | number, count: ExpNum | number, source?: ParseNode): ContextSet<ExpShape> {
        // TODO: implement this.
        const rank = ExpShape.getRank(shape);
        const axisCtr = this.genAnd(this.genLte(0, axis, source), this.genLt(axis, rank, source), source);
        const countCtr = this.genLte(0, count, source);

        return this.require([axisCtr, countCtr], 'shRepeat constraint failed', source).map((ctx) => {
            const leftShape = ExpShape.slice(shape, undefined, axis, source);
            const rightShape = ExpShape.slice(shape, axis, undefined, source);
            const repeated = ExpShape.concat(
                ExpShape.concat(leftShape, ExpShape.fromConst(1, [count], source), source),
                rightShape,
                source
            );
            return ctx.setRetVal(repeated);
        });
    }

    require(ctr: Constraint | Constraint[], failMsg?: string, source?: ParseNode): ContextSet<T | SVError> {
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

    ifThenElse(ctr: Constraint, source?: ParseNode): [ContextSet<T>, ContextSet<T>] {
        return this.toSet().ifThenElse(ctr, source);
    }

    // immediate linear SMT : generates conservative range.
    getCachedRange(num: number | ExpNum): NumRange | undefined {
        return this.ctrSet.getCachedRange(num);
    }

    checkImmediate(constraint: Constraint): boolean | undefined {
        return this.ctrSet.checkImmediate(constraint);
    }

    // Check if this ctx has path constraints.
    hasPathCtr(): boolean {
        return !this.ctrSet.pathCtr.isEmpty();
    }
}

export class ContextSetImpl<T> implements ContextSet<T> {
    env: false;
    // running path
    private _ctxList: List<Context<T>>;
    // failed path
    private _failed: List<Context<SVError>>;
    // stopped path: failed but possibly unreachable path
    // TODO: path constraint resolution
    private _stopped: List<Context<SVError>>;

    constructor(ctxList: List<Context<T>>, failed?: List<Context<SVError>>, stopped?: List<Context<SVError>>) {
        this.env = false;
        this._ctxList = ctxList;
        this._failed = failed ? failed : List();
        this._stopped = stopped ? stopped : List();
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
        return this._ctxList;
    }

    getFailed(): List<Context<SVError>> {
        return this._failed;
    }

    getStopped(): List<Context<SVError>> {
        return this._stopped;
    }

    filter(tester: (ctx: Context<T>) => boolean): ContextSet<T> {
        return new ContextSetImpl(this._ctxList.filter(tester), this._failed, this._stopped);
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
        return new ContextSetImpl(succeed, this._failed.concat(failed), this._stopped.concat(stopped));
    }

    map<A>(mapper: (ctx: Context<T>) => Context<A>): ContextSet<A> {
        return this.setCtxList(this._ctxList.map(mapper));
    }

    flatMap<A>(mapper: (ctx: Context<T>) => ContextSet<A>): ContextSet<A> {
        const mapped = this._ctxList.map(mapper);
        const succeed = mapped.toArray().flatMap((cs) => cs.getList().toArray());
        const failed = mapped.toArray().flatMap((cs) => cs.getFailed().toArray());
        const stopped = mapped.toArray().flatMap((cs) => cs.getStopped().toArray());

        return new ContextSetImpl(
            List(succeed),
            List(failed).concat(this._failed),
            List(stopped).concat(this._stopped)
        );
    }

    return<A>(retVal: A): ContextSet<A> {
        return new ContextSetImpl(
            this._ctxList.map((ctx) => ctx.setRetVal(retVal)),
            this._failed,
            this._stopped
        );
    }

    fail(errMsg: string, source?: ParseNode): ContextSet<SVError> {
        return new ContextSetImpl(
            List(),
            this._ctxList
                .filter((ctx) => {
                    !ctx.hasPathCtr();
                })
                .map((ctx) => {
                    const err = SVError.create(errMsg, source);
                    return ctx.fail(err);
                })
                .concat(this._failed),
            this._ctxList
                .filter((ctx) => {
                    ctx.hasPathCtr();
                })
                .map((ctx) => {
                    const err = SVError.create(errMsg, source);
                    return ctx.fail(err);
                })
                .concat(this._stopped)
        );
    }

    join<A>(ctxSet: ContextSet<A>): ContextSet<A | T> {
        const thisList: List<Context<T | A>> = this._ctxList;
        const thatList: List<Context<T | A>> = ctxSet.getList();

        // TODO: precalculate failed set.
        const failedSet: { [id: number]: Context<SVError> } = {};
        const thatFailed = ctxSet.getFailed();
        this._failed.forEach((ctx) => (failedSet[ctx.failId] = ctx));
        thatFailed.forEach((ctx) => (failedSet[ctx.failId] = ctx));

        const stoppedSet: { [id: number]: Context<SVError> } = {};
        const thatStopped = ctxSet.getStopped();
        this._stopped.forEach((ctx) => (stoppedSet[ctx.failId] = ctx));
        thatStopped.forEach((ctx) => (stoppedSet[ctx.failId] = ctx));

        return new ContextSetImpl(
            thisList.concat(thatList),
            List(Object.values(failedSet)),
            List(Object.values(stoppedSet))
        );
    }

    require(ctr: Constraint | Constraint[], failMsg?: string, source?: ParseNode): ContextSet<T | SVError> {
        // immediate constraint check
        const ctrList: Constraint[] = Array.isArray(ctr) ? ctr : [ctr];

        return this.map((ctx) => {
            const ctrSet = ctx.ctrSet.requireAll(ctrList);
            if (ctrSet.valid === false) {
                const ctrMsg = ctrList.map((ctr) => `  ${ctrToStr(ctr)}`).join('\n');
                const errMsg = `${failMsg ? failMsg : 'runtime constraint mismatch'}\n  CONSTRAINTS:\n ${ctrMsg}\n`;
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

    ifThenElse(ctr: Constraint, source?: ParseNode): [ContextSet<T>, ContextSet<T>] {
        const ifPath: Context<T>[] = [];
        const elsePath: Context<T>[] = [];

        this._ctxList.forEach((ctx) => {
            const ctrSet = ctx.ctrSet.addIf(ctr);
            if (ctrSet.valid !== false) {
                ifPath.push(ctx.setCtrSet(ctrSet));
            }

            const elseSet = ctx.ctrSet.addIf(ctx.ctrSet.genNot(ctr, source));
            if (elseSet.valid !== false) {
                elsePath.push(ctx.setCtrSet(elseSet));
            }
        });

        return [
            new ContextSetImpl(List(ifPath), this._failed, this._stopped),
            new ContextSetImpl(List(elsePath), this._failed, this._stopped),
        ];
    }

    isEmpty(): boolean {
        return this._ctxList.isEmpty();
    }

    addLog(message: string, source?: ParseNode): ContextSet<T> {
        return this.map((ctx) => ctx.addLog(message, source));
    }
    addLogValue(value: ShValue): ContextSet<T> {
        return this.map((ctx) => ctx.addLogValue(value));
    }
}
