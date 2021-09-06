/*
 * constraintSet.ts
 * Copyright (c) Seoul National University.
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo (mersshs@gmail.com)
 *
 * Environment and heaps for dynamic semantics of PyTea internal languages.
 */
import chalk from 'chalk';
import { List, Map, Record, Set } from 'immutable';

import { FilePathStore } from '../service/executionPaths';
import { PyteaService } from '../service/pyteaService';
import { formatCodeSource } from '../service/pyteaUtils';
import { absIndexByLen, sanitizeSource } from './backUtils';
import { ConstraintSolver, expToCtr } from './constraintSolver';
import {
    CompareConstraintType,
    Constraint,
    ConstraintIndex,
    ConstraintType,
    CtrAnd,
    CtrBroad,
    CtrExpBool,
    CtrFail,
    CtrForall,
    CtrNot,
    CtrOr,
    ctrToStr,
    EqualityConstraint,
    NumConstraint,
} from './constraintType';
import { isStructuallyEq, simplifyConstraint, simplifyExp } from './expUtils';
import { NumRange } from './range';
import { CodeSource } from './sharpValues';
import {
    BoolOpType,
    ExpBool,
    ExpNum,
    ExpShape,
    ExpShapeConst,
    ExpString,
    NumBopType,
    NumOpType,
    NumUopType,
    SEType,
    ShapeOpType,
    StringOpType,
    SymbolIndex,
    SymbolType,
    SymBool,
    SymExp,
    SymFloat,
    SymInt,
    SymShape,
    SymString,
    SymVal,
} from './symExpressions';

export interface ConstraintGen {
    genSymInt(name: string, source: CodeSource | undefined): SymInt;
    genSymFloat(name: string, source: CodeSource | undefined): SymFloat;
    genSymBool(name: string, source: CodeSource | undefined): SymBool;
    genSymString(name: string, source: CodeSource | undefined): SymString;
    genSymShape(name: string, rank: ExpNum, source: CodeSource | undefined): SymShape;
    genSymIntGte(name: string, min: number | ExpNum, source: CodeSource | undefined): CSReturn<SymInt>;
    genSymIntEq(name: string, min: number | ExpNum, source: CodeSource | undefined): CSReturn<SymInt>;
    genSymFloatGte(name: string, min: number | ExpNum, source: CodeSource | undefined): CSReturn<SymFloat>;
    genShaped(
        name: string,
        rank: number,
        dims: (ExpNum | number)[] | undefined,
        source: CodeSource | undefined
    ): CSReturn<ExpShapeConst>;

    genFromBool(exp: ExpBool, source: CodeSource | undefined): CtrExpBool;
    genEquality(
        type: ConstraintType.Equal | ConstraintType.NotEqual,
        left: SymExp,
        right: SymExp,
        source: CodeSource | undefined
    ): EqualityConstraint;
    genNumCompare(
        type: CompareConstraintType,
        left: ExpNum,
        right: ExpNum,
        source: CodeSource | undefined
    ): NumConstraint | EqualityConstraint;

    genAnd(left: Constraint, right: Constraint, source: CodeSource | undefined): CtrAnd;
    genOr(left: Constraint, right: Constraint, source: CodeSource | undefined): CtrOr;
    genNot(constraint: Constraint, source: CodeSource | undefined): CtrNot;
    genBroad(left: ExpShape, right: ExpShape, source: CodeSource | undefined): CtrBroad;
    genForall(
        symbol: SymInt,
        range: [number | ExpNum, number | ExpNum],
        constraint: Constraint,
        source: CodeSource | undefined
    ): CtrForall;
    genFail(reason: string, source: CodeSource | undefined): CtrFail;
}

// ID Manager is shared for all paths.
class IdManager {
    ctrIdMax: ConstraintIndex;
    symIdMax: SymbolIndex;

    constructor() {
        this.ctrIdMax = 0;
        this.symIdMax = 0;
    }

    getCtrId(): ConstraintIndex {
        return ++this.ctrIdMax;
    }

    getSymId(): SymbolIndex {
        return ++this.symIdMax;
    }
}

interface ConstraintSetProps {
    readonly idManager: IdManager;
    readonly ctrPool: List<Constraint>; // do not shrink it or delete somthing in it!

    // these 3 values contains indices of ctrPool
    readonly hardCtr: List<number>; // constraints that cannot be violated. added by guarantee or genSymXXXGte, genShaped
    readonly softCtr: List<number>; // constraints that can be violated. added by require
    readonly pathCtr: List<number>; // constraints that indicates paths conditions, added by if/then/else clause.

    readonly notPrunedCtrMax: number; // max index of constraint that is checked pruning condition.
    readonly ctrIdCache: Set<ConstraintIndex>;
    readonly rangeCache: Map<SymbolIndex, NumRange>;
    readonly stringCache: Map<SymbolIndex, string>;
    readonly nonStringCache: Map<SymbolIndex, Set<string>>;
    readonly valid: boolean | undefined;
}

const constraintSetDefaults: ConstraintSetProps = {
    idManager: new IdManager(),
    ctrPool: List(),
    hardCtr: List(),
    softCtr: List(),
    pathCtr: List(),
    notPrunedCtrMax: -1,
    ctrIdCache: Set(),
    rangeCache: Map<number, NumRange>(),
    stringCache: Map<number, string>(),
    nonStringCache: Map<number, Set<string>>(),
    valid: true,
};

// return error message if generated constraint has trivial error
export type CSReturnE<T> = [T, ConstraintSet] | string;
export type CSReturn<T> = [T, ConstraintSet];

export class ConstraintSet extends Record(constraintSetDefaults) implements ConstraintSetProps, ConstraintGen {
    constructor(params?: Partial<ConstraintSetProps>) {
        params ? super(params) : super();
    }

    toString(pathStore?: FilePathStore): string {
        return this.ctrPool
            .map((ctr) => simplifyConstraint(this, ctr))
            .map((ctr, idx) => {
                const str = ctrToStr(ctr);
                const source = formatCodeSource(ctr.source, pathStore);
                const result = `${idx + 1}: ${str} - ${source}`;

                if (this.hardCtr.contains(idx)) {
                    return chalk.gray(result);
                } else if (this.pathCtr.contains(idx)) {
                    return chalk.yellow(result);
                } else {
                    return result;
                }
            })
            .join('\n');
    }

    count(): number {
        return this.ctrPool.count();
    }

    getConstraints(): Constraint[] {
        return this.ctrPool.map((ctr) => simplifyConstraint(this, ctr)).toArray();
    }

    getConstraintObject(pathStore?: FilePathStore): object {
        const ctrList = this.getConstraints().map((ctr) => {
            const codePos = formatCodeSource(ctr.source, pathStore);
            const str = ctrToStr(simplifyConstraint(this, ctr));
            if (codePos) {
                ctr.message = `${ctr.message ?? '-'} at <${codePos}>\n  ${str}`;
            } else {
                ctr.message = `constraint at <${codePos}>\n  ${str}`;
            }
            return ctr;
        });

        return {
            ctrPool: sanitizeSource(ctrList, pathStore),
            hardCtr: this.hardCtr.toArray(),
            softCtr: this.softCtr.toArray(),
            pathCtr: this.pathCtr.toArray(),
        };
    }

    getConstraintJSON(pathStore?: FilePathStore): string {
        return JSON.stringify(this.getConstraintObject(pathStore));
    }

    // return false if added constraint is trivially unsat.
    require(constraint: Constraint): ConstraintSet {
        const immCheck = this.checkImmediate(constraint);

        if (immCheck === true) {
            // do not append trivial cases.
            return this;
        } else if (immCheck === false) {
            return this.set('valid', false)._pushSoft(constraint);
        }

        return this._pushSoft(constraint);
    }

    requireAll(constraint: Constraint[]): ConstraintSet {
        let cs: ConstraintSet = this;
        constraint.forEach((ctr) => {
            cs = cs.require(ctr);
        });
        return cs;
    }

    guarantee(constraint: Constraint): ConstraintSet {
        const immCheck = this.checkImmediate(constraint);

        if (immCheck === true) {
            // do not append trivial cases.
            return this;
        } else if (immCheck === false) {
            return this.set('valid', false)._pushHard(constraint);
        }

        return this._cacheConstraint(constraint)._pushHard(constraint);
    }

    guaranteeAll(constraint: Constraint[]): ConstraintSet {
        let cs: ConstraintSet = this;
        constraint.forEach((ctr) => {
            cs = cs.guarantee(ctr);
        });
        return cs;
    }

    addIf(constraint: Constraint): ConstraintSet {
        const immCheck = this.checkImmediate(constraint);

        if (immCheck === true) {
            // do not append trivial cases.
            return this;
        } else if (immCheck === false) {
            return this.set('valid', false)._pushPath(constraint);
        }

        return this._cacheConstraint(constraint)._pushPath(constraint);
    }

    addIfAll(constraint: Constraint[]): ConstraintSet {
        let cs: ConstraintSet = this;
        constraint.forEach((ctr) => {
            cs = cs.addIf(ctr);
        });
        return cs;
    }

    /// ConstraintGen Implementations.

    genSymInt(name: string, source: CodeSource | undefined): SymInt {
        const id = this._getNextSymId();
        return {
            type: SymbolType.Int,
            id,
            name: `${name}_I${id}`,
            source,
        };
    }

    genSymFloat(name: string, source: CodeSource | undefined): SymFloat {
        const id = this._getNextSymId();
        return {
            type: SymbolType.Float,
            id,
            name: `${name}_F${id}`,
            source,
        };
    }

    genSymBool(name: string, source: CodeSource | undefined): SymBool {
        const id = this._getNextSymId();
        return {
            type: SymbolType.Bool,
            id,
            name: `${name}_B${id}`,
            source,
        };
    }

    genSymString(name: string, source: CodeSource | undefined): SymString {
        const id = this._getNextSymId();
        return {
            type: SymbolType.String,
            id,
            name: `${name}_Str${id}`,
            source,
        };
    }

    genSymShape(name: string, rank: ExpNum, source: CodeSource | undefined): SymShape {
        const id = this._getNextSymId();
        return {
            type: SymbolType.Shape,
            id,
            name: `${name}_Shp${id}`,
            rank,
            source,
        };
    }

    genSymIntGte(name: string, min: number | ExpNum, source: CodeSource | undefined): CSReturn<SymInt> {
        const newSym = this.genSymInt(name, source);
        const comp = this.genNumCompare(
            ConstraintType.LessThanOrEqual,
            ExpNum.fromSymbol(newSym),
            SymExp.fromConst(min) as ExpNum,
            source
        );
        return [newSym, this.guarantee(comp)];
    }

    genSymIntEq(name: string, value: number | ExpNum, source: CodeSource | undefined): CSReturn<SymInt> {
        const newSym = this.genSymInt(name, source);
        const exp = SymExp.fromConst(value) as ExpNum;
        const ctr = this.genEquality(ConstraintType.Equal, ExpNum.fromSymbol(newSym), exp, source);

        // bypass guarantee
        const range = this.getCachedRange(value);
        if (range?.valid()) {
            const cache = this.rangeCache.set(newSym.id, range);
            const newCtr = this.set('rangeCache', cache);
            return [newSym, newCtr._pushHard(ctr)];
        } else {
            return [newSym, this.guarantee(ctr)];
        }
    }

    genSymFloatGte(name: string, min: number | ExpNum, source: CodeSource | undefined): CSReturn<SymFloat> {
        const newSym = this.genSymFloat(name, source);
        const comp = this.genNumCompare(
            ConstraintType.LessThanOrEqual,
            ExpNum.fromSymbol(newSym),
            SymExp.fromConst(min) as ExpNum,
            source
        );
        return [newSym, this.guarantee(comp)];
    }

    // This method clearly differs from `ExpShape.fromConst`.
    // It automatically generates constraints and push it to constraint set.
    genShaped(
        name: string,
        rank: number,
        dims: (ExpNum | number)[] | undefined,
        source: CodeSource | undefined
    ): CSReturn<ExpShapeConst> {
        if (rank < 0) {
            throw `making shape '${name} got negative rank ${rank}`;
        }

        let cs: ConstraintSet = this;

        // TODO: check our theory. What if the rank is not equal with the length of dims?
        const newDims: ExpNum[] = [];
        if (!dims) {
            for (let i = 0; i < rank; i++) {
                const [newDim, newCs] = this.genSymIntGte(`${name}_dim${i}`, 0, source);
                cs = newCs;

                const dimExp = ExpNum.fromSymbol(newDim);
                newDims.push(dimExp);
            }
        } else {
            for (const dim of dims) {
                if (typeof dim === 'number') {
                    newDims.push(ExpNum.fromConst(dim, source));
                } else {
                    newDims.push(dim);
                }
            }
        }

        return [ExpShape.fromConst(rank, newDims, source), cs];
    }

    // boolean to integer cast
    castBoolToInt(exp: boolean | ExpBool, source: CodeSource | undefined): CSReturn<number | ExpNum> {
        if (typeof exp === 'boolean') {
            return [+exp, this];
        }
        if (exp.opType === BoolOpType.Const) {
            return [exp.value ? 1 : 0, this as ConstraintSet];
        }

        const isTrue = this.checkImmediate(exp);

        if (isTrue !== undefined) {
            return [isTrue ? 1 : 0, this as ConstraintSet];
        }

        // add (exp && num == 1) || (!exp && num == 0)
        const ctr = this.genFromBool(exp, source);
        const ctrNot = this.genNot(ctr, source);
        const num = ExpNum.fromSymbol(this.genSymInt(`num$castbool_${ctr.id}`, source));

        const numZero = this.genEquality(ConstraintType.Equal, num, ExpNum.fromConst(0, source), source);
        const numOne = this.genEquality(ConstraintType.Equal, num, ExpNum.fromConst(1, source), source);

        const finalCtr = this.genOr(this.genAnd(ctr, numOne, source), this.genAnd(ctrNot, numZero, source), source);

        // return num
        return [num, this.guarantee(finalCtr)];
    }

    // integer to boolean cast
    castNumToBool(exp: number | ExpNum, source: CodeSource | undefined): CSReturnE<boolean | ExpBool> {
        if (typeof exp === 'number') {
            return [!!exp, this];
        }

        const range = this.getCachedRange(exp);
        if (range === undefined) {
            return 'castNumToBool: exp is non-numeric';
        }

        if (range.gt(0) || range.lt(0)) return [true, this];
        else if (range.eq(0)) return [false, this];

        const isZero = this.genEquality(ConstraintType.Equal, exp, ExpNum.fromConst(0, source), source);
        const isNZero = this.genEquality(ConstraintType.NotEqual, exp, ExpNum.fromConst(0, source), source);

        const sym = this.genSymBool(`bool$castnum_${isZero.id}`, source);
        const expSym = ExpBool.fromSymbol(sym);
        const ctrSym = this.genFromBool(expSym, source);

        const finalCtr = this.genOr(
            this.genAnd(ctrSym, isNZero, source),
            this.genAnd(this.genNot(ctrSym, source), isZero, source),
            source
        );
        return [expSym, this.guarantee(finalCtr)];
    }

    /// CONSTRAINT GENERATOR

    genFromBool(exp: ExpBool, source: CodeSource | undefined): CtrExpBool {
        const id = this._getNextCtrId();
        return {
            type: ConstraintType.ExpBool,
            id,
            exp,
            source,
        };
    }

    // return cached sat or unsat. if unsat or found exceptions, return error message.
    genEquality(
        type: ConstraintType.Equal | ConstraintType.NotEqual,
        left: SymExp,
        right: SymExp,
        source: CodeSource | undefined
    ): EqualityConstraint {
        if (left.expType === SEType.Num && right.expType === SEType.Num) {
            return this.genNumCompare(type, left, right, source) as EqualityConstraint;
        }

        const id = this._getNextCtrId();
        const constraint: EqualityConstraint = {
            type,
            id,
            left,
            right,
            source,
        };

        return constraint;
    }

    // return cached sat or unsat or error message
    genNumCompare(
        type: CompareConstraintType,
        left: number | ExpNum,
        right: number | ExpNum,
        source: CodeSource | undefined
    ): NumConstraint | EqualityConstraint {
        const id = this._getNextCtrId();
        left = typeof left === 'number' ? ExpNum.fromConst(left, source) : left;
        right = typeof right === 'number' ? ExpNum.fromConst(right, source) : right;
        const constraint: NumConstraint | EqualityConstraint = {
            type,
            id,
            left,
            right,
            source,
        };

        return constraint;
    }

    genAnd(left: Constraint, right: Constraint, source: CodeSource | undefined): CtrAnd {
        const id = this._getNextCtrId();
        const constraint: CtrAnd = {
            type: ConstraintType.And,
            id,
            left,
            right,
            source,
        };

        return constraint;
    }

    genOr(left: Constraint, right: Constraint, source: CodeSource | undefined): CtrOr {
        const id = this._getNextCtrId();
        const constraint: CtrOr = {
            type: ConstraintType.Or,
            id,
            left,
            right,
            source,
        };

        return constraint;
    }

    genNot(constraint: Constraint, source: CodeSource | undefined): CtrNot {
        const id = this._getNextCtrId();
        const notCtr: CtrNot = {
            type: ConstraintType.Not,
            id,
            constraint,
            source,
        };

        return notCtr;
    }

    genBroad(left: ExpShape, right: ExpShape, source: CodeSource | undefined): CtrBroad {
        const id = this._getNextCtrId();
        const constraint: CtrBroad = {
            type: ConstraintType.Broadcastable,
            id,
            left,
            right,
            source,
        };

        return constraint;
    }

    genForall(
        symbol: SymInt,
        range: [number | ExpNum, number | ExpNum],
        constraint: Constraint,
        source: CodeSource | undefined
    ): CtrForall {
        const id = this._getNextCtrId();

        const expRange: [ExpNum, ExpNum] = [
            typeof range[0] === 'number' ? ExpNum.fromConst(range[0], source) : range[0],
            typeof range[1] === 'number' ? ExpNum.fromConst(range[1], source) : range[1],
        ];

        return {
            type: ConstraintType.Forall,
            id,
            symbol,
            range: expRange,
            constraint,
            source,
        };
    }

    genFail(reason: string, source: CodeSource | undefined): CtrFail {
        const id = this._getNextCtrId();
        const failCtr: CtrFail = {
            type: ConstraintType.Fail,
            id,
            reason,
            source,
        };

        return failCtr;
    }

    /// UTILITIES

    // evaluate exp and return range, or undefined if it is not valid
    getCachedRange(exp: number | ExpNum): NumRange | undefined {
        if (typeof exp === 'number') {
            return NumRange.fromConst(exp);
        }

        switch (exp.opType) {
            case NumOpType.Const:
                return NumRange.fromConst(exp.value);
            case NumOpType.Uop: {
                const baseRng = this.getCachedRange(exp.baseValue);
                if (!baseRng) return undefined;
                let rng: NumRange | undefined;
                switch (exp.uopType) {
                    case NumUopType.Neg:
                        rng = baseRng.neg();
                        break;
                    case NumUopType.Ceil:
                        rng = baseRng.ceil();
                        break;
                    case NumUopType.Floor:
                        rng = baseRng.floor();
                        break;
                    case NumUopType.Abs:
                        rng = baseRng.abs();
                        break;
                }

                return rng?.valid() ? rng : undefined;
            }
            case NumOpType.Bop: {
                const leftRng = this.getCachedRange(exp.left);
                const rightRng = this.getCachedRange(exp.right);
                if (!leftRng || !rightRng) return undefined;
                const calced = this._calcRangeBop(exp.bopType, leftRng, rightRng);
                return calced.valid() ? calced : undefined;
            }
            case NumOpType.Symbol: {
                const cache = this.rangeCache.get(exp.symbol.id);
                if (!cache) {
                    return NumRange.genTop();
                }
                return cache.valid() ? cache : undefined;
            }
            case NumOpType.Max:
                if (exp.values.length === 0) {
                    return undefined;
                } else {
                    const calced = exp.values
                        .map((num) => this.getCachedRange(num))
                        .reduce((prevRng, currRng) => {
                            return currRng ? prevRng?.max(currRng) : undefined;
                        });
                    return calced?.valid() ? calced : undefined;
                }
            case NumOpType.Min:
                if (exp.values.length === 0) {
                    return undefined;
                } else {
                    const calced = exp.values
                        .map((num) => this.getCachedRange(num))
                        .reduce((prevRng, currRng) => {
                            return currRng ? prevRng?.min(currRng) : undefined;
                        });
                    return calced?.valid() ? calced : undefined;
                }
            // TODO: write this.
            case NumOpType.Index: {
                const idxRng = this.getCachedRange(exp.index);
                if (!idxRng || !idxRng.valid()) return;
                if (!idxRng.isConst()) return NumRange.genGte(0);
                const idx = idxRng.start;
                const shape = exp.baseShape;

                switch (shape.opType) {
                    case ShapeOpType.Const:
                        if (idx < shape.dims.length) {
                            return this.getCachedRange(shape.dims[idx]);
                        } else {
                            return;
                        }
                    case ShapeOpType.Set: {
                        const setIdx = this.getCachedRange(shape.axis);
                        if (!setIdx) return;
                        if (setIdx.isConst() && setIdx.start === idx) return this.getCachedRange(shape.dim);
                        if (setIdx.contains(idx)) return NumRange.genGte(0);
                        return this.getCachedRange(ExpNum.index(shape.baseShape, idx, exp.source));
                    }
                    case ShapeOpType.Concat: // TODO: get first.
                    case ShapeOpType.Slice:
                    case ShapeOpType.Broadcast:
                    case ShapeOpType.Symbol: {
                        const shapeCache = this.getCachedShape(shape);
                        if (!shapeCache || idx >= shapeCache.length) return;
                        return this.getCachedRange(shapeCache[idx]);
                    }
                    default:
                        return;
                }
            }
            case NumOpType.Numel: {
                const shape = this.getCachedShape(exp.shape);
                if (shape) {
                    if (shape.length === 0) return NumRange.fromConst(0);
                    let rng = NumRange.fromConst(1);
                    for (const dim of shape) {
                        const dimRng = this.getCachedRange(dim);
                        if (!dimRng) {
                            return;
                        }
                        rng = rng.mul(dimRng);
                    }
                    if (rng.valid()) return rng;
                }
                return;
            }
        }
    }

    // evaluate exp and return constant dimensions if statically calculatable
    getCachedShape(exp: ExpShape): ExpNum[] | undefined {
        switch (exp.opType) {
            case ShapeOpType.Const:
                return exp.dims;
            case ShapeOpType.Symbol:
                // TODO: cache shape
                return;
            case ShapeOpType.Set: {
                const base = this.getCachedShape(exp.baseShape);
                const axis = this.getCachedRange(exp.axis);
                if (base && axis && axis.isConst() && base.length > axis.end) {
                    const temp = base.slice();
                    temp[axis.start] = exp.dim;
                    return temp;
                }
                return;
            }
            case ShapeOpType.Slice:
            case ShapeOpType.Concat:
            case ShapeOpType.Broadcast:
                break;
        }
        return;
    }

    // evaluate exp and return string if statically calculatable.
    getCachedString(exp: string | ExpString): string | undefined {
        if (typeof exp === 'string') {
            return exp;
        }

        switch (exp.opType) {
            case StringOpType.Const:
                return exp.value;
            case StringOpType.Concat: {
                const str1 = this.getCachedString(exp.left);
                const str2 = this.getCachedString(exp.right);
                if (str1 !== undefined && str2 !== undefined) {
                    return str1.concat(str2);
                }
                break;
            }
            case StringOpType.Slice: {
                const str = this.getCachedString(exp.baseString);
                if (str === undefined) break;

                const strLen = str.length;
                let start = 0;
                let end = strLen;

                if (exp.start !== undefined) {
                    const rngTemp = this.getCachedRange(exp.start);
                    if (rngTemp?.isConst() !== true) break;
                    start = rngTemp.start;
                }

                if (exp.end !== undefined) {
                    const rngTemp = this.getCachedRange(exp.end);
                    if (rngTemp?.isConst() !== true) break;
                    end = rngTemp.end;
                }

                return str.slice(absIndexByLen(strLen, start), absIndexByLen(strLen, end));
            }
            case StringOpType.Symbol:
                if (this.stringCache.has(exp.symbol.id)) {
                    return this.stringCache.get(exp.symbol.id);
                }
                break;
        }
        return;
    }

    // check exp is not equal to tester
    checkNonString(exp: string | ExpString, tester: string): boolean | undefined {
        if (typeof exp === 'string') {
            return exp !== tester;
        }

        const strCache = this.getCachedString(exp);
        if (strCache !== undefined) {
            return strCache !== tester;
        }

        if (exp.opType === StringOpType.Symbol) {
            const nonSet = this.nonStringCache.get(exp.symbol.id);
            if (nonSet && nonSet.has(tester)) {
                return true;
            }
        }

        return;
    }

    // return cached equal string or not-equal string
    // getCachedString(symbolId: SymbolIndex): [string | undefined, string[] | undefined] {
    //     return [this._stringCache.get(symbolId), this._nonStringCache.get(symbolId)];
    // }

    // check constraint immediately
    checkImmediate(constraint: ExpBool | Constraint): boolean | undefined {
        // global flag check.
        if (!PyteaService.shouldCheckImmediate()) {
            return;
        }

        // is ExpBool
        if ('expType' in constraint) {
            if (constraint.opType === BoolOpType.Const) {
                return constraint.value;
            }

            constraint = {
                type: ConstraintType.ExpBool,
                id: -1,
                exp: constraint,
            } as CtrExpBool;
        }

        switch (constraint.type) {
            case ConstraintType.ExpBool:
                return this.checkImmediate(expToCtr(constraint.exp));
            case ConstraintType.Equal: {
                const leftExp = simplifyExp(this, constraint.left);
                const rightExp = simplifyExp(this, constraint.right);
                if (leftExp.expType === rightExp.expType) {
                    switch (leftExp.expType) {
                        case SEType.Num:
                            {
                                const left = this.getCachedRange(leftExp);
                                const right = this.getCachedRange(rightExp as ExpNum);
                                if (left && right && left.isConst() && right.isConst())
                                    return left.start === right.start;
                            }
                            break;
                        case SEType.Bool:
                            {
                                let left: boolean | undefined;
                                let right: boolean | undefined;
                                switch (leftExp.opType) {
                                    case BoolOpType.Const:
                                        left = leftExp.value;
                                        break;
                                    case BoolOpType.Symbol:
                                        {
                                            const boolRange = this.getSymbolRange(leftExp.symbol);
                                            if (boolRange?.isConst()) {
                                                left = boolRange.start !== 0;
                                            }
                                        }
                                        break;
                                    default:
                                        left = this.checkImmediate(expToCtr(leftExp));
                                        break;
                                }

                                const rightE = rightExp as ExpBool;
                                switch (rightE.opType) {
                                    case BoolOpType.Const:
                                        right = rightE.value;
                                        break;
                                    case BoolOpType.Symbol:
                                        {
                                            const boolRange = this.getSymbolRange(rightE.symbol);
                                            if (boolRange?.isConst()) {
                                                right = boolRange.start !== 0;
                                            }
                                        }
                                        break;
                                    default:
                                        right = this.checkImmediate(expToCtr(rightE));
                                        break;
                                }
                                if (left !== undefined && right !== undefined) return left === right;
                            }
                            break;
                        case SEType.Shape: {
                            // check falsy constant shape
                            if (
                                leftExp.opType === ShapeOpType.Const &&
                                rightExp.expType === SEType.Shape &&
                                rightExp.opType === ShapeOpType.Const
                            ) {
                                if (leftExp.dims.length !== rightExp.dims.length) {
                                    return false;
                                }
                                for (let i = 0; i < leftExp.dims.length; i++) {
                                    const ld = leftExp.dims[i];
                                    const rd = rightExp.dims[i];
                                    if (
                                        ld.opType === NumOpType.Const &&
                                        rd.opType === NumOpType.Const &&
                                        ld.value !== rd.value
                                    ) {
                                        return false;
                                    }
                                }
                            }
                            break;
                        }
                        case SEType.String:
                        default:
                            break;
                    }
                    const isStructEq = isStructuallyEq(leftExp, rightExp);
                    if (isStructEq) return true;
                    return;
                }
                // TODO: check number / boolean compare (?)
                return false;
            }
            case ConstraintType.NotEqual: {
                const leftExp = simplifyExp(this, constraint.left);
                const rightExp = simplifyExp(this, constraint.right);
                if (leftExp.expType === rightExp.expType) {
                    switch (leftExp.expType) {
                        case SEType.Num: {
                            const left = this.getCachedRange(leftExp);
                            const right = this.getCachedRange(rightExp as ExpNum);
                            if (!left || !right) return;

                            if (left.isConst() && right.isConst()) return !(left.start === right.start);

                            if (left.ltRange(right) || right.ltRange(left)) return true;
                            return undefined;
                        }
                        case SEType.Bool: {
                            const left = this.checkImmediate(expToCtr(leftExp));
                            const right = this.checkImmediate(expToCtr(rightExp as ExpBool));
                            if (left !== undefined && right !== undefined) return left !== right;
                            else return;
                        }
                        case SEType.Shape:
                            {
                                const isStructEq = isStructuallyEq(leftExp, rightExp);
                                if (isStructEq) return false;
                            }
                            break;
                        case SEType.String:
                            {
                                const isStructEq = isStructuallyEq(leftExp, rightExp);
                                if (isStructEq) return false;
                            }
                            break;
                        default:
                            return;
                    }
                    return;
                }
                // TODO: check number / boolean compare (?)
                return true;
            }
            case ConstraintType.And: {
                const left = this.checkImmediate(constraint.left);
                if (left === false) return false;

                const right = this.checkImmediate(constraint.right);

                if (left === true) return right;
                return right === false ? false : undefined;
            }
            case ConstraintType.Or: {
                const left = this.checkImmediate(constraint.left);
                if (left === true) return true;

                const right = this.checkImmediate(constraint.right);

                if (left === false) return right;
                return right === true ? true : undefined;
            }
            case ConstraintType.Not: {
                const inner = this.checkImmediate(constraint.constraint);
                if (inner === undefined) return;
                return !inner;
            }
            case ConstraintType.LessThan: {
                const left = this.getCachedRange(constraint.left);
                const right = this.getCachedRange(constraint.right);
                if (!left || !right) return;
                return left.ltRange(right);
            }
            case ConstraintType.LessThanOrEqual: {
                const left = this.getCachedRange(constraint.left);
                const right = this.getCachedRange(constraint.right);
                if (!left || !right) return;
                return left.lteRange(right);
            }
            case ConstraintType.Broadcastable:
                {
                    const left = simplifyExp(this, constraint.left);
                    const right = simplifyExp(this, constraint.right);
                    if (left.opType === ShapeOpType.Const && right.opType === ShapeOpType.Const) {
                        const [baseShape, rightShape] = left.rank < right.rank ? [right, left] : [left, right];

                        const rankdiff = baseShape.dims.length - rightShape.dims.length;

                        for (let i = 0; i < baseShape.dims.length; i++) {
                            if (i < rankdiff) continue;

                            const dim = this.selectBroadcastable(baseShape.dims[i], rightShape.dims[i - rankdiff]);
                            if (dim === false) {
                                return false;
                            } else if (dim === undefined) {
                                return;
                            }
                        }
                        return true;
                    }
                }
                break;
            case ConstraintType.Forall:
                return;
            case ConstraintType.Fail:
                return false;
        }
        return;
    }

    // return broadcasted maximum value. if cannot infer it is broadcastable, return undefined.
    // if it is never broadcastable, return false
    // (left === right) => left / left === 1 => right / right === 1 => left
    selectBroadcastable(left: ExpNum, right: ExpNum): ExpNum | false | undefined {
        // TODO: check [1, 2)
        const leftRng = this.getCachedRange(left);
        const rightRng = this.getCachedRange(right);
        if (!leftRng || !rightRng) return;

        if (leftRng.isConst()) {
            const ln = leftRng.start;
            if (ln === 1) return right;
            if (rightRng.isConst()) {
                if (rightRng.start === 1) return left;
                if (ln !== rightRng.start) return false;
                return left;
            }
            if (!rightRng.contains(ln)) return false;
        }

        if (rightRng.isConst()) {
            const rn = rightRng.start;
            if (rn === 1) return left;
            if (!leftRng.contains(rn)) return false;
        }

        if (leftRng.intersect(rightRng).valid() === false) return false;
        if (isStructuallyEq(left, right)) return left;

        return;
    }

    // return cached range
    getSymbolRange(symbol: SymInt | SymFloat | SymBool): NumRange | undefined {
        return this.rangeCache.get(symbol.id);
    }

    // filter constraints which has exactly one non-constant symbolic variable
    hasSingleVar(constraint: Constraint): SymVal | boolean {
        switch (constraint.type) {
            case ConstraintType.ExpBool:
                return SymExp.hasSingleVar(this, constraint.exp);
            case ConstraintType.LessThan:
            case ConstraintType.LessThanOrEqual:
            case ConstraintType.Equal:
            case ConstraintType.NotEqual: {
                return SymExp.mergeSingleVar([
                    SymExp.hasSingleVar(this, constraint.left),
                    SymExp.hasSingleVar(this, constraint.right),
                ]);
            }
            case ConstraintType.And:
            case ConstraintType.Or: {
                return SymExp.mergeSingleVar([this.hasSingleVar(constraint.left), this.hasSingleVar(constraint.right)]);
            }
            case ConstraintType.Not:
                return this.hasSingleVar(constraint.constraint);
            case ConstraintType.Forall:
            case ConstraintType.Broadcastable:
            case ConstraintType.Fail:
                return true;
        }
    }

    // cache each symbolic variables' new range constrainted by `constraint`
    private _cacheConstraint(constraint: Constraint): ConstraintSet {
        const solver = new ConstraintSolver(this);

        return solver.solve(constraint).ctrSet;
    }

    private _calcRangeBop(bopType: NumBopType, left: NumRange, right: NumRange): NumRange {
        switch (bopType) {
            case NumBopType.Add:
                return left.add(right);
            case NumBopType.Sub:
                return left.sub(right);
            case NumBopType.Mul:
                return left.mul(right);
            case NumBopType.FloorDiv:
                return left.floordiv(right);
            case NumBopType.TrueDiv:
                return left.truediv(right);
            case NumBopType.Mod:
                return left.mod(right);
        }
    }

    // uncheck satisfiability. just push and cache it.
    private _pushHard(constraint: Constraint): ConstraintSet {
        let ctrSet = this;

        if (!this.ctrIdCache.has(constraint.id)) {
            ctrSet = ctrSet.set('hardCtr', ctrSet.hardCtr.push(ctrSet.ctrPool.count()));
            ctrSet = ctrSet.set('ctrPool', ctrSet.ctrPool.push(constraint));
            ctrSet = ctrSet.set('ctrIdCache', ctrSet.ctrIdCache.add(constraint.id));
        }

        return ctrSet;
    }

    private _pushSoft(constraint: Constraint): ConstraintSet {
        let ctrSet = this;

        if (!this.ctrIdCache.has(constraint.id)) {
            ctrSet = ctrSet.set('softCtr', ctrSet.softCtr.push(ctrSet.ctrPool.count()));
            ctrSet = ctrSet.set('ctrPool', ctrSet.ctrPool.push(constraint));
            ctrSet = ctrSet.set('ctrIdCache', ctrSet.ctrIdCache.add(constraint.id));
        }

        return ctrSet;
    }

    private _pushPath(constraint: Constraint): ConstraintSet {
        let ctrSet = this;

        if (!this.ctrIdCache.has(constraint.id)) {
            ctrSet = ctrSet.set('pathCtr', ctrSet.pathCtr.push(ctrSet.ctrPool.count()));
            ctrSet = ctrSet.set('ctrPool', ctrSet.ctrPool.push(constraint));
            ctrSet = ctrSet.set('ctrIdCache', ctrSet.ctrIdCache.add(constraint.id));
        }

        return ctrSet;
    }

    private _getNextCtrId(): number {
        return this.idManager.getCtrId();
    }

    private _getNextSymId(): number {
        return this.idManager.getSymId();
    }
}
