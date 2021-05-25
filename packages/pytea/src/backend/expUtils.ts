/*
 * expUtils.ts
 * Copyright (c) Seoul National University.
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo (mersshs@gmail.com)
 *
 * Simple expression comparer and calculator
 */

import { fetchAddr, trackMro } from './backUtils';
import { ConstraintSet } from './constraintSet';
import { Constraint, ConstraintType } from './constraintType';
import { Context } from './context';
import { Fraction } from './fraction';
import { ShEnv, ShHeap } from './sharpEnvironments';
import { CodeSource, ShValue, SVAddr, SVType } from './sharpValues';
import {
    BoolOpType,
    ExpBool,
    ExpNum,
    ExpNumBop,
    ExpNumConst,
    ExpShape,
    ExpString,
    NumBopType,
    NumOpType,
    NumUopType,
    SEType,
    ShapeOpType,
    StringOpType,
    SymbolType,
    SymExp,
} from './symExpressions';

// expression with coefficient
export interface CoExpNum<T extends ExpNum> {
    exp: T;
    coeff: Fraction;
}

// normalized sum of indivisible expressions
export interface NormalExp {
    list: CoExpNum<ExpNum>[];
    constant: Fraction;
}

function emptyNormalExp(exp: ExpNum): NormalExp {
    return {
        list: [{ exp, coeff: new Fraction(1, 1) }],
        constant: new Fraction(0, 1),
    };
}

function mergeCoExpList<T extends ExpNum>(list1: CoExpNum<T>[], list2: CoExpNum<T>[]): CoExpNum<T>[] {
    const [mainList, subList] = list1.length >= list2.length ? [list1, list2] : [list2, list1];
    const newList = mainList.map((cexp) => ({ ...cexp }));
    const leftLen = newList.length;
    for (const right of subList) {
        let found = false;
        for (let i = 0; i < leftLen; i++) {
            const left = newList[i];
            if (isStructuallyEq(left.exp, right.exp)) {
                left.coeff = left.coeff.add(right.coeff);
                found = true;
                break;
            }
        }
        if (!found) newList.push({ ...right });
    }

    return newList.filter((n) => n.coeff.up !== 0);
}

// make NormalExp
export function normalizeExpNum(exp: ExpNum): NormalExp {
    switch (exp.opType) {
        case NumOpType.Const:
            return {
                list: [],
                constant: new Fraction(exp.value, 1),
            };
        case NumOpType.Symbol:
            return emptyNormalExp(exp);
        case NumOpType.Uop:
            if (exp.uopType === NumUopType.Neg) {
                const norm = normalizeExpNum(exp.baseValue);
                return {
                    list: norm.list.map((n) => ({ exp: n.exp, coeff: n.coeff.neg() })),
                    constant: norm.constant.neg(),
                };
            }
            return emptyNormalExp(exp);
        case NumOpType.Bop: {
            const left = normalizeExpNum(exp.left);
            let right = normalizeExpNum(exp.right);
            switch (exp.bopType) {
                case NumBopType.Sub:
                    right = {
                        list: right.list.map((n) => ({ exp: n.exp, coeff: n.coeff.neg() })),
                        constant: right.constant.neg(),
                    };
                // eslint-disable-next-line no-fallthrough
                case NumBopType.Add:
                    return {
                        list: mergeCoExpList(left.list, right.list),
                        constant: left.constant.add(right.constant),
                    };
                case NumBopType.Mul: {
                    let newList: CoExpNum<ExpNum>[] = [];
                    for (const lexp of left.list) {
                        const tempList = right.list.map((rexp) => ({
                            exp: ExpNum.bop(NumBopType.Mul, lexp.exp, rexp.exp, exp.source),
                            coeff: lexp.coeff.mul(rexp.coeff),
                        }));
                        newList = mergeCoExpList(newList, tempList);
                    }

                    let isZero = true;
                    if (right.constant.up !== 0) {
                        const tempList = left.list.map((lexp) => ({
                            exp: lexp.exp,
                            coeff: lexp.coeff.mul(right.constant),
                        }));
                        newList = mergeCoExpList(newList, tempList);
                        isZero = false;
                    }
                    if (left.constant.up !== 0) {
                        const tempList = right.list.map((rexp) => ({
                            exp: rexp.exp,
                            coeff: rexp.coeff.mul(left.constant),
                        }));
                        newList = mergeCoExpList(newList, tempList);
                        isZero = false;
                    }

                    return {
                        list: newList,
                        constant: isZero ? new Fraction(0, 1) : left.constant.mul(right.constant),
                    };
                }
                case NumBopType.Mod:
                    if (left.list.length === 0 && right.list.length === 0) {
                        return {
                            list: [],
                            constant: new Fraction(left.constant.toNum() % right.constant.toNum(), 1),
                        };
                    }
                    return emptyNormalExp(
                        ExpNum.bop(NumBopType.Mod, denormalizeExpNum(left), denormalizeExpNum(right), exp.source)
                    );
                case NumBopType.FloorDiv:
                    if (left.list.length === 0 && right.list.length === 0 && right.constant.up !== 0) {
                        return { list: [], constant: left.constant.div(right.constant).floor() };
                    }
                    return emptyNormalExp(
                        ExpNum.bop(NumBopType.FloorDiv, denormalizeExpNum(left), denormalizeExpNum(right), exp.source)
                    );
                case NumBopType.TrueDiv:
                    if (right.list.length === 0 && right.constant.up !== 0) {
                        return {
                            list: left.list.map((c) => ({ exp: c.exp, coeff: c.coeff.div(right.constant) })),
                            constant: left.constant.div(right.constant),
                        };
                    }
                    return emptyNormalExp(
                        ExpNum.bop(NumBopType.TrueDiv, denormalizeExpNum(left), denormalizeExpNum(right), exp.source)
                    );
            }
            break;
        }

        default:
            return emptyNormalExp(exp);
    }
}

export function denormalizeExpNum(nexp: NormalExp): ExpNum {
    let left: ExpNum | undefined;
    for (const cexp of nexp.list) {
        const coeff = cexp.coeff.norm();

        if (coeff.down === 1) {
            if (coeff.up === 1) {
                left = left ? ExpNum.bop(NumBopType.Add, left, cexp.exp, cexp.exp.source) : cexp.exp;
            } else {
                const right = ExpNum.bop(NumBopType.Mul, cexp.exp, coeff.up, cexp.exp.source);
                left = left ? ExpNum.bop(NumBopType.Add, left, right, left.source) : right;
            }
        } else {
            let right: ExpNum;
            if (coeff.up === 1) {
                right = ExpNum.bop(NumBopType.TrueDiv, cexp.exp, coeff.down, cexp.exp.source);
            } else {
                right = ExpNum.bop(
                    NumBopType.TrueDiv,
                    ExpNum.bop(NumBopType.Mul, cexp.exp, coeff.up, cexp.exp.source),
                    coeff.down,
                    cexp.exp.source
                );
            }
            left = left ? ExpNum.bop(NumBopType.Add, left, right, left.source) : right;
        }
    }

    const ncst = nexp.constant.norm();
    if (ncst.up === 0) {
        return left ? left : ExpNum.fromConst(0, undefined);
    }

    const cstExp =
        ncst.down === 1
            ? ExpNum.fromConst(ncst.up, undefined)
            : ExpNum.bop(NumBopType.TrueDiv, ExpNum.fromConst(ncst.up, undefined), ncst.down, undefined);
    return left ? ExpNum.bop(NumBopType.Add, left, cstExp, left.source) : cstExp;
}

export function simplifyString(ctrSet: ConstraintSet, exp: ExpString): ExpString {
    switch (exp.opType) {
        case StringOpType.Concat: {
            const left = simplifyString(ctrSet, exp.left);
            const right = simplifyString(ctrSet, exp.right);
            if (left.opType === StringOpType.Const && right.opType === StringOpType.Const) {
                return ExpString.fromConst(left.value + right.value, exp.source);
            }
            return ExpString.concat(left, right, exp.source);
        }
        case StringOpType.Slice: {
            const base = simplifyString(ctrSet, exp.baseString);
            const start = exp.start ? simplifyNum(ctrSet, exp.start) : undefined;
            const end = exp.end ? simplifyNum(ctrSet, exp.end) : undefined;
            if (
                base.opType === StringOpType.Const &&
                (!start || start.opType === NumOpType.Const) &&
                (!end || end.opType === NumOpType.Const)
            ) {
                const len = base.value.length;
                const startId = start ? (start.value < 0 ? len - start.value : start.value) : 0;
                const endId = end ? (end.value < 0 ? len - end.value : end.value) : len;

                if (startId <= endId) return ExpString.fromConst(base.value.substring(startId, endId), exp.source);
            }

            return ExpString.slice(base, start, end, exp.source);
        }
        case StringOpType.Const:
        case StringOpType.Symbol:
            return exp;
    }
}

export function simplifyBool(ctrSet: ConstraintSet, exp: ExpBool): ExpBool {
    switch (exp.opType) {
        case BoolOpType.And: {
            const left = simplifyBool(ctrSet, exp.left);
            const right = simplifyBool(ctrSet, exp.right);
            if (left.opType === BoolOpType.Const) {
                return left.value ? right : left;
            } else if (right.opType === BoolOpType.Const) {
                return right.value ? left : right;
            }

            // do not check immediate. that's not simplifier's job.
            return ExpBool.and(left, right, exp.source);
        }
        case BoolOpType.Or: {
            const left = simplifyBool(ctrSet, exp.left);
            const right = simplifyBool(ctrSet, exp.right);
            if (left.opType === BoolOpType.Const) {
                return left.value ? left : right;
            } else if (right.opType === BoolOpType.Const) {
                return right.value ? right : left;
            }

            // do not check immediate. that's not simplifier's job.
            return ExpBool.or(left, right, exp.source);
        }
        case BoolOpType.Equal:
        case BoolOpType.NotEqual: {
            const left = simplifyExp(ctrSet, exp.left);
            const right = simplifyExp(ctrSet, exp.right);

            if (left.expType === SEType.Num && right.expType === SEType.Num) {
                if (left.opType === NumOpType.Const && right.opType === NumOpType.Const) {
                    return exp.opType === BoolOpType.Equal
                        ? ExpBool.fromConst(left.value === right.value, exp.source)
                        : ExpBool.fromConst(left.value !== right.value, exp.source);
                }
            } else if (left.expType === SEType.Bool && right.expType === SEType.Bool) {
                if (left.opType === BoolOpType.Const && right.opType === BoolOpType.Const) {
                    return exp.opType === BoolOpType.Equal
                        ? ExpBool.fromConst(left.value === right.value, exp.source)
                        : ExpBool.fromConst(left.value !== right.value, exp.source);
                }
            } else if (left.expType === SEType.String && right.expType === SEType.String) {
                if (left.opType === StringOpType.Const && right.opType === StringOpType.Const) {
                    return exp.opType === BoolOpType.Equal
                        ? ExpBool.fromConst(left.value === right.value, exp.source)
                        : ExpBool.fromConst(left.value !== right.value, exp.source);
                }
            }

            // do not check immediate. that's not simplifier's job.
            return exp.opType === BoolOpType.Equal
                ? ExpBool.eq(left, right, exp.source)
                : ExpBool.neq(left, right, exp.source);
        }
        case BoolOpType.LessThan:
        case BoolOpType.LessThanOrEqual: {
            const left = simplifyNum(ctrSet, exp.left);
            const right = simplifyNum(ctrSet, exp.right);

            if (left.opType === NumOpType.Const && right.opType === NumOpType.Const) {
                return exp.opType === BoolOpType.LessThan
                    ? ExpBool.fromConst(left.value < right.value, exp.source)
                    : ExpBool.fromConst(left.value <= right.value, exp.source);
            }

            // do not check immediate. that's not simplifier's job.
            return exp.opType === BoolOpType.LessThan
                ? ExpBool.lt(left, right, exp.source)
                : ExpBool.lte(left, right, exp.source);
        }
        case BoolOpType.Not: {
            const base = simplifyBool(ctrSet, exp.baseBool);
            switch (base.opType) {
                case BoolOpType.Equal:
                    return ExpBool.neq(base.left, base.right, exp.source);
                case BoolOpType.NotEqual:
                    return ExpBool.eq(base.left, base.right, exp.source);
                case BoolOpType.Const:
                    return ExpBool.fromConst(!base.value, exp.source);
                case BoolOpType.LessThan:
                    return ExpBool.lte(base.right, base.left, exp.source);
                case BoolOpType.LessThanOrEqual:
                    return ExpBool.lt(base.right, base.left, exp.source);
                case BoolOpType.Not:
                    return base.baseBool;
                default:
                    return ExpBool.not(base, exp.source);
            }
        }
        case BoolOpType.Symbol: {
            const rng = ctrSet.getSymbolRange(exp.symbol);
            if (rng && rng.isConst()) {
                return ExpBool.fromConst(rng.start !== 0, exp.symbol.source);
            }
            return exp;
        }
        default:
            return exp;
    }
}

export function simplifyNum(ctrSet: ConstraintSet, exp: ExpNum): ExpNum {
    switch (exp.opType) {
        case NumOpType.Uop: {
            const baseValue = simplifyNum(ctrSet, exp.baseValue);
            switch (baseValue.opType) {
                case NumOpType.Const:
                    switch (exp.uopType) {
                        case NumUopType.Neg:
                            return ExpNum.fromConst(-baseValue.value, exp.source);
                        case NumUopType.Floor:
                            return ExpNum.fromConst(Math.floor(baseValue.value), exp.source);
                        case NumUopType.Ceil:
                            return ExpNum.fromConst(Math.ceil(baseValue.value), exp.source);
                        case NumUopType.Abs:
                            return ExpNum.fromConst(Math.abs(baseValue.value), exp.source);
                    }
                    break;
                case NumOpType.Uop:
                    switch (exp.uopType) {
                        case NumUopType.Ceil:
                        case NumUopType.Floor:
                            // if inner value is structually integer, remove ceil and floor
                            if (isStructuallyInt(baseValue)) {
                                return baseValue;
                            }
                            break;
                        case NumUopType.Neg:
                            // double negation
                            if (baseValue.uopType === NumUopType.Neg) {
                                return baseValue.baseValue;
                            }
                            break;
                        case NumUopType.Abs:
                            // cancel double abs
                            if (baseValue.uopType === NumUopType.Abs) {
                                return baseValue;
                            }
                            break;
                    }
                    break;
                default:
                    break;
            }

            const rng = ctrSet.getCachedRange(baseValue);
            if (rng) {
                if (rng.isConst()) {
                    const base = rng.start;
                    switch (exp.uopType) {
                        case NumUopType.Neg:
                            return ExpNum.fromConst(-base, exp.source);
                        case NumUopType.Floor:
                            return ExpNum.fromConst(Math.floor(base), exp.source);
                        case NumUopType.Ceil:
                            return ExpNum.fromConst(Math.ceil(base), exp.source);
                        case NumUopType.Abs:
                            return ExpNum.fromConst(Math.abs(base), exp.source);
                    }
                }

                if (exp.uopType === NumUopType.Abs) {
                    if (rng.gte(0)) {
                        return baseValue;
                    } else if (rng.lte(0)) {
                        return simplifyNum(ctrSet, ExpNum.uop(NumUopType.Neg, baseValue, exp.source));
                    }
                }
            }

            return ExpNum.uop(exp.uopType, baseValue, exp.source);
        }
        case NumOpType.Bop: {
            const left = simplifyNum(ctrSet, exp.left);
            const right = simplifyNum(ctrSet, exp.right);
            if (left.opType === NumOpType.Const) {
                const leftVal = left.value;
                if (leftVal === 0) {
                    switch (exp.bopType) {
                        case NumBopType.Add:
                            return right;
                        case NumBopType.FloorDiv:
                        case NumBopType.TrueDiv:
                        case NumBopType.Mod:
                        case NumBopType.Mul:
                            return ExpNum.fromConst(0, exp.source);
                        case NumBopType.Sub:
                            break;
                    }
                } else if (leftVal === 1 && exp.bopType === NumBopType.Mul) {
                    return right;
                }

                if (right.opType === NumOpType.Const) {
                    switch (exp.bopType) {
                        case NumBopType.Add:
                            return ExpNum.fromConst(leftVal + right.value, exp.source);
                        case NumBopType.FloorDiv:
                            return ExpNum.fromConst(Math.floor(leftVal / right.value), exp.source);
                        case NumBopType.Mod:
                            return ExpNum.fromConst(leftVal % right.value, exp.source);
                        case NumBopType.Mul:
                            return ExpNum.fromConst(leftVal * right.value, exp.source);
                        case NumBopType.Sub:
                            return ExpNum.fromConst(leftVal - right.value, exp.source);
                        case NumBopType.TrueDiv:
                            return ExpNum.fromConst(leftVal / right.value, exp.source);
                    }
                } else if (right.opType === NumOpType.Bop) {
                    const rl = right.left;
                    const rr = right.right;
                    switch (right.bopType) {
                        case NumBopType.Add:
                            if (rl.opType === NumOpType.Const) {
                                if (exp.bopType === NumBopType.Add) {
                                    return ExpNum.bop(NumBopType.Add, leftVal + rl.value, rr, exp.source);
                                } else if (exp.bopType === NumBopType.Sub) {
                                    return ExpNum.bop(NumBopType.Sub, leftVal - rl.value, rr, exp.source);
                                } else if (exp.bopType === NumBopType.Mul) {
                                    return ExpNum.bop(
                                        NumBopType.Add,
                                        leftVal * rl.value,
                                        ExpNum.bop(NumBopType.Mul, leftVal, rr, exp.source),
                                        exp.source
                                    );
                                }
                            }
                            if (rr.opType === NumOpType.Const) {
                                if (exp.bopType === NumBopType.Add) {
                                    return ExpNum.bop(NumBopType.Add, leftVal + rr.value, rl, exp.source);
                                } else if (exp.bopType === NumBopType.Sub) {
                                    return ExpNum.bop(NumBopType.Sub, leftVal - rr.value, rl, exp.source);
                                } else if (exp.bopType === NumBopType.Mul) {
                                    return ExpNum.bop(
                                        NumBopType.Add,
                                        ExpNum.bop(NumBopType.Mul, leftVal, rl, exp.source),
                                        leftVal * rr.value,
                                        exp.source
                                    );
                                }
                            }
                            break;
                        case NumBopType.Sub:
                            if (rl.opType === NumOpType.Const) {
                                if (exp.bopType === NumBopType.Add) {
                                    return ExpNum.bop(NumBopType.Sub, leftVal + rl.value, rr, exp.source);
                                } else if (exp.bopType === NumBopType.Sub) {
                                    return ExpNum.bop(NumBopType.Add, leftVal - rl.value, rr, exp.source);
                                } else if (exp.bopType === NumBopType.Mul) {
                                    return ExpNum.bop(
                                        NumBopType.Sub,
                                        leftVal * rl.value,
                                        ExpNum.bop(NumBopType.Mul, leftVal, rr, exp.source),
                                        exp.source
                                    );
                                }
                            }
                            if (rr.opType === NumOpType.Const) {
                                if (exp.bopType === NumBopType.Add) {
                                    return ExpNum.bop(NumBopType.Add, leftVal - rr.value, rl, exp.source);
                                } else if (exp.bopType === NumBopType.Sub) {
                                    return ExpNum.bop(NumBopType.Sub, leftVal + rr.value, rl, exp.source);
                                } else if (exp.bopType === NumBopType.Mul) {
                                    return ExpNum.bop(
                                        NumBopType.Add,
                                        ExpNum.bop(NumBopType.Mul, leftVal, rl, exp.source),
                                        leftVal * rr.value,
                                        exp.source
                                    );
                                }
                            }
                            break;
                        case NumBopType.Mul:
                            if (rl.opType === NumOpType.Const) {
                                if (exp.bopType === NumBopType.Mul) {
                                    return ExpNum.bop(NumBopType.Mul, leftVal * rl.value, rr, exp.source);
                                } else if (exp.bopType === NumBopType.TrueDiv && rl.value !== 0) {
                                    return ExpNum.bop(NumBopType.TrueDiv, leftVal / rl.value, rr, exp.source);
                                }
                            }
                            if (rr.opType === NumOpType.Const) {
                                if (exp.bopType === NumBopType.Mul) {
                                    return ExpNum.bop(NumBopType.Mul, leftVal * rr.value, rl, exp.source);
                                } else if (exp.bopType === NumBopType.TrueDiv && rr.value !== 0) {
                                    return ExpNum.bop(NumBopType.TrueDiv, leftVal / rr.value, rl, exp.source);
                                }
                            }
                            break;
                        case NumBopType.TrueDiv:
                            if (rl.opType === NumOpType.Const) {
                                if (exp.bopType === NumBopType.Mul) {
                                    return ExpNum.bop(NumBopType.TrueDiv, leftVal * rl.value, rr, exp.source);
                                }
                            }
                            break;
                    }
                }
            } else if (right.opType === NumOpType.Const) {
                const rightVal = right.value;
                if (rightVal === 0) {
                    switch (exp.bopType) {
                        case NumBopType.Add:
                        case NumBopType.Sub:
                            return left;
                        case NumBopType.Mul:
                            return ExpNum.fromConst(0, exp.source);
                        default:
                            break;
                    }
                } else if (rightVal === 1) {
                    switch (exp.bopType) {
                        case NumBopType.TrueDiv:
                            return left;
                        case NumBopType.FloorDiv:
                            if (isStructuallyInt(left)) {
                                return left;
                            }
                            break;
                        default:
                            break;
                    }
                }

                if (left.opType === NumOpType.Bop) {
                    const ll = left.left;
                    const lr = left.right;
                    switch (left.bopType) {
                        case NumBopType.Add:
                            if (ll.opType === NumOpType.Const) {
                                if (exp.bopType === NumBopType.Add) {
                                    return ExpNum.bop(NumBopType.Add, lr, ll.value + rightVal, exp.source);
                                } else if (exp.bopType === NumBopType.Sub) {
                                    return ExpNum.bop(NumBopType.Add, lr, ll.value - rightVal, exp.source);
                                } else if (exp.bopType === NumBopType.Mul) {
                                    return ExpNum.bop(
                                        NumBopType.Add,
                                        ll.value * rightVal,
                                        ExpNum.bop(NumBopType.Mul, lr, rightVal, exp.source),
                                        exp.source
                                    );
                                }
                            }
                            if (lr.opType === NumOpType.Const) {
                                if (exp.bopType === NumBopType.Add) {
                                    return ExpNum.bop(NumBopType.Add, ll, lr.value + rightVal, exp.source);
                                } else if (exp.bopType === NumBopType.Sub) {
                                    return ExpNum.bop(NumBopType.Add, ll, lr.value - rightVal, exp.source);
                                } else if (exp.bopType === NumBopType.Mul) {
                                    return ExpNum.bop(
                                        NumBopType.Add,
                                        ExpNum.bop(NumBopType.Mul, ll, rightVal, exp.source),
                                        lr.value * rightVal,
                                        exp.source
                                    );
                                }
                            }
                            break;
                        case NumBopType.Sub:
                            if (ll.opType === NumOpType.Const) {
                                if (exp.bopType === NumBopType.Add) {
                                    return ExpNum.bop(NumBopType.Sub, ll.value + rightVal, lr, exp.source);
                                } else if (exp.bopType === NumBopType.Sub) {
                                    return ExpNum.bop(NumBopType.Sub, ll.value - rightVal, lr, exp.source);
                                } else if (exp.bopType === NumBopType.Mul) {
                                    return ExpNum.bop(
                                        NumBopType.Sub,
                                        ll.value * rightVal,
                                        ExpNum.bop(NumBopType.Mul, lr, rightVal, exp.source),
                                        exp.source
                                    );
                                }
                            }
                            if (lr.opType === NumOpType.Const) {
                                if (exp.bopType === NumBopType.Add) {
                                    return ExpNum.bop(NumBopType.Add, ll, rightVal - lr.value, exp.source);
                                } else if (exp.bopType === NumBopType.Sub) {
                                    return ExpNum.bop(NumBopType.Add, ll, -lr.value - rightVal, exp.source);
                                } else if (exp.bopType === NumBopType.Mul) {
                                    return ExpNum.bop(
                                        NumBopType.Sub,
                                        ExpNum.bop(NumBopType.Mul, ll, rightVal, exp.source),
                                        lr.value * rightVal,
                                        exp.source
                                    );
                                }
                            }
                            break;
                        case NumBopType.Mul:
                            if (ll.opType === NumOpType.Const) {
                                if (exp.bopType === NumBopType.Mul) {
                                    return ExpNum.bop(NumBopType.Mul, ll.value * rightVal, lr, exp.source);
                                } else if (exp.bopType === NumBopType.TrueDiv && rightVal !== 0) {
                                    return ExpNum.bop(NumBopType.Mul, ll.value / rightVal, lr, exp.source);
                                }
                            }
                            if (lr.opType === NumOpType.Const) {
                                if (exp.bopType === NumBopType.Mul) {
                                    return ExpNum.bop(NumBopType.Mul, ll, lr.value * rightVal, exp.source);
                                } else if (exp.bopType === NumBopType.TrueDiv && rightVal !== 0) {
                                    return ExpNum.bop(NumBopType.Mul, ll, lr.value / rightVal, exp.source);
                                }
                            }
                            break;
                        case NumBopType.TrueDiv:
                            if (ll.opType === NumOpType.Const) {
                                if (exp.bopType === NumBopType.Mul) {
                                    return ExpNum.bop(NumBopType.TrueDiv, ll.value * rightVal, lr, exp.source);
                                }
                            }
                            break;
                    }
                }
            }

            return ExpNum.bop(exp.bopType, left, right, exp.source);
        }
        case NumOpType.Const:
            return exp;
        case NumOpType.Index: {
            let base = simplifyShape(ctrSet, exp.baseShape);
            const idx = simplifyNum(ctrSet, exp.index);

            if (idx.opType === NumOpType.Const) {
                let doBreak = false;
                let index = idx.value;
                if (Math.floor(index) !== index) {
                    return ExpNum.index(base, idx, exp.source);
                }

                while (!doBreak) {
                    switch (base.opType) {
                        case ShapeOpType.Const:
                            if (0 <= index && index < base.dims.length) {
                                return simplifyNum(ctrSet, base.dims[index]);
                            } else {
                                doBreak = true;
                            }
                            break;
                        case ShapeOpType.Concat:
                            {
                                const firstRank = ExpShape.getRank(base.left);
                                const rankRng = ctrSet.getCachedRange(firstRank);
                                if (rankRng?.lte(index)) {
                                    // take second
                                    base = base.right;
                                    if (rankRng.isConst()) {
                                        index -= rankRng.start;
                                    } else {
                                        return ExpNum.index(
                                            base,
                                            ExpNum.bop(NumBopType.Sub, idx, firstRank, idx.source),
                                            exp.source
                                        );
                                    }
                                } else if (rankRng?.gte(index + 1)) {
                                    // take first
                                    base = base.left;
                                } else {
                                    doBreak = true;
                                }
                            }
                            break;
                        case ShapeOpType.Set:
                            {
                                const axisRng = ctrSet.getCachedRange(base.axis);
                                if (axisRng) {
                                    if (axisRng.isConst()) {
                                        if (axisRng.start === index) {
                                            return { ...base.dim, source: exp.source };
                                        } else {
                                            base = base.baseShape;
                                        }
                                    } else if (axisRng.contains(index)) {
                                        doBreak = true;
                                    } else {
                                        base = base.baseShape;
                                    }
                                }
                                doBreak = true;
                            }
                            break;

                        case ShapeOpType.Slice:
                        case ShapeOpType.Broadcast:
                        case ShapeOpType.Symbol:
                            // TODO: implement it.
                            doBreak = true;
                            break;
                    }
                }
            }
            return ExpNum.index(base, idx, exp.source);
        }
        case NumOpType.Max: {
            const values = exp.values.map((v) => simplifyNum(ctrSet, v));
            if (values.length > 0 && values.every((v) => v.opType === NumOpType.Const)) {
                let maxVal = (values[0] as ExpNumConst).value;
                values.forEach((v) => {
                    const currVal = (v as ExpNumConst).value;
                    if (maxVal < currVal) maxVal = currVal;
                });
                return ExpNum.fromConst(maxVal, exp.source);
            }
            return ExpNum.max(values, exp.source);
        }
        case NumOpType.Min: {
            const values = exp.values.map((v) => simplifyNum(ctrSet, v));
            if (values.length > 0 && values.every((v) => v.opType === NumOpType.Const)) {
                let minVal = (values[0] as ExpNumConst).value;
                values.forEach((v) => {
                    const currVal = (v as ExpNumConst).value;
                    if (minVal > currVal) minVal = currVal;
                });
                return ExpNum.fromConst(minVal, exp.source);
            }
            return ExpNum.min(values, exp.source);
        }
        case NumOpType.Numel: {
            const base = simplifyShape(ctrSet, exp.shape);
            if (base.opType === ShapeOpType.Const) {
                let numel: ExpNum | undefined;
                base.dims.forEach((v) => {
                    if (numel) {
                        numel = ExpNum.bop(NumBopType.Mul, numel, v, exp.source);
                    } else {
                        numel = v;
                    }
                });

                // TODO: potential inifinity loop?
                if (numel) {
                    return simplifyNum(ctrSet, numel);
                } else {
                    return ExpNum.fromConst(0, exp.source);
                }
            } else if (base.opType === ShapeOpType.Concat) {
                return simplifyNum(
                    ctrSet,
                    ExpNum.bop(
                        NumBopType.Add,
                        ExpNum.numel(base.left, exp.source),
                        ExpNum.numel(base.right, exp.source),
                        exp.source
                    )
                );
            }

            return ExpNum.numel(base, exp.source);
        }
        case NumOpType.Symbol: {
            const rng = ctrSet.getSymbolRange(exp.symbol);
            if (rng && rng.isConst()) {
                return ExpNum.fromConst(rng.start, exp.source);
            }
            return exp;
        }
    }
}

export function simplifyShape(ctrSet: ConstraintSet, exp: ExpShape): ExpShape {
    switch (exp.opType) {
        case ShapeOpType.Broadcast: {
            // TODO: implement it.
            const left = simplifyShape(ctrSet, exp.left);
            const right = simplifyShape(ctrSet, exp.right);

            if (left.opType === ShapeOpType.Const && right.opType === ShapeOpType.Const) {
                // baseShape has larger rank.
                const [baseShape, rightShape] = left.rank < right.rank ? [right, left] : [left, right];

                const dims: ExpNum[] = [];
                let isSimple = true;
                const rankdiff = baseShape.dims.length - rightShape.dims.length;
                for (let i = 0; i < baseShape.dims.length; i++) {
                    if (i < rankdiff) {
                        dims.push(baseShape.dims[i]);
                        continue;
                    }
                    const dim = ctrSet.selectBroadcastable(baseShape.dims[i], rightShape.dims[i - rankdiff]);
                    if (!dim) {
                        isSimple = false;
                        break;
                    }
                    dims.push(dim);
                }

                if (isSimple) {
                    return ExpShape.fromConst(baseShape.rank, dims, exp.source);
                }
            }
            return ExpShape.broadcast(left, right, exp.source);
        }
        case ShapeOpType.Concat: {
            const left = simplifyShape(ctrSet, exp.left);
            const right = simplifyShape(ctrSet, exp.right);
            if (left.opType === ShapeOpType.Const && right.opType === ShapeOpType.Const) {
                return ExpShape.fromConst(left.rank + right.rank, [...left.dims, ...right.dims], exp.source);
            }
            return ExpShape.concat(left, right, exp.source);
        }
        case ShapeOpType.Set: {
            const base = simplifyShape(ctrSet, exp.baseShape);
            const axis = simplifyNum(ctrSet, exp.axis);
            const dim = simplifyNum(ctrSet, exp.dim);

            if (base.opType === ShapeOpType.Const && axis.opType === NumOpType.Const) {
                const idx = axis.value;
                if (0 <= idx && idx < base.rank) {
                    const newDims = [...base.dims];
                    newDims[idx] = dim;
                    return ExpShape.fromConst(base.rank, newDims, exp.source);
                }
            }

            return ExpShape.setDim(base, axis, dim, exp.source);
        }

        case ShapeOpType.Slice: {
            const base = simplifyShape(ctrSet, exp.baseShape);
            const start = exp.start !== undefined ? simplifyNum(ctrSet, exp.start) : undefined;
            const end = exp.end !== undefined ? simplifyNum(ctrSet, exp.end) : undefined;

            if (base.opType === ShapeOpType.Const) {
                const startPos = start === undefined ? 0 : start.opType === NumOpType.Const ? start.value : -1;
                const endPos = end === undefined ? base.rank : end.opType === NumOpType.Const ? end.value : -1;
                if (
                    startPos >= 0 &&
                    endPos >= 0 &&
                    Math.floor(startPos) === startPos &&
                    Math.floor(endPos) === endPos
                ) {
                    const newDims = base.dims.slice(startPos, endPos);
                    return ExpShape.fromConst(newDims.length, newDims, exp.source);
                }
            }

            return ExpShape.slice(base, start, end, exp.source);
        }
        case ShapeOpType.Const: {
            const dims = exp.dims.map((num) => simplifyNum(ctrSet, num));
            return ExpShape.fromConst(exp.rank, dims, exp.source);
        }

        case ShapeOpType.Symbol: {
            const dims = ctrSet.getCachedShape(exp);
            if (dims) return ExpShape.fromConst(dims.length, dims, exp.source);
            return exp;
        }
    }
}

export function simplifyExp(ctrSet: ConstraintSet, exp: SymExp): SymExp {
    switch (exp.expType) {
        case SEType.String:
            return simplifyString(ctrSet, exp);
        case SEType.Bool:
            return simplifyBool(ctrSet, exp);
        case SEType.Num:
            return simplifyNum(ctrSet, exp);
        case SEType.Shape:
            return simplifyShape(ctrSet, exp);
    }
}

export function simplifyConstraint(ctrSet: ConstraintSet, ctr: Constraint): Constraint {
    switch (ctr.type) {
        case ConstraintType.And:
        case ConstraintType.Or:
            return { ...ctr, left: simplifyConstraint(ctrSet, ctr.left), right: simplifyConstraint(ctrSet, ctr.right) };
        case ConstraintType.Equal:
        case ConstraintType.NotEqual:
            return { ...ctr, left: simplifyExp(ctrSet, ctr.left), right: simplifyExp(ctrSet, ctr.right) };
        case ConstraintType.LessThan:
        case ConstraintType.LessThanOrEqual:
            return { ...ctr, left: simplifyNum(ctrSet, ctr.left), right: simplifyNum(ctrSet, ctr.right) };
        case ConstraintType.ExpBool:
            return { ...ctr, exp: simplifyBool(ctrSet, ctr.exp) };
        case ConstraintType.Broadcastable:
            return { ...ctr, left: simplifyShape(ctrSet, ctr.left), right: simplifyShape(ctrSet, ctr.right) };
        case ConstraintType.Forall:
            return {
                ...ctr,
                range: [simplifyNum(ctrSet, ctr.range[0]), simplifyNum(ctrSet, ctr.range[1])],
                constraint: simplifyConstraint(ctrSet, ctr),
            };
        case ConstraintType.Not:
            return { ...ctr, constraint: simplifyConstraint(ctrSet, ctr.constraint) };
        default:
            return ctr;
    }
}

export function ceilDiv(left: ExpNum | number, right: ExpNum | number, source: CodeSource | undefined): ExpNumBop {
    if (typeof left === 'number') {
        left = ExpNum.fromConst(left, source);
    }
    if (typeof right === 'number') {
        right = ExpNum.fromConst(right, source);
    }

    const exp1 = ExpNum.bop(NumBopType.Sub, right, 1, source);
    const exp2 = ExpNum.bop(NumBopType.Add, left, exp1, source);
    return ExpNum.bop(NumBopType.FloorDiv, exp2, right, source);
}

// if exp can be recognized as integer structually, return true
export function isStructuallyInt(exp: ExpNum): boolean {
    switch (exp.opType) {
        case NumOpType.Bop:
            if (exp.bopType === NumBopType.TrueDiv) return false;
            return isStructuallyInt(exp.left) && isStructuallyInt(exp.right);
        case NumOpType.Const:
            return Number.isInteger(exp.value);
        case NumOpType.Max:
        case NumOpType.Min:
            return exp.values.every((e) => isStructuallyInt(e));
        case NumOpType.Symbol:
            return exp.symbol.type === SymbolType.Int;
        case NumOpType.Numel:
        case NumOpType.Index:
            return true;
        case NumOpType.Uop:
            if (exp.uopType === NumUopType.Floor || exp.uopType === NumUopType.Ceil) return true;
            return isStructuallyInt(exp.baseValue);
    }
}

// simple structural equality checker.
export function isStructuallyEq(left?: SymExp, right?: SymExp): boolean {
    if (!left) return !right;
    if (!right) return false;

    if (left.expType !== right.expType || left.opType !== right.opType) return false;

    switch (left.expType) {
        case SEType.Bool:
            switch (left.opType) {
                case BoolOpType.And: {
                    const re = right as typeof left;
                    return isStructuallyEq(left.left, re.left) && isStructuallyEq(left.right, re.right);
                }
                case BoolOpType.Const: {
                    const re = right as typeof left;
                    return left.value === re.value;
                }
                case BoolOpType.Equal: {
                    const re = right as typeof left;
                    return isStructuallyEq(left.left, re.left) && isStructuallyEq(left.right, re.right);
                }
                case BoolOpType.LessThan: {
                    const re = right as typeof left;
                    return isStructuallyEq(left.left, re.left) && isStructuallyEq(left.right, re.right);
                }
                case BoolOpType.LessThanOrEqual: {
                    const re = right as typeof left;
                    return isStructuallyEq(left.left, re.left) && isStructuallyEq(left.right, re.right);
                }
                case BoolOpType.Not: {
                    const re = right as typeof left;
                    return isStructuallyEq(left.baseBool, re.baseBool);
                }
                case BoolOpType.NotEqual: {
                    const re = right as typeof left;
                    return isStructuallyEq(left.left, re.left) && isStructuallyEq(left.right, re.right);
                }
                case BoolOpType.Or: {
                    const re = right as typeof left;
                    return isStructuallyEq(left.left, re.left) && isStructuallyEq(left.right, re.right);
                }
                case BoolOpType.Symbol: {
                    const re = right as typeof left;
                    return left.symbol.id === re.symbol.id;
                }
            }
            break;
        case SEType.Num:
            switch (left.opType) {
                case NumOpType.Uop: {
                    const re = right as typeof left;
                    return left.uopType === re.uopType && isStructuallyEq(left.baseValue, re.baseValue);
                }
                case NumOpType.Bop: {
                    const re = right as typeof left;
                    return (
                        left.bopType === re.bopType &&
                        isStructuallyEq(left.left, re.left) &&
                        isStructuallyEq(left.right, re.right)
                    );
                }
                case NumOpType.Const: {
                    const re = right as typeof left;
                    return left.value === re.value;
                }
                case NumOpType.Index: {
                    const re = right as typeof left;
                    return isStructuallyEq(left.index, re.index) && isStructuallyEq(left.baseShape, re.baseShape);
                }
                case NumOpType.Max:
                case NumOpType.Min: {
                    const re = right as typeof left;
                    return (
                        left.opType === right.opType &&
                        left.values.length === re.values.length &&
                        left.values.every((v, i) => isStructuallyEq(v, re.values[i]))
                    );
                }
                case NumOpType.Numel: {
                    const re = right as typeof left;
                    return isStructuallyEq(left.shape, re.shape);
                }
                case NumOpType.Symbol: {
                    const re = right as typeof left;
                    return left.symbol.id === re.symbol.id;
                }
            }
            break;
        case SEType.Shape:
            switch (left.opType) {
                case ShapeOpType.Broadcast: {
                    const re = right as typeof left;
                    return isStructuallyEq(left.left, re.left) && isStructuallyEq(left.right, re.right);
                }
                case ShapeOpType.Concat: {
                    const re = right as typeof left;
                    return isStructuallyEq(left.left, re.left) && isStructuallyEq(left.right, re.right);
                }
                case ShapeOpType.Const: {
                    const re = right as typeof left;
                    return (
                        left.rank === re.rank &&
                        left.dims.length === re.dims.length &&
                        left.dims.every((v, i) => isStructuallyEq(v, re.dims[i]))
                    );
                }
                case ShapeOpType.Set: {
                    const re = right as typeof left;
                    return (
                        isStructuallyEq(left.axis, re.axis) &&
                        isStructuallyEq(left.dim, re.dim) &&
                        isStructuallyEq(left.baseShape, re.baseShape)
                    );
                }
                case ShapeOpType.Slice: {
                    const re = right as typeof left;
                    return (
                        isStructuallyEq(left.start, re.start) &&
                        isStructuallyEq(left.end, re.end) &&
                        isStructuallyEq(left.baseShape, re.baseShape)
                    );
                }
                case ShapeOpType.Symbol: {
                    const re = right as typeof left;
                    return left.symbol.id === re.symbol.id;
                }
            }
            break;
        case SEType.String:
            switch (left.opType) {
                case StringOpType.Concat: {
                    const re = right as typeof left;
                    return isStructuallyEq(left.left, re.left) && isStructuallyEq(left.right, re.right);
                }
                case StringOpType.Const: {
                    const re = right as typeof left;
                    return left.value === re.value;
                }
                case StringOpType.Slice: {
                    const re = right as typeof left;
                    return (
                        isStructuallyEq(left.start, re.start) &&
                        isStructuallyEq(left.end, re.end) &&
                        isStructuallyEq(left.baseString, re.baseString)
                    );
                }
                case StringOpType.Symbol: {
                    const re = right as typeof left;
                    return left.symbol.id === re.symbol.id;
                }
            }
    }
}

// return symbolic length of str
export function strLen(
    ctx: Context<any>,
    str: string | ExpString,
    source: CodeSource | undefined
): number | ExpNum | Context<ExpNum> {
    if (typeof str === 'string') {
        return str.length;
    }

    switch (str.opType) {
        case StringOpType.Const:
            return str.value.length;
        case StringOpType.Concat: {
            let leftIsCtx = false;
            let left = strLen(ctx, str.left, source);
            if (left instanceof Context) {
                ctx = left;
                left = left.retVal;
                leftIsCtx = true;
            }
            const right = strLen(ctx, str.right, source);
            if (right instanceof Context) {
                const exp = simplifyNum(right.ctrSet, ExpNum.bop(NumBopType.Add, left, right.retVal, source));
                return right.setRetVal(exp);
            } else {
                const exp = simplifyNum(ctx.ctrSet, ExpNum.bop(NumBopType.Add, left, right, source));
                return leftIsCtx ? ctx.setRetVal(exp) : exp;
            }
        }
        case StringOpType.Slice:
            if (str.end === undefined) {
                const baseLen = strLen(ctx, str.baseString, source);
                if (baseLen instanceof Context) {
                    return baseLen.setRetVal(ExpNum.bop(NumBopType.Sub, baseLen.retVal, str.start ?? 0, source));
                }
                return simplifyNum(ctx.ctrSet, ExpNum.bop(NumBopType.Sub, baseLen, str.start ?? 0, source));
            }
            return simplifyNum(ctx.ctrSet, ExpNum.bop(NumBopType.Sub, str.end, str.start ?? 0, source));
        case StringOpType.Symbol:
            // TODO: cache length of symbolic string
            return ctx.genIntGte(`${str.symbol.name}_${str.symbol.id}_len`, 0, source);
    }
}

// return closed expression of normalized index of list. (e.g. a[-1] with length 5 => 4)
// if it cannot determine that index is neg or pos,
// return i + r * (1 - (1 // (abs(i) - i + 1))) for safety
// this is equivalent to i if i >= 0 else r - i
export function absExpIndexByLen(
    index: number | ExpNum,
    rank: number | ExpNum,
    source: CodeSource | undefined,
    ctrSet?: ConstraintSet
): number | ExpNum {
    if (typeof index === 'number') {
        if (index >= 0) return index;
        else if (typeof rank === 'number') return rank + index;

        const r = ctrSet ? simplifyNum(ctrSet, rank) : rank;
        if (r.opType === NumOpType.Const) return r.value + index;
        return ExpNum.bop(NumBopType.Add, r, index, source);
    }

    const i = ctrSet ? simplifyNum(ctrSet, index) : index;
    if (i.opType === NumOpType.Const) {
        index = i.value;
        if (index >= 0) return index;
        else if (typeof rank === 'number') return rank + index;

        const r = ctrSet ? simplifyNum(ctrSet, rank) : rank;
        if (r.opType === NumOpType.Const) return r.value + index;
        return ExpNum.bop(NumBopType.Add, r, index, source);
    } else if (ctrSet) {
        const idxRange = ctrSet.getCachedRange(i);
        if (idxRange?.gte(0)) {
            return i;
        }
    }

    // don't know sign of i
    // return i + r * (1 - (1 // (abs(i) - i + 1)))
    const r = typeof rank === 'number' ? ExpNum.fromConst(rank, source) : ctrSet ? simplifyNum(ctrSet, rank) : rank;
    const result = ExpNum.bop(
        NumBopType.Add,
        i,
        ExpNum.bop(
            NumBopType.Mul,
            r,
            ExpNum.bop(
                NumBopType.Sub,
                1,
                ExpNum.bop(
                    NumBopType.FloorDiv,
                    1,
                    ExpNum.bop(
                        NumBopType.Sub,
                        ExpNum.bop(NumBopType.Sub, ExpNum.uop(NumUopType.Abs, i, source), i, source),
                        1,
                        source
                    ),
                    source
                ),
                source
            ),

            source
        ),
        source
    );
    return result;
}

// return closed expression of length of a:b
// return b >= a ? b - a : 0
// that is, ((b - a) + |b - a|) // 2
export function reluLen(
    a: number | ExpNum,
    b: number | ExpNum,
    source: CodeSource | undefined,
    ctrSet?: ConstraintSet
): number | ExpNum {
    if (typeof a === 'number' && typeof b === 'number') return b - a >= 0 ? b - a : 0;
    const bma = ExpNum.bop(NumBopType.Sub, b, a, source);
    const result = ExpNum.bop(
        NumBopType.FloorDiv,
        ExpNum.bop(NumBopType.Add, bma, ExpNum.uop(NumUopType.Abs, bma, source), source),
        2,
        source
    );
    return ctrSet ? simplifyNum(ctrSet, result) : result;
}

// return undefined if inputVal points undefined value. else, return isinstance(inputVal, classVal)
export function isInstanceOf(inputVal: ShValue, classVal: ShValue, env: ShEnv, heap: ShHeap): boolean | undefined {
    const input = fetchAddr(inputVal, heap);
    if (!input) return;

    const mroList = trackMro(inputVal, heap, env);

    if (classVal.type !== SVType.Addr) {
        // direct comparison between int / float type function
        return (
            mroList.findIndex((v) => {
                if (v === undefined) return false;
                return fetchAddr(heap.getVal(v), heap) === classVal;
            }) >= 0
        );
    }

    let classPoint: SVAddr = classVal;
    while (true) {
        const next = heap.getVal(classPoint);
        if (next?.type !== SVType.Addr) {
            break;
        }
        classPoint = next;
    }

    return mroList.findIndex((v) => v === classPoint.addr) >= 0;
}
