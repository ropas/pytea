/*
 * constraintSolver.ts
 * Copyright (c) Seoul National University.
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo (mersshs@gmail.com)
 *
 * Analyze Symbolic Constraint and add it to ConstraintSet
 */

import { PyteaService } from '../service/pyteaService';
import { ConstraintSet } from './constraintSet';
import { Constraint, ConstraintType } from './constraintType';
import { CoExpNum, NormalExp, normalizeExpNum, simplifyConstraint } from './expUtils';
import { Fraction } from './fraction';
import { NumRange } from './range';
import {
    BoolOpType,
    ExpBool,
    ExpNum,
    ExpNumSymbol,
    NumBopType,
    NumOpType,
    NumUopType,
    SEType,
    SymbolType,
} from './symExpressions';

enum SolveType {
    LT,
    LTE,
    GT,
    GTE,
    EQ,
    NEQ,
}

// reverse type by negation
function flipSolveType(type: SolveType) {
    switch (type) {
        case SolveType.LT:
            return SolveType.GT;
        case SolveType.LTE:
            return SolveType.GTE;
        case SolveType.GT:
            return SolveType.LT;
        case SolveType.GTE:
            return SolveType.LTE;
        default:
            return type;
    }
}

interface SolveState {
    // should this state add to constraint set?
    addable: boolean;
    // is this state finally resolved?
    // if _run is finished even this is not set, then it means solver is timeouted.
    finished: boolean;
}

abstract class SubSolver {
    ctrSet: ConstraintSet;
    state?: SolveState;

    constructor(ctrSet: ConstraintSet) {
        this.ctrSet = ctrSet;
    }

    // initial constraint should be simplified by simplifyConstraint
    abstract initState(constraint: Constraint): void;
    // return false if finished
    abstract step(): boolean;
    // update ctrSet based on State
    abstract resolveState(): void;

    solve(): void {
        let stepCount = 0;
        // TODO: move setting to explicit maximal step count
        while (this.state?.finished === false && stepCount < 100) {
            if (!this.step()) {
                break;
            }
            stepCount++;
        }

        if (stepCount >= 100) {
            PyteaService.log('solver step count exceeded maximal loop limit');
        }

        const state = this.state;
        if (state?.finished && state?.addable) {
            this.resolveState();
        }
    }
}

export class ConstraintSolver {
    public ctrSet: ConstraintSet;
    private solver?: SubSolver;

    constructor(ctrSet: ConstraintSet) {
        this.ctrSet = ctrSet;
        this.solver = undefined;
    }

    // resolve constraint and save cached ranges to ctrSet
    // constraint should be filtered by filtering functions below.
    // this should accept only equal / notequla / lt / lte / forall / broadcastable.
    // return this for chaining.
    solve(constraint: Constraint): ConstraintSolver {
        // simplifier automatically resolve constant variable
        //   => filter secondary symbolic variable while subsolver is running
        const filtered = this._destructConstraint(simplifyConstraint(this.ctrSet, constraint));
        // .filter((ctr) => this.ctrSet.hasSingleVar(ctr) !== false);

        for (const ctr of filtered) {
            this._setState(ctr);
            if (this.solver) {
                if (this.solver.state === undefined) {
                    continue;
                }

                this.solver.solve();
                this.ctrSet = this.solver.ctrSet;
            }
        }

        return this;
    }

    solveAll(constraints: Constraint[]): ConstraintSolver {
        constraints.forEach((ctr) => this.solve(ctr));
        return this;
    }

    private _setState(constraint: Constraint): void {
        // TODO: implement it.
        this.solver = undefined;
        switch (constraint.type) {
            case ConstraintType.LessThan:
            case ConstraintType.LessThanOrEqual:
                this.solver = new NumSubSolver(this.ctrSet);
                break;
            case ConstraintType.Equal:
            case ConstraintType.NotEqual:
                {
                    switch (constraint.left.expType) {
                        case SEType.Num:
                            this.solver = new NumSubSolver(this.ctrSet);
                            break;
                        case SEType.Shape:
                            this.solver = new ShapeSubSolver(this.ctrSet);
                            break;
                        case SEType.String:
                        case SEType.Bool:
                            break;
                    }
                }
                break;
            case ConstraintType.Forall:
            case ConstraintType.Broadcastable:
                this.solver = new ShapeSubSolver(this.ctrSet);
                break;
            default:
                break;
        }

        if (this.solver) {
            this.solver.initState(constraint);
        }
    }

    // destruct and filter primitive (string-number equality / comparison) constraints from single constraint
    // all not, and, or, constraints are normalized to primitives
    private _destructConstraint(constraint: Constraint): Constraint[] {
        switch (constraint.type) {
            case ConstraintType.ExpBool:
                return this._destructConstraint(expToCtr(constraint.exp));
            case ConstraintType.Equal:
            case ConstraintType.NotEqual:
                return constraint.left.expType === constraint.left.expType ? [constraint] : [];
            case ConstraintType.Not: {
                const negated = evaluateNot(constraint);
                if (negated.type === ConstraintType.Not) return [];
                return this._destructConstraint(negated);
            }
            case ConstraintType.And:
                return [...this._destructConstraint(constraint.left), ...this._destructConstraint(constraint.right)];
            case ConstraintType.Or: {
                const leftVal = this.ctrSet.checkImmediate(constraint.left);
                if (leftVal === true) {
                    return [];
                } else if (leftVal === false) {
                    return [constraint.right];
                }

                const rightVal = this.ctrSet.checkImmediate(constraint.right);
                if (rightVal === false) {
                    return [constraint.left];
                }

                return [];
            }
            case ConstraintType.LessThan:
            case ConstraintType.LessThanOrEqual:
            case ConstraintType.Forall:
            case ConstraintType.Broadcastable:
                return [constraint];
            default:
                return [];
        }
    }
}

interface NumSolveState extends SolveState {
    // which holds symbolic variable
    left: NormalExp;
    // constant value
    right: Fraction;
    // main symbol with coefficient
    symCexp?: CoExpNum<ExpNumSymbol>;
    // main compare type
    type: SolveType;
}

export class NumSubSolver extends SubSolver {
    state?: NumSolveState;

    // constraint should be simplified.
    initState(constraint: Constraint): void {
        this.state = undefined;
        switch (constraint.type) {
            case ConstraintType.LessThan:
            case ConstraintType.LessThanOrEqual:
                this.state = {
                    left: normalizeExpNum(
                        ExpNum.bop(NumBopType.Sub, constraint.left, constraint.right, constraint.source)
                    ),
                    right: new Fraction(0, 1),
                    symCexp: undefined,
                    type: constraint.type === ConstraintType.LessThan ? SolveType.LT : SolveType.LTE,
                    addable: true,
                    finished: false,
                };
                break;
            case ConstraintType.Equal:
            case ConstraintType.NotEqual:
                if (constraint.left.expType === SEType.Num && constraint.right.expType === SEType.Num) {
                    this.state = {
                        left: normalizeExpNum(
                            ExpNum.bop(NumBopType.Sub, constraint.left, constraint.right, constraint.source)
                        ),
                        right: new Fraction(0, 1),
                        symCexp: undefined,
                        type: constraint.type === ConstraintType.Equal ? SolveType.EQ : SolveType.NEQ,
                        addable: true,
                        finished: false,
                    };
                }
        }

        if (!this.state) {
            return;
        }
    }

    step(): boolean {
        if (!this.state) {
            return false;
        }

        const state = this.state;
        const { list, constant } = this.state.left;

        if (constant.up !== 0) {
            state.right = state.right.sub(constant).norm();
            state.left.constant = new Fraction(0, 1);
        }

        if (list.length === 0) {
            state.finished = true;
            return false;
        }

        const nonVar: CoExpNum<ExpNum>[] = [];
        for (const cexp of list) {
            if (cexp.exp.opType === NumOpType.Symbol) {
                if (state.symCexp) {
                    if (state.symCexp.exp.symbol.id !== cexp.exp.symbol.id) {
                        // TODO: resolve multi variables
                        state.addable = false;
                        state.finished = true;
                        return false;
                    } else {
                        state.symCexp.coeff = state.symCexp.coeff.add(cexp.coeff).norm();
                    }
                } else {
                    state.symCexp = { exp: cexp.exp, coeff: cexp.coeff };
                }
            } else {
                nonVar.push(cexp);
            }
        }

        if (nonVar.length === 0) {
            state.finished = true;
            return false;
        } else if (nonVar.length === 2) {
            state.addable = false;
            state.finished = true;
            return false;
        }

        const remain = nonVar[0];
        switch (remain.exp.opType) {
            case NumOpType.Const:
                // should never happen
                state.right = state.right.sub(remain.coeff.mulN(remain.exp.value));
            // eslint-disable-next-line no-fallthrough
            case NumOpType.Symbol:
                // never happen
                state.finished = true;
                return false;
            case NumOpType.Uop:
                // TODO: implement uop
                switch (remain.exp.uopType) {
                    case NumUopType.Abs:
                    case NumUopType.Ceil:
                    case NumUopType.Floor:
                    case NumUopType.Neg:
                        state.addable = false;
                        state.finished = true;
                        return false;
                }
                break;
            case NumOpType.Bop:
                switch (remain.exp.bopType) {
                    // TODO: implement floordiv, truediv / mod
                    case NumBopType.FloorDiv:
                    case NumBopType.Mod:
                    case NumBopType.TrueDiv:
                    case NumBopType.Add:
                    case NumBopType.Sub:
                    case NumBopType.Mul:
                        // remained after normalized: this means the formula is non-linear
                        state.addable = false;
                        state.finished = true;
                        return false;
                }
                break;
            case NumOpType.Index:
            case NumOpType.Max:
            case NumOpType.Min:
            case NumOpType.Numel:
                // TODO: cache somewhere
                state.addable = false;
                state.finished = true;
                return false;
        }

        return true;
    }

    // update ctrSet based on state
    resolveState(): void {
        if (this.state?.addable !== true || this.state.symCexp === undefined) {
            return;
        }

        const symbol = this.state.symCexp.exp.symbol;
        const coeff = this.state.symCexp.coeff;
        let right = this.state.right;
        let type = this.state.type;

        if (coeff.up * coeff.down < 0) {
            right = right.div(coeff);
            type = flipSolveType(type);
        } else if (coeff.up === 0) {
            return;
        } else {
            right = right.div(coeff);
        }

        let range = this.ctrSet.getSymbolRange(symbol);
        let subRange: NumRange;
        switch (type) {
            case SolveType.LT:
                subRange = NumRange.genLt(right.toNum());
                break;
            case SolveType.LTE:
                subRange = NumRange.genLte(right.toNum());
                break;
            case SolveType.GT:
                subRange = NumRange.genGt(right.toNum());
                break;
            case SolveType.GTE:
                subRange = NumRange.genGte(right.toNum());
                break;
            case SolveType.EQ:
                subRange = NumRange.fromConst(right.toNum());
                break;
            case SolveType.NEQ:
                // solved below
                subRange = NumRange.fromConst(right.toNum());
                break;
        }

        if (type !== SolveType.NEQ) {
            range = range ? range.intersect(subRange) : subRange;

            if (symbol.type === SymbolType.Int) {
                range = range.toIntRange();
            }
            if (range) {
                const rangeCache = this.ctrSet.rangeCache;
                this.ctrSet = this.ctrSet.set('rangeCache', rangeCache.set(symbol.id, range));
            }
        } else if (range) {
            const num = subRange.end;
            let modified = false;

            if (range.end === num && range.hasEnd) {
                modified = true;
                range = new NumRange(range.start, range.end, range.hasStart, false);
            } else if (range.start === num && range.hasStart) {
                modified = true;
                range = new NumRange(range.start, range.end, false, range.hasEnd);
            }

            if (modified) {
                if (symbol.type === SymbolType.Int) {
                    range = range.toIntRange();
                }
                if (range) {
                    const rangeCache = this.ctrSet.rangeCache;
                    this.ctrSet = this.ctrSet.set('rangeCache', rangeCache.set(symbol.id, range));
                }
            }
        }
    }
}

export class ShapeSubSolver extends SubSolver {
    state?: NumSolveState;

    initState(constraint: Constraint): void {
        // TODO:
    }

    step(): boolean {
        return false;
    }

    resolveState(): void {
        // TODO:
    }
}

// -- UTILITY FUNCTIONS --

// deeply evaluate negation of constraint.
export function evaluateNot(constraint: Constraint): Constraint {
    switch (constraint.type) {
        case ConstraintType.ExpBool:
            return evaluateNot(expToCtr(constraint.exp));
        case ConstraintType.Equal:
            return {
                ...constraint,
                type: ConstraintType.NotEqual,
            };
        case ConstraintType.NotEqual:
            return {
                ...constraint,
                type: ConstraintType.Equal,
            };
        case ConstraintType.And:
            return {
                ...constraint,
                left: evaluateNot(constraint.left),
                right: evaluateNot(constraint.right),
                type: ConstraintType.Or,
            };
        case ConstraintType.Or:
            return {
                ...constraint,
                left: evaluateNot(constraint.left),
                right: evaluateNot(constraint.right),
                type: ConstraintType.And,
            };
        case ConstraintType.Not:
            return constraint.constraint;
        case ConstraintType.LessThan:
            return {
                ...constraint,
                left: constraint.right,
                right: constraint.left,
                type: ConstraintType.LessThanOrEqual,
            };
        case ConstraintType.LessThanOrEqual:
            return {
                ...constraint,
                left: constraint.right,
                right: constraint.left,
                type: ConstraintType.LessThan,
            };
        case ConstraintType.Forall:
            return {
                id: -1,
                constraint: constraint,
                type: ConstraintType.Not,
                source: constraint.source,
            };
        case ConstraintType.Broadcastable:
            return {
                id: -1,
                constraint: constraint,
                type: ConstraintType.Not,
                source: constraint.source,
            };
        case ConstraintType.Fail:
            return expToCtr(ExpBool.fromConst(true, constraint.source));
    }
}

export function expToCtr(exp: ExpBool): Constraint {
    switch (exp.opType) {
        case BoolOpType.Symbol:
        case BoolOpType.Const:
            return {
                type: ConstraintType.Equal,
                id: -1,
                left: exp,
                right: ExpBool.fromConst(true, exp.source),
                source: exp.source,
            };
        case BoolOpType.Equal:
            return {
                type: ConstraintType.Equal,
                id: -1,
                left: exp.left,
                right: exp.right,
                source: exp.source,
            };
        case BoolOpType.NotEqual:
            return {
                type: ConstraintType.NotEqual,
                id: -1,
                left: exp.left,
                right: exp.right,
                source: exp.source,
            };
        case BoolOpType.LessThan:
            return {
                type: ConstraintType.LessThan,
                id: -1,
                left: exp.left,
                right: exp.right,
                source: exp.source,
            };
        case BoolOpType.LessThanOrEqual:
            return {
                type: ConstraintType.LessThanOrEqual,
                id: -1,
                left: exp.left,
                right: exp.right,
                source: exp.source,
            };
        case BoolOpType.Not:
            return {
                type: ConstraintType.Not,
                id: -1,
                constraint: expToCtr(exp.baseBool),
                source: exp.source,
            };
        case BoolOpType.And:
            return {
                type: ConstraintType.And,
                id: -1,
                left: expToCtr(exp.left),
                right: expToCtr(exp.right),
                source: exp.source,
            };
        case BoolOpType.Or:
            return {
                type: ConstraintType.Or,
                id: -1,
                left: expToCtr(exp.left),
                right: expToCtr(exp.right),
                source: exp.source,
            };
    }
}
