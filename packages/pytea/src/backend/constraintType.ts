/*
 * constraint.ts
 * Copyright (c) Seoul National University .
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo (mersshs@gmail.com)
 *
 * Definitions and types of constraints.
 */

import { CodeSource } from './sharpValues';
import { ExpBool, ExpNum, ExpShape, SymExp, SymInt } from './symExpressions';

export const enum ConstraintType {
    ExpBool,
    Equal,
    NotEqual,
    And,
    Or,
    Not,
    LessThan,
    LessThanOrEqual,
    Forall,
    Broadcastable,
    Fail,
}

export type Constraint =
    | CtrExpBool
    | CtrEq
    | CtrNeq
    | CtrAnd
    | CtrOr
    | CtrNot
    | CtrLt
    | CtrLte
    | CtrForall
    | CtrBroad
    | CtrFail;

export type ConstraintIndex = number;

export interface ConstraintBase {
    type: ConstraintType;
    id: ConstraintIndex;
    source: CodeSource | undefined;
    message?: string;
}

export interface CtrExpBool extends ConstraintBase {
    type: ConstraintType.ExpBool;
    exp: ExpBool;
}

export interface CtrEq extends ConstraintBase {
    type: ConstraintType.Equal;
    left: SymExp;
    right: SymExp;
}

export interface CtrNeq extends ConstraintBase {
    type: ConstraintType.NotEqual;
    left: SymExp;
    right: SymExp;
}

export interface CtrAnd extends ConstraintBase {
    type: ConstraintType.And;
    left: Constraint;
    right: Constraint;
}

export interface CtrOr extends ConstraintBase {
    type: ConstraintType.Or;
    left: Constraint;
    right: Constraint;
}

export interface CtrNot extends ConstraintBase {
    type: ConstraintType.Not;
    constraint: Constraint;
}

export interface CtrLt extends ConstraintBase {
    type: ConstraintType.LessThan;
    left: ExpNum;
    right: ExpNum;
}

export interface CtrLte extends ConstraintBase {
    type: ConstraintType.LessThanOrEqual;
    left: ExpNum;
    right: ExpNum;
}

export interface CtrForall extends ConstraintBase {
    type: ConstraintType.Forall;
    symbol: SymInt;
    range: [ExpNum, ExpNum];
    constraint: Constraint;
}

export interface CtrBroad extends ConstraintBase {
    type: ConstraintType.Broadcastable;
    left: ExpShape;
    right: ExpShape;
}

export interface CtrFail extends ConstraintBase {
    type: ConstraintType.Fail;
    reason: string;
}

export type CompareConstraintType =
    | ConstraintType.Equal
    | ConstraintType.NotEqual
    | ConstraintType.LessThan
    | ConstraintType.LessThanOrEqual;
export type BooleanConstraintType =
    | ConstraintType.And
    | ConstraintType.Or
    | ConstraintType.Not
    | ConstraintType.ExpBool;
export type EqualityConstraintType = ConstraintType.Equal | ConstraintType.NotEqual;

export type NumConstraint = CtrLt | CtrLte;
export type BoolConstraint = CtrNot | CtrAnd | CtrOr;
export type EqualityConstraint = CtrEq | CtrNeq;

export function ctrToStr(ctr: Constraint): string {
    let str: string;
    switch (ctr.type) {
        case ConstraintType.ExpBool:
            str = SymExp.toString(ctr.exp);
            break;
        case ConstraintType.Equal:
            str = `(${SymExp.toString(ctr.left)} == ${SymExp.toString(ctr.right)})`;
            break;
        case ConstraintType.NotEqual:
            str = `(${SymExp.toString(ctr.left)} != ${SymExp.toString(ctr.right)})`;
            break;
        case ConstraintType.And:
            str = `(${ctrToStr(ctr.left as Constraint)} && ${ctrToStr(ctr.right as Constraint)})`;
            break;
        case ConstraintType.Or:
            str = `(${ctrToStr(ctr.left as Constraint)} || ${ctrToStr(ctr.right as Constraint)})`;
            break;
        case ConstraintType.Not:
            str = `~(${ctrToStr(ctr.constraint as Constraint)})`;
            break;
        case ConstraintType.LessThan:
            str = `(${SymExp.toString(ctr.left)} < ${SymExp.toString(ctr.right)})`;
            break;
        case ConstraintType.LessThanOrEqual:
            str = `(${SymExp.toString(ctr.left)} <= ${SymExp.toString(ctr.right)})`;
            break;
        case ConstraintType.Forall:
            str = `forall[${ctr.symbol.name} in (${SymExp.toString(ctr.range[0])}:${SymExp.toString(
                ctr.range[1]
            )})](${ctrToStr(ctr.constraint)})`;
            break;
        case ConstraintType.Broadcastable:
            str = `broadcastable(${SymExp.toString(ctr.left)}, ${SymExp.toString(ctr.right)})`;
            break;
        case ConstraintType.Fail:
            str = `fail(${ctr.reason}`;
            break;
    }

    return str;
}

// return symbol indices used in constraints
export function extractSymbols(ctr: Constraint): number[] {
    const list: number[] = [];

    function append(exp: SymExp | number | string | boolean | undefined): void {
        const l = SymExp.extractSymbols(exp);
        for (const v of l) {
            list.push(v);
        }
    }

    function extract(ctr: Constraint | undefined): void {
        if (!ctr) return;
        switch (ctr.type) {
            case ConstraintType.ExpBool:
                append(ctr.exp);
                return;
            case ConstraintType.Equal:
            case ConstraintType.NotEqual:
            case ConstraintType.LessThan:
            case ConstraintType.LessThanOrEqual:
            case ConstraintType.Broadcastable:
                append(ctr.left);
                append(ctr.right);
                return;
            case ConstraintType.And:
            case ConstraintType.Or:
                extract(ctr.left);
                extract(ctr.right);
                return;
            case ConstraintType.Not:
                extract(ctr.constraint);
                return;
            case ConstraintType.Forall:
                extract(ctr.constraint);
                append(ctr.range[0]);
                append(ctr.range[1]);
                return;
            case ConstraintType.Fail:
                return;
        }
    }

    extract(ctr);

    return list;
}
