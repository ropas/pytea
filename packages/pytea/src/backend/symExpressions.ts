/*
 * symExpressions.ts
 * Copyright (c) Seoul National University
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo (mersshs@gmail.com)
 *
 * Definitions and types of symbolic variables and expressions
 */
import { ConstraintSet } from './constraintSet';
import { CodeSource } from './sharpValues';

/// Base types. import and use these types.

export type SymVal = SymInt | SymFloat | SymString | SymBool | SymShape;
export type SymNumeric = SymInt | SymFloat;
export type SymExp = ExpShape | ExpNum | ExpString | ExpBool;
export type ExpNum =
    | ExpNumConst
    | ExpNumConstBoxed
    | ExpNumSymbol
    | ExpNumBop
    | ExpNumIndex
    | ExpNumMax
    | ExpNumMin
    | ExpNumNumel
    | ExpNumUop;
export type ExpShape =
    | ExpShapeConst
    | ExpShapeSymbol
    | ExpShapeSet
    | ExpShapeSlice
    | ExpShapeConcat
    | ExpShapeBroadcast;
export type ExpString = ExpStringConst | ExpStringSymbol | ExpStringSlice | ExpStringConcat;
export type ExpBool =
    | ExpBoolConst
    | ExpBoolSymbol
    | ExpBoolEq
    | ExpBoolNeq
    | ExpBoolLt
    | ExpBoolLte
    | ExpBoolNot
    | ExpBoolAnd
    | ExpBoolOr;

/// SYMBOLS

export const enum SymbolType {
    Int,
    Float,
    String,
    Bool,
    Shape,
}

export type SymbolIndex = number;

export interface SymbolBase {
    readonly type: SymbolType;
    readonly id: SymbolIndex;
    readonly name: string;
    source: CodeSource | undefined;
}

export interface SymInt extends SymbolBase {
    readonly type: SymbolType.Int;
}

export interface SymFloat extends SymbolBase {
    readonly type: SymbolType.Float;
}

export interface SymString extends SymbolBase {
    readonly type: SymbolType.String;
}

export interface SymBool extends SymbolBase {
    readonly type: SymbolType.Bool;
}

export interface SymShape extends SymbolBase {
    readonly type: SymbolType.Shape;
    rank: ExpNum;
}

/// SYMBOLIC VALUES

export const enum SEType {
    Shape,
    Num,
    Bool,
    String,
}

export interface SymExpBase {
    expType: SEType;
    opType: ShapeOpType | NumOpType | StringOpType | BoolOpType;
    source: CodeSource | undefined;
}

//// BOOLEAN EXPRESSION

export const enum BoolOpType {
    Const,
    Symbol,
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    Not,
    And,
    Or,
}

export interface ExpBoolBase extends SymExpBase {
    expType: SEType.Bool;
    opType: BoolOpType;
}

export interface ExpBoolConst extends ExpBoolBase {
    opType: BoolOpType.Const;
    value: boolean;
}

export interface ExpBoolSymbol extends ExpBoolBase {
    opType: BoolOpType.Symbol;
    symbol: SymBool;
}

export interface ExpBoolEq extends ExpBoolBase {
    opType: BoolOpType.Equal;
    left: SymExp;
    right: SymExp;
}

export interface ExpBoolNeq extends ExpBoolBase {
    opType: BoolOpType.NotEqual;
    left: SymExp;
    right: SymExp;
}

export interface ExpBoolLt extends ExpBoolBase {
    opType: BoolOpType.LessThan;
    left: ExpNum;
    right: ExpNum;
}

export interface ExpBoolLte extends ExpBoolBase {
    opType: BoolOpType.LessThanOrEqual;
    left: ExpNum;
    right: ExpNum;
}

export interface ExpBoolNot extends ExpBoolBase {
    opType: BoolOpType.Not;
    baseBool: ExpBool;
}

export interface ExpBoolAnd extends ExpBoolBase {
    opType: BoolOpType.And;
    left: ExpBool;
    right: ExpBool;
}

export interface ExpBoolOr extends ExpBoolBase {
    opType: BoolOpType.Or;
    left: ExpBool;
    right: ExpBool;
}

//// NUMBER EXPRESSION

export const enum NumOpType {
    Const,
    Symbol,
    Bop,
    Index,
    Max,
    Numel,
    Uop,
    Min,
}

export const enum NumBopType {
    Add,
    Sub,
    Mul,
    TrueDiv,
    FloorDiv,
    Mod,
}

export const enum NumUopType {
    Neg,
    Floor,
    Ceil,
    Abs,
}

export interface ExpNumBase extends SymExpBase {
    expType: SEType.Num;
    opType: NumOpType;
}

export interface ExpNumConst extends ExpNumBase {
    opType: NumOpType.Const;
    value: number;
}

export interface ExpNumConstBoxed extends ExpNumBase {
    opType: NumOpType.Const;
    value: number;
    boxed: true;
}

export interface ExpNumSymbol extends ExpNumBase {
    opType: NumOpType.Symbol;
    symbol: SymNumeric;
}

export interface ExpNumBop extends ExpNumBase {
    opType: NumOpType.Bop;
    bopType: NumBopType;
    left: ExpNum;
    right: ExpNum;
}

// index is 0-based
export interface ExpNumIndex extends ExpNumBase {
    opType: NumOpType.Index;
    baseShape: ExpShape;
    index: ExpNum;
}

export interface ExpNumMax extends ExpNumBase {
    opType: NumOpType.Max;
    values: ExpNum[];
}

export interface ExpNumMin extends ExpNumBase {
    opType: NumOpType.Min;
    values: ExpNum[];
}

export interface ExpNumNumel extends ExpNumBase {
    opType: NumOpType.Numel;
    shape: ExpShape;
}

export interface ExpNumUop extends ExpNumBase {
    opType: NumOpType.Uop;
    uopType: NumUopType;
    baseValue: ExpNum;
}

//// SHAPE EXPRESSION

export const enum ShapeOpType {
    Const,
    Symbol,
    Set,
    Slice,
    Concat,
    Broadcast,
}

export interface ExpShapeBase extends SymExpBase {
    expType: SEType.Shape;
    opType: ShapeOpType;
}

export interface ExpShapeConst extends ExpShapeBase {
    opType: ShapeOpType.Const;
    rank: number;
    dims: ExpNum[];
}

export interface ExpShapeSymbol extends ExpShapeBase {
    opType: ShapeOpType.Symbol;
    symbol: SymShape;
}

export interface ExpShapeSet extends ExpShapeBase {
    opType: ShapeOpType.Set;
    baseShape: ExpShape;
    axis: ExpNum;
    dim: ExpNum;
}

// range is 0-based exclusive
export interface ExpShapeSlice extends ExpShapeBase {
    opType: ShapeOpType.Slice;
    baseShape: ExpShape;
    start?: ExpNum;
    end?: ExpNum;
}

export interface ExpShapeConcat extends ExpShapeBase {
    opType: ShapeOpType.Concat;
    left: ExpShape;
    right: ExpShape;
}

export interface ExpShapeBroadcast extends ExpShapeBase {
    opType: ShapeOpType.Broadcast;
    left: ExpShape;
    right: ExpShape;
}

//// STRING EXPRESSION

export const enum StringOpType {
    Const,
    Symbol,
    Slice,
    Concat,
}

export interface ExpStringBase extends SymExpBase {
    expType: SEType.String;
    opType: StringOpType;
}

export interface ExpStringConst extends ExpStringBase {
    opType: StringOpType.Const;
    value: string;
}

export interface ExpStringSymbol extends ExpStringBase {
    opType: StringOpType.Symbol;
    symbol: SymString;
}

// range is 0-based exclusive.
export interface ExpStringSlice extends ExpStringBase {
    opType: StringOpType.Slice;
    baseString: ExpString;
    start?: ExpNum;
    end?: ExpNum;
}

export interface ExpStringConcat extends ExpStringBase {
    opType: StringOpType.Concat;
    left: ExpString;
    right: ExpString;
}

export type ExpConsts = ExpNumConst | ExpShapeConst | ExpStringConst;

/// CONSTRUCTOR FUNCTIONS

export namespace SymExp {
    export function fromConst(value: SymExp | string | boolean | number): SymExp {
        switch (typeof value) {
            case 'number':
                return ExpNum.fromConst(value, undefined);
            case 'boolean':
                return ExpBool.fromConst(value, undefined);
            case 'string':
                return ExpString.fromConst(value, undefined);
            default:
                return value;
        }
    }

    // merge multiple hasSingleVar
    export function mergeSingleVar(variables: (SymVal | boolean)[]): SymVal | boolean {
        const vars: SymVal[] = [];
        for (const v of variables) {
            if (v === false) return false;
            if (v !== true) vars.push(v);
        }

        if (vars.length === 0) return true;

        const mainVar = vars[0];
        for (const v of vars) {
            if (mainVar.id !== v.id || mainVar.name !== v.name || mainVar.type !== v.type) return false;
        }

        return mainVar;
    }

    // check expression has only one variable name.
    // if found, return that symbolic variable
    // if found more than two, return false, else if not found any, return true
    export function hasSingleVar(ctrSet: ConstraintSet, exp: SymExp): SymVal | boolean {
        switch (exp.expType) {
            case SEType.Bool: {
                switch (exp.opType) {
                    case BoolOpType.Symbol: {
                        const range = ctrSet.getSymbolRange(exp.symbol);
                        if (range?.isConst()) {
                            return true;
                        }
                        return exp.symbol;
                    }
                    case BoolOpType.And:
                    case BoolOpType.Or:
                    case BoolOpType.Equal:
                    case BoolOpType.NotEqual:
                    case BoolOpType.LessThan:
                    case BoolOpType.LessThanOrEqual: {
                        const left = hasSingleVar(ctrSet, exp.left);
                        const right = hasSingleVar(ctrSet, exp.right);
                        return mergeSingleVar([left, right]);
                    }
                    case BoolOpType.Not:
                        return hasSingleVar(ctrSet, exp.baseBool);
                    default:
                        return true;
                }
            }
            case SEType.Num: {
                switch (exp.opType) {
                    case NumOpType.Symbol: {
                        const range = ctrSet.getSymbolRange(exp.symbol);
                        if (range?.isConst()) {
                            return true;
                        }
                        return exp.symbol;
                    }
                    case NumOpType.Const:
                        return true;
                    case NumOpType.Index:
                        // shape related operations is evaluated from constraintSet
                        return true;
                    case NumOpType.Max:
                    case NumOpType.Min:
                        return mergeSingleVar(exp.values.map((v) => hasSingleVar(ctrSet, v)));
                    case NumOpType.Numel:
                        return hasSingleVar(ctrSet, exp.shape);
                    case NumOpType.Uop:
                        return hasSingleVar(ctrSet, exp.baseValue);
                    case NumOpType.Bop: {
                        const left = hasSingleVar(ctrSet, exp.left);
                        const right = hasSingleVar(ctrSet, exp.right);
                        return mergeSingleVar([left, right]);
                    }
                }
                break;
            }
            case SEType.String:
            case SEType.Shape:
                // string and shape related operations is evaluated from constraintSet
                return true;
        }
    }

    export function toString(value: SymExp | string | number | boolean): string {
        let str: string;
        if (typeof value !== 'object') {
            return `${value}`;
        }
        switch (value.expType) {
            case SEType.String:
                str = ExpString.toString(value);
                break;
            case SEType.Bool:
                str = ExpBool.toString(value);
                break;
            case SEType.Num:
                str = ExpNum.toString(value);
                break;
            case SEType.Shape:
                str = ExpShape.toString(value);
                break;
        }

        return str;
    }

    // return symbol indices used in value
    export function extractSymbols(value: SymExp | string | number | boolean | undefined): number[] {
        const list: number[] = [];
        if (typeof value !== 'object') return list;

        function extract(value: SymExp | string | number | boolean | undefined): void {
            if (typeof value !== 'object') return;

            switch (value.expType) {
                case SEType.Bool: {
                    switch (value.opType) {
                        case BoolOpType.Symbol:
                            list.push(value.symbol.id);
                            return;
                        case BoolOpType.And:
                        case BoolOpType.Or:
                        case BoolOpType.Equal:
                        case BoolOpType.NotEqual:
                        case BoolOpType.LessThan:
                        case BoolOpType.LessThanOrEqual:
                            extract(value.left);
                            extract(value.right);
                            return;
                        case BoolOpType.Not:
                            extract(value.baseBool);
                            return;
                        default:
                            return;
                    }
                }
                case SEType.Num: {
                    switch (value.opType) {
                        case NumOpType.Symbol:
                            list.push(value.symbol.id);
                            return;
                        case NumOpType.Index:
                            extract(value.baseShape);
                            extract(value.index);
                            return;
                        case NumOpType.Max:
                        case NumOpType.Min:
                            value.values.forEach((v) => extract(v));
                            return;
                        case NumOpType.Numel:
                            extract(value.shape);
                            return;
                        case NumOpType.Uop:
                            extract(value.baseValue);
                            return;
                        case NumOpType.Bop: {
                            extract(value.left);
                            extract(value.right);
                            return;
                        }
                        default:
                            return;
                    }
                }
                case SEType.String:
                    switch (value.opType) {
                        case StringOpType.Symbol:
                            list.push(value.symbol.id);
                            return;
                        case StringOpType.Concat:
                            extract(value.left);
                            extract(value.right);
                            return;
                        case StringOpType.Slice:
                            extract(value.baseString);
                            extract(value.start);
                            extract(value.end);
                            return;
                        default:
                            return;
                    }
                case SEType.Shape:
                    switch (value.opType) {
                        case ShapeOpType.Symbol:
                            list.push(value.symbol.id);
                            return;
                        case ShapeOpType.Broadcast:
                        case ShapeOpType.Concat:
                            extract(value.left);
                            extract(value.right);
                            return;
                        case ShapeOpType.Slice:
                            extract(value.baseShape);
                            extract(value.start);
                            extract(value.end);
                            return;
                        case ShapeOpType.Set:
                            extract(value.baseShape);
                            extract(value.axis);
                            extract(value.dim);
                            return;
                        default:
                            return;
                    }
            }
        }

        extract(value);

        return list;
    }
}

export namespace ExpBool {
    export function fromConst(value: boolean, source: CodeSource | undefined): ExpBoolConst {
        return {
            expType: SEType.Bool,
            opType: BoolOpType.Const,
            value,
            source,
        };
    }

    export function fromSymbol(symbol: SymBool): ExpBoolSymbol {
        return {
            expType: SEType.Bool,
            opType: BoolOpType.Symbol,
            symbol,
            source: symbol.source,
        };
    }

    export function eq(left: SymExp, right: SymExp, source: CodeSource | undefined): ExpBoolEq {
        return {
            expType: SEType.Bool,
            opType: BoolOpType.Equal,
            left,
            right,
            source,
        };
    }

    export function neq(left: SymExp, right: SymExp, source: CodeSource | undefined): ExpBoolNeq {
        return {
            expType: SEType.Bool,
            opType: BoolOpType.NotEqual,
            left,
            right,
            source,
        };
    }
    export function lt(left: ExpNum | number, right: ExpNum | number, source: CodeSource | undefined): ExpBoolLt {
        return {
            expType: SEType.Bool,
            opType: BoolOpType.LessThan,
            left: ExpNum.toExp(left, undefined),
            right: ExpNum.toExp(right, undefined),
            source,
        };
    }
    export function lte(left: ExpNum | number, right: ExpNum | number, source: CodeSource | undefined): ExpBoolLte {
        return {
            expType: SEType.Bool,
            opType: BoolOpType.LessThanOrEqual,
            left: ExpNum.toExp(left, undefined),
            right: ExpNum.toExp(right, undefined),
            source,
        };
    }

    export function not(baseBool: ExpBool, source: CodeSource | undefined): ExpBoolNot {
        return {
            expType: SEType.Bool,
            opType: BoolOpType.Not,
            baseBool,
            source: source ? source : baseBool.source,
        };
    }

    export function and(left: ExpBool, right: ExpBool, source: CodeSource | undefined): ExpBoolAnd {
        return {
            expType: SEType.Bool,
            opType: BoolOpType.And,
            left,
            right,
            source,
        };
    }

    export function or(left: ExpBool, right: ExpBool, source: CodeSource | undefined): ExpBoolOr {
        return {
            expType: SEType.Bool,
            opType: BoolOpType.Or,
            left,
            right,
            source,
        };
    }

    export function toString(exp: ExpBool): string {
        switch (exp.opType) {
            case BoolOpType.Const:
                return `${exp.value}`;
            case BoolOpType.Symbol:
                return `${exp.symbol.name}`;
            case BoolOpType.Equal:
                return `(${SymExp.toString(exp.left)} == ${SymExp.toString(exp.right)})`;
            case BoolOpType.NotEqual:
                return `(${SymExp.toString(exp.left)} != ${SymExp.toString(exp.right)})`;
            case BoolOpType.LessThan:
                return `(${SymExp.toString(exp.left)} < ${SymExp.toString(exp.right)})`;
            case BoolOpType.LessThanOrEqual:
                return `(${SymExp.toString(exp.left)} <= ${SymExp.toString(exp.right)})`;
            case BoolOpType.Not:
                return `(${~SymExp.toString(exp.baseBool)})`;
            case BoolOpType.And:
                return `(${SymExp.toString(exp.left)} && ${SymExp.toString(exp.right)})`;
            case BoolOpType.Or:
                return `(${SymExp.toString(exp.left)} || ${SymExp.toString(exp.right)})`;
        }
    }
}

export namespace ExpNum {
    export function toExp(value: number | ExpNum, source: CodeSource | undefined): ExpNum {
        if (typeof value === 'number') {
            return ExpNum.fromConst(value, source);
        } else {
            return { ...value, source: source ? source : value.source };
        }
    }

    export function fromConst(value: number, source: CodeSource | undefined): ExpNumConst {
        return {
            expType: SEType.Num,
            opType: NumOpType.Const,
            value,
            source,
        };
    }

    export function fromSymbol(symbol: SymNumeric): ExpNumSymbol {
        return {
            expType: SEType.Num,
            opType: NumOpType.Symbol,
            symbol,
            source: symbol.source,
        };
    }

    export function box(value: number, source: CodeSource | undefined): ExpNumConstBoxed {
        return {
            expType: SEType.Num,
            opType: NumOpType.Const,
            value,
            source,
            boxed: true,
        };
    }

    export function bop(
        bopType: NumBopType,
        left: ExpNum | number,
        right: ExpNum | number,
        source: CodeSource | undefined
    ): ExpNumBop {
        if (typeof left === 'number') {
            left = ExpNum.fromConst(left, source);
        }
        if (typeof right === 'number') {
            right = ExpNum.fromConst(right, source);
        }
        return {
            expType: SEType.Num,
            opType: NumOpType.Bop,
            bopType,
            left: ExpNum.toExp(left, undefined),
            right: ExpNum.toExp(right, undefined),
            source,
        };
    }

    export function index(baseShape: ExpShape, index: ExpNum | number, source: CodeSource | undefined): ExpNumIndex {
        return {
            expType: SEType.Num,
            opType: NumOpType.Index,
            baseShape,
            index: ExpNum.toExp(index, undefined),
            source: source ? source : baseShape.source,
        };
    }

    export function max(values: (ExpNum | number)[], source: CodeSource | undefined): ExpNumMax {
        return {
            expType: SEType.Num,
            opType: NumOpType.Max,
            values: values.map((num) => (typeof num === 'number' ? ExpNum.fromConst(num, source) : num)),
            source,
        };
    }

    export function min(values: (ExpNum | number)[], source: CodeSource | undefined): ExpNumMin {
        return {
            expType: SEType.Num,
            opType: NumOpType.Min,
            values: values.map((num) => (typeof num === 'number' ? ExpNum.fromConst(num, source) : num)),
            source,
        };
    }

    export function numel(shape: ExpShape, source: CodeSource | undefined): ExpNumNumel {
        return {
            expType: SEType.Num,
            opType: NumOpType.Numel,
            shape,
            source: source ? source : shape.source,
        };
    }

    export function uop(uopType: NumUopType, baseValue: number | ExpNum, source: CodeSource | undefined): ExpNumUop {
        if (typeof baseValue === 'number') {
            baseValue = ExpNum.fromConst(baseValue, source);
        }
        return {
            expType: SEType.Num,
            opType: NumOpType.Uop,
            uopType,
            baseValue: ExpNum.toExp(baseValue, undefined),
            source,
        };
    }

    export function toString(exp: number | ExpNum): string {
        if (typeof exp === 'number') {
            return exp.toString();
        }
        switch (exp.opType) {
            case NumOpType.Const:
                return `${exp.value}`;
            case NumOpType.Symbol:
                return `${exp.symbol.name}`;
            case NumOpType.Bop:
                return `(${SymExp.toString(exp.left)} ${bopToString(exp.bopType)} ${SymExp.toString(exp.right)})`;
            case NumOpType.Index:
                return `index(${SymExp.toString(exp.baseShape)}, ${SymExp.toString(exp.index)})`;
            case NumOpType.Max:
                return `max(${exp.values.map((val) => SymExp.toString(val)).join(', ')})`;
            case NumOpType.Min:
                return `min(${exp.values.map((val) => SymExp.toString(val)).join(', ')})`;
            case NumOpType.Numel:
                return `numel(${ExpShape.toString(exp.shape)})`;
            case NumOpType.Uop:
                return `${uopToString(exp.uopType)}(${ExpNum.toString(exp.baseValue)})`;
        }
    }

    export function bopToString(bop: NumBopType): string {
        switch (bop) {
            case NumBopType.Add:
                return '+';
            case NumBopType.Sub:
                return '-';
            case NumBopType.Mul:
                return '*';
            case NumBopType.TrueDiv:
                return '/.';
            case NumBopType.FloorDiv:
                return '//';
            case NumBopType.Mod:
                return '%';
        }
    }

    export function uopToString(uop: NumUopType): string {
        switch (uop) {
            case NumUopType.Neg:
                return '-';
            case NumUopType.Floor:
                return 'floor';
            case NumUopType.Ceil:
                return 'ceil';
            case NumUopType.Abs:
                return 'abs';
        }
    }
}

export namespace ExpShape {
    export function fromConst(rank: number, dims: (number | ExpNum)[], source: CodeSource | undefined): ExpShapeConst {
        return {
            expType: SEType.Shape,
            opType: ShapeOpType.Const,
            rank,
            dims: dims.map((d) => ExpNum.toExp(d, undefined)),
            source,
        };
    }

    export function fromSymbol(symbol: SymShape): ExpShapeSymbol {
        return {
            expType: SEType.Shape,
            opType: ShapeOpType.Symbol,
            symbol,
            source: symbol.source,
        };
    }

    export function setDim(
        baseShape: ExpShape,
        axis: number | ExpNum,
        dim: number | ExpNum,
        source: CodeSource | undefined
    ): ExpShapeSet {
        return {
            expType: SEType.Shape,
            opType: ShapeOpType.Set,
            baseShape,
            axis: typeof axis === 'number' ? ExpNum.fromConst(axis, undefined) : axis,
            dim: typeof dim === 'number' ? ExpNum.fromConst(dim, undefined) : dim,
            source,
        };
    }

    export function slice(
        baseShape: ExpShape,
        start: ExpNum | number | undefined,
        end: ExpNum | number | undefined,
        source: CodeSource | undefined
    ): ExpShapeSlice {
        return {
            expType: SEType.Shape,
            opType: ShapeOpType.Slice,
            baseShape,
            start: start !== undefined ? ExpNum.toExp(start, undefined) : undefined,
            end: end !== undefined ? ExpNum.toExp(end, undefined) : undefined,
            source: source ? source : baseShape.source,
        };
    }

    export function concat(left: ExpShape, right: ExpShape, source: CodeSource | undefined): ExpShapeConcat {
        return {
            expType: SEType.Shape,
            opType: ShapeOpType.Concat,
            left,
            right,
            source,
        };
    }

    export function broadcast(left: ExpShape, right: ExpShape, source: CodeSource | undefined): ExpShapeBroadcast {
        return {
            expType: SEType.Shape,
            opType: ShapeOpType.Broadcast,
            left,
            right,
            source,
        };
    }

    export function toString(exp: ExpShape): string {
        switch (exp.opType) {
            case ShapeOpType.Const:
                return `T[${exp.dims.map((dim) => SymExp.toString(dim)).join(',')}]`;
            case ShapeOpType.Symbol:
                return `TSym[${exp.symbol.name}; ${SymExp.toString(exp.symbol.rank)}]`;
            case ShapeOpType.Set:
                return `set(${ExpShape.toString(exp.baseShape)}, ${ExpNum.toString(exp.axis)}, ${ExpNum.toString(
                    exp.dim
                )})`;
            case ShapeOpType.Slice:
                return `slice(${ExpShape.toString(exp.baseShape)}, ${exp.start ? ExpNum.toString(exp.start) : ':'}, ${
                    exp.end ? ExpNum.toString(exp.end) : ':'
                })`;
            case ShapeOpType.Concat:
                return `concat(${ExpShape.toString(exp.left)}, ${ExpShape.toString(exp.right)})`;
            case ShapeOpType.Broadcast:
                return `broadcast(${ExpShape.toString(exp.left)}, ${ExpShape.toString(exp.right)})`;
        }
    }

    export function getRank(exp: ExpShape): number | ExpNum {
        switch (exp.opType) {
            case ShapeOpType.Const:
                return exp.rank;
            case ShapeOpType.Symbol:
                return exp.symbol.rank;
            case ShapeOpType.Set:
                return getRank(exp.baseShape);
            case ShapeOpType.Slice: {
                const base = getRank(exp.baseShape);
                let start: number | ExpNum = exp.start === undefined ? 0 : exp.start;
                let end: number | ExpNum = exp.end === undefined ? base : exp.end;

                if (typeof start !== 'number' && start.opType === NumOpType.Const) {
                    start = start.value;
                }

                if (typeof end !== 'number' && end.opType === NumOpType.Const) {
                    end = end.value;
                }

                if (typeof start === 'number' && typeof end === 'number') {
                    const diff = end - start;
                    return diff >= 0 ? diff : 0;
                }

                return ExpNum.bop(NumBopType.Sub, end, start, exp.source);
            }
            case ShapeOpType.Concat: {
                let left: number | ExpNum = getRank(exp.left);
                let right: number | ExpNum = getRank(exp.right);

                if (typeof left !== 'number' && left.opType === NumOpType.Const) {
                    left = left.value;
                }

                if (typeof right !== 'number' && right.opType === NumOpType.Const) {
                    right = right.value;
                }

                if (typeof left === 'number' && typeof right === 'number') {
                    return left + right;
                }

                return ExpNum.bop(NumBopType.Add, left, right, exp.source);
            }
            case ShapeOpType.Broadcast: {
                let left: number | ExpNum = getRank(exp.left);
                let right: number | ExpNum = getRank(exp.right);

                if (typeof left !== 'number' && left.opType === NumOpType.Const) {
                    left = left.value;
                }

                if (typeof right !== 'number' && right.opType === NumOpType.Const) {
                    right = right.value;
                }

                if (typeof left === 'number' && typeof right === 'number') {
                    return left > right ? left : right;
                }

                return ExpNum.max([left, right], exp.source);
            }
        }
    }
}

export namespace ExpString {
    export function toExp(value: string | ExpString, source: CodeSource | undefined): ExpString {
        if (typeof value === 'string') {
            return ExpString.fromConst(value, source);
        } else if (source) {
            return { ...value, source };
        } else {
            return value;
        }
    }

    export function fromConst(value: string, source: CodeSource | undefined): ExpStringConst {
        return {
            expType: SEType.String,
            opType: StringOpType.Const,
            value,
            source,
        };
    }

    export function fromSymbol(symbol: SymString): ExpStringSymbol {
        return {
            expType: SEType.String,
            opType: StringOpType.Symbol,
            symbol,
            source: symbol.source,
        };
    }

    export function slice(
        baseString: ExpString,
        start: ExpNum | number | undefined,
        end: ExpNum | number | undefined,
        source: CodeSource | undefined
    ): ExpStringSlice {
        return {
            expType: SEType.String,
            opType: StringOpType.Slice,
            baseString,
            start: ExpNum.toExp(start ? start : 0, undefined),
            end: ExpNum.toExp(end ? end : 0, undefined),
            source,
        };
    }

    export function concat(
        left: ExpString | string,
        right: ExpString | string,
        source: CodeSource | undefined
    ): ExpStringConcat {
        return {
            expType: SEType.String,
            opType: StringOpType.Concat,
            left: ExpString.toExp(left, undefined),
            right: ExpString.toExp(right, undefined),
            source,
        };
    }

    export function toString(exp: ExpString): string {
        switch (exp.opType) {
            case StringOpType.Const:
                return exp.value;
            case StringOpType.Symbol:
                return exp.symbol.name;
            case StringOpType.Slice:
                return `slice(${ExpString.toString(exp.baseString)}, ${exp.start ? ExpNum.toString(exp.start) : ':'}, ${
                    exp.end ? ExpNum.toString(exp.end) : ':'
                })`;
            case StringOpType.Concat:
                return `concat(${ExpString.toString(exp.left)}, ${ExpString.toString(exp.right)})`;
        }
    }
}
