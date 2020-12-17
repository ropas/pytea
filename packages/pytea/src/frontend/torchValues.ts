/*
 * torchValues.ts
 * Copyright (c) Seoul National University.
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo
 *
 * Values for PyTea internal languages with Immutable.js
 */

import { List, Map, Record } from 'immutable';

import { ParseNode } from 'pyright-internal/parser/parseNodes';

import { ThEnv } from './torchEnvironments';
import { ThStmt, TSPass } from './torchStatements';

export type ThValue =
    | TVAddr
    | TVInt
    | TVFloat
    | TVString
    | TVBool
    | TVObject
    | TVFunc
    | TVNone
    | TVNotImpl
    | TVUndef
    | TVError;

export enum ThContFlag {
    Run,
    Cnt,
    Brk,
}

export const enum TVType {
    Addr,
    Int,
    Float,
    String,
    Bool,
    Object,
    Func,
    None,
    NotImpl,
    Undef,
    Error,
}

let _tvId = 0;
export function getNextTVId(): number {
    return ++_tvId;
}

interface ThValueBase {
    readonly type: TVType;
    readonly attrs: Map<string, ThValue>;
    readonly source?: ParseNode;
}

export namespace ThValue {
    export function toString(value: ThValue | ThContFlag): string {
        if (typeof value === 'object') {
            return value.toString();
        } else {
            switch (value) {
                case ThContFlag.Run:
                    return 'RUN';
                case ThContFlag.Cnt:
                    return 'CNT';
                case ThContFlag.Brk:
                    return 'BRK';
            }
        }
    }

    export function toStringStrMap(map: Map<string, ThValue>): string {
        if (map.count() === 0) {
            return '{}';
        }
        const keyArr = [...map.keys()];
        keyArr.sort();
        return `{ ${keyArr.map((i) => `${i} => ${map.get(i)?.toString()}`).join(', ')} }`;
    }

    export function toStringNumMap(map: Map<number, ThValue>): string {
        if (map.count() === 0) {
            return '{}';
        }
        const keyArr = [...map.keys()];
        keyArr.sort((a, b) => a - b);
        return `{ ${keyArr.map((i) => `${i} => ${map.get(i)?.toString()}`).join(', ')} }`;
    }

    // add address offset
    export function addOffset(value: ThValue, offset: number): ThValue {
        let newVal = value;

        switch (newVal.type) {
            case TVType.Addr:
                newVal = newVal.addOffset(offset);
                break;
            case TVType.Object:
                newVal = newVal.set(
                    'attrs',
                    newVal.attrs.mapEntries(([k, attr]) => [k, addOffset(attr, offset)])
                );
                newVal = newVal.set(
                    'indices',
                    newVal.indices.mapEntries(([k, attr]) => [k, addOffset(attr, offset)])
                );
                newVal = newVal.set(
                    'keyValues',
                    newVal.keyValues.mapEntries(([k, attr]) => [k, addOffset(attr, offset)])
                );
                break;
            case TVType.Func:
                TVFunc;
                newVal = newVal.set(
                    'defaults',
                    newVal.defaults.mapEntries(([k, v]) => [k, addOffset(v, offset)])
                );
                if (newVal.funcEnv) {
                    newVal = newVal.set('funcEnv', newVal.funcEnv.addOffset(offset));
                }
                break;
            default:
                break;
        }

        return newVal;
    }
}

interface TVAddrProps extends ThValueBase {
    readonly type: TVType.Addr;
    readonly addr: number;
}

const tvAddrDefaults: TVAddrProps = {
    type: TVType.Addr,
    addr: -1,
    attrs: Map(),
    source: undefined,
};

export class TVAddr extends Record(tvAddrDefaults) implements TVAddrProps {
    readonly type!: TVType.Addr;

    constructor(values?: Partial<TVAddrProps>) {
        values ? super(values) : super();
    }

    static create(addr: number, source?: ParseNode): TVAddr {
        const value: TVAddr = new TVAddr({
            addr,
            source,
        });
        return value;
    }

    toString(): string {
        return `Loc(${this.addr})`;
    }

    addOffset(offset: number): TVAddr {
        return this.addr >= 0 ? this.set('addr', this.addr + offset) : this;
    }
}

interface TVIntProps extends ThValueBase {
    readonly type: TVType.Int;
    readonly value: number;
}

const tvIntDefaults: TVIntProps = {
    type: TVType.Int,
    value: 0,
    attrs: Map(), // TODO: default methods
    source: undefined,
};

export class TVInt extends Record(tvIntDefaults) implements TVIntProps {
    readonly type!: TVType.Int;

    constructor(values?: Partial<TVIntProps>) {
        values ? super(values) : super();
    }

    static create(intValue: number, source?: ParseNode): TVInt {
        const value: TVInt = new TVInt({
            value: intValue,
            source,
        });
        return value;
    }

    toString(): string {
        return this.value.toString();
    }
}

interface TVFloatProps extends ThValueBase {
    readonly type: TVType.Float;
    readonly value: number;
}

const tvFloatDefaults: TVFloatProps = {
    type: TVType.Float,
    value: 0,
    attrs: Map(), // TODO: default methods
    source: undefined,
};

export class TVFloat extends Record(tvFloatDefaults) implements TVFloatProps {
    readonly type!: TVType.Float;

    constructor(values?: Partial<TVFloatProps>) {
        values ? super(values) : super();
    }

    static create(floatValue: number, source?: ParseNode): TVFloat {
        const value: TVFloat = new TVFloat({
            value: floatValue,
            source,
        });
        return value;
    }

    toString(): string {
        if (Number.isInteger(this.value)) {
            return `${this.value}.0`;
        } else {
            return this.value.toString();
        }
    }
}
interface TVStringProps extends ThValueBase {
    readonly type: TVType.String;
    readonly value: string;
}

const tvStringDefaults: TVStringProps = {
    type: TVType.String,
    value: '',
    attrs: Map(), // TODO: default methods
    source: undefined,
};

export class TVString extends Record(tvStringDefaults) implements TVStringProps {
    readonly type!: TVType.String;

    constructor(values?: Partial<TVStringProps>) {
        values ? super(values) : super();
    }

    static create(strValue: string, source?: ParseNode): TVString {
        const value: TVString = new TVString({
            value: strValue,
            source,
        });
        return value;
    }

    toString(): string {
        return `"${this.value}"`;
    }
}
interface TVBoolProps extends ThValueBase {
    readonly type: TVType.Bool;
    readonly value: boolean;
}

const tvBoolDefaults: TVBoolProps = {
    type: TVType.Bool,
    value: false,
    attrs: Map(), // TODO: default methods
    source: undefined,
};

export class TVBool extends Record(tvBoolDefaults) implements TVBoolProps {
    readonly type!: TVType.Bool;

    constructor(values?: Partial<TVBoolProps>) {
        values ? super(values) : super();
    }

    static create(boolValue: boolean, source?: ParseNode): TVBool {
        const value: TVBool = new TVBool({
            value: boolValue,
            source,
        });
        return value;
    }

    toString(): string {
        return this.value ? 'true' : 'false';
    }
}

interface TVObjectProps extends ThValueBase {
    readonly type: TVType.Object;
    readonly id: number;
    readonly attrs: Map<string, ThValue>;
    readonly indices: Map<number, ThValue>;
    readonly keyValues: Map<string, ThValue>;
}

const tvObjectDefaults: TVObjectProps = {
    type: TVType.Object,
    id: -1,
    attrs: Map(), // TODO: default methods
    indices: Map(),
    keyValues: Map(),
    source: undefined,
};

export class TVObject extends Record(tvObjectDefaults) implements TVObjectProps {
    readonly type!: TVType.Object;

    constructor(values?: Partial<TVObjectProps>) {
        values ? super(values) : super();
    }

    static create(source?: ParseNode): TVObject {
        const value: TVObject = new TVObject({
            id: getNextTVId(),
            source,
        });
        return value;
    }

    setAttr(attr: string, value: ThValue): TVObject {
        return this.set('attrs', this.attrs.set(attr, value));
    }

    setIndice(index: number, value: ThValue): TVObject {
        return this.set('indices', this.indices.set(index, value));
    }

    setKeyVal(key: string, value: ThValue): TVObject {
        return this.set('keyValues', this.keyValues.set(key, value));
    }

    getAttr(attr: string): ThValue | undefined {
        return this.attrs.get(attr);
    }

    getIndice(index: number): ThValue | undefined {
        return this.indices.get(index);
    }

    getKeyVal(key: string): ThValue | undefined {
        return this.keyValues.get(key);
    }

    toString(): string {
        const attrStr = `${ThValue.toStringStrMap(this.attrs)}`;
        const indStr = `${ThValue.toStringNumMap(this.indices)}`;
        const kvStr = `${ThValue.toStringStrMap(this.keyValues)}`;
        return `{ ${attrStr}, ${indStr}, ${kvStr} }`;
    }
}

interface TVFuncProps extends ThValueBase {
    readonly type: TVType.Func;
    readonly id: number;
    readonly name: string;
    readonly params: List<string>;
    readonly defaults: Map<string, ThValue>;
    readonly funcBody: ThStmt;
    readonly funcEnv?: ThEnv; // make it optional to avoid TypeScript circular import dependency
    readonly varargsParam?: string;
    readonly kwargsParam?: string;
}

const tvFuncDefaults: TVFuncProps = {
    type: TVType.Func,
    id: -1,
    name: '',
    params: List(),
    defaults: Map(),
    attrs: Map(), // TODO: default methods
    funcBody: TSPass.get(),
    funcEnv: undefined,
    varargsParam: undefined,
    kwargsParam: undefined,
    source: undefined,
};

export class TVFunc extends Record(tvFuncDefaults) implements TVFuncProps {
    readonly type!: TVType.Func;

    constructor(values?: Partial<TVFuncProps>) {
        values ? super(values) : super();
    }

    static create(
        name: string,
        params: List<string>,
        funcBody: ThStmt,
        funcEnv: ThEnv,
        source?: ParseNode,
        bindable?: boolean
    ): TVFunc {
        const value: TVFunc = new TVFunc({
            id: getNextTVId(),
            name,
            params,
            funcBody,
            funcEnv,
            source,
        });
        return value;
    }

    setDefaults(defaults: Map<string, ThValue>): TVFunc {
        return this.set('defaults', defaults);
    }

    setVKParam(varargsParam?: string, kwargsParam?: string) {
        return this.set('varargsParam', varargsParam).set('kwargsParam', kwargsParam);
    }

    toString(): string {
        return `${this.name}(${this.params.join(', ')})`;
    }

    bound(selfAddr: TVAddr): TVFunc | undefined {
        // TODO: staticmethod.
        // self value should be given as address.
        if (this.params.count() === 0) {
            return;
        }
        const selfName = this.params.get(0)!;
        const boundParams = this.params.slice(1);
        const newEnv = this.funcEnv?.setId(selfName, selfAddr);

        return this.set('params', boundParams).set('funcEnv', newEnv);
    }
}

interface TVNoneProps extends ThValueBase {
    readonly type: TVType.None;
}

const tvNoneDefaults: TVNoneProps = {
    type: TVType.None,
    attrs: Map(),
    source: undefined,
};

export class TVNone extends Record(tvNoneDefaults) implements TVNoneProps {
    readonly type!: TVType.None;
    static _none = new TVNone();

    private constructor(values?: Partial<TVNoneProps>) {
        values ? super(values) : super();
    }

    static create(source?: ParseNode): TVNone {
        if (!source) return TVNone._none;
        const value: TVNone = new TVNone({
            source,
        });
        return value;
    }

    toString(): string {
        return 'None';
    }
}

interface TVNotImplProps extends ThValueBase {
    readonly type: TVType.NotImpl;
    readonly reason?: string;
}

const tvNotImplDefaults: TVNotImplProps = {
    type: TVType.NotImpl,
    attrs: Map(),
    reason: undefined,
    source: undefined,
};

export class TVNotImpl extends Record(tvNotImplDefaults) implements TVNotImplProps {
    readonly type!: TVType.NotImpl;
    static _notImpl = new TVNotImpl();

    private constructor(values?: Partial<TVNotImplProps>) {
        values ? super(values) : super();
    }

    static create(reason?: string, source?: ParseNode): TVNotImpl {
        if (!reason && !source) {
            return this._notImpl;
        }
        const value: TVNotImpl = new TVNotImpl({
            reason,
            source,
        });
        return value;
    }

    toString(): string {
        return `NotImpl(${this.reason ? this.reason : ''})`;
    }
}

interface TVUndefProps extends ThValueBase {
    readonly type: TVType.Undef;
}

const tvUndefDefaults: TVUndefProps = {
    type: TVType.Undef,
    attrs: Map(),
    source: undefined,
};

export class TVUndef extends Record(tvUndefDefaults) implements TVUndefProps {
    readonly type!: TVType.Undef;
    static _undef = new TVUndef();

    private constructor(values?: Partial<TVUndefProps>) {
        values ? super(values) : super();
    }

    static create(source?: ParseNode): TVUndef {
        if (!source) {
            return TVUndef._undef;
        }
        const value: TVUndef = new TVUndef({
            source,
        });
        return value;
    }

    toString(): string {
        return 'UNDEF';
    }
}

interface TVErrorProps extends ThValueBase {
    readonly type: TVType.Error;
    readonly reason: string;
}

const tvErrorDefaults: TVErrorProps = {
    type: TVType.Error,
    reason: 'unexpected error',
    attrs: Map(),
    source: undefined,
};

export class TVError extends Record(tvErrorDefaults) implements TVErrorProps {
    readonly type!: TVType.Error;

    private constructor(values?: Partial<TVErrorProps>) {
        values ? super(values) : super();
    }

    static create(reason: string, source?: ParseNode): TVError {
        const value: TVError = new TVError({
            reason,
            source,
        });
        return value;
    }

    toString(): string {
        return `ERROR(${this.reason})`;
    }
}
