/*
 * guard.ts
 * Copyright (c) Seoul National University.
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo (mersshs@gmail.com)
 *
 * require guard in pylib side
 */
import { fetchAddr, isSize } from '../backend/backUtils';
import { Context, ContextSet } from '../backend/context';
import { CodeSource, ShValue, SVBool, SVInt, SVType } from '../backend/sharpValues';
import { LCImpl } from '.';
import { LCBase } from './libcall';

// use with assert
// e.g.) assert LibCall.guard.require_lt(3, 5, "3 < 5")
export namespace GuardLCImpl {
    // LibCall.guard.require_lt(a, b, message): add require(a < b, message) to this context. return true
    export function require_lt(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.guard.require_lt': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSetWith(SVBool.create(true, source));
        }

        const heap = ctx.heap;
        const [addrA, addrB, messageAddr] = params;
        const a = fetchAddr(addrA, heap);
        const b = fetchAddr(addrB, heap);
        const msg = fetchAddr(messageAddr, heap);

        if (
            !(a?.type === SVType.Int || a?.type === SVType.Float) ||
            !(b?.type === SVType.Int || b?.type === SVType.Float)
        ) {
            return ctx
                .warnWithMsg(`from 'LibCall.guard.require_lt': operand is not a numeric`, source)
                .toSetWith(SVBool.create(true, source));
        }

        if (!(msg?.type === SVType.String && typeof msg.value === 'string')) {
            return ctx
                .warnWithMsg(`from 'LibCall.guard.require_lt: message is not a constant string`, source)
                .toSetWith(SVBool.create(true, source));
        }

        return ctx.require(ctx.genLt(a.value, b.value, source), msg.value, source).return(SVBool.create(true, source));
    }

    export function require_lte(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.guard.require_lte': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSetWith(SVBool.create(true, source));
        }

        const heap = ctx.heap;
        const [addrA, addrB, messageAddr] = params;
        const a = fetchAddr(addrA, heap);
        const b = fetchAddr(addrB, heap);
        const msg = fetchAddr(messageAddr, heap);

        if (
            !(a?.type === SVType.Int || a?.type === SVType.Float) ||
            !(b?.type === SVType.Int || b?.type === SVType.Float)
        ) {
            return ctx
                .warnWithMsg(`from 'LibCall.guard.require_lte': operand is not a numeric`, source)
                .toSetWith(SVBool.create(true, source));
        }

        if (!(msg?.type === SVType.String && typeof msg.value === 'string')) {
            return ctx
                .warnWithMsg(`from 'LibCall.guard.require_lte: message is not a constant string`, source)
                .toSetWith(SVBool.create(true, source));
        }

        return ctx.require(ctx.genLte(a.value, b.value, source), msg.value, source).return(SVBool.create(true, source));
    }

    export function require_eq(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.guard.require_eq': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSetWith(SVBool.create(true, source));
        }

        const heap = ctx.heap;
        const [addrA, addrB, messageAddr] = params;
        const a = fetchAddr(addrA, heap);
        const b = fetchAddr(addrB, heap);
        const msg = fetchAddr(messageAddr, heap);

        if (
            !(a?.type === SVType.Int || a?.type === SVType.Float || a?.type === SVType.String) ||
            !(b?.type === SVType.Int || b?.type === SVType.Float || b?.type === SVType.String)
        ) {
            return ctx
                .warnWithMsg(`from 'LibCall.guard.require_eq': operand is not a numeric or string`, source)
                .toSetWith(SVBool.create(true, source));
        }

        if (!(msg?.type === SVType.String && typeof msg.value === 'string')) {
            return ctx
                .warnWithMsg(`from 'LibCall.guard.require_eq: message is not a constant string`, source)
                .toSetWith(SVBool.create(true, source));
        }

        return ctx.require(ctx.genEq(a.value, b.value, source), msg.value, source).return(SVBool.create(true, source));
    }

    export function require_neq(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.guard.require_neq': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSetWith(SVBool.create(true, source));
        }

        const heap = ctx.heap;
        const [addrA, addrB, messageAddr] = params;
        const a = fetchAddr(addrA, heap);
        const b = fetchAddr(addrB, heap);
        const msg = fetchAddr(messageAddr, heap);

        if (
            !(a?.type === SVType.Int || a?.type === SVType.Float || a?.type === SVType.String) ||
            !(b?.type === SVType.Int || b?.type === SVType.Float || b?.type === SVType.String)
        ) {
            return ctx
                .warnWithMsg(`from 'LibCall.guard.require_neq': operand is not a numeric or string`, source)
                .toSetWith(SVBool.create(true, source));
        }

        if (!(msg?.type === SVType.String && typeof msg.value === 'string')) {
            return ctx
                .warnWithMsg(`from 'LibCall.guard.require_neq: message is not a constant string`, source)
                .toSetWith(SVBool.create(true, source));
        }

        return ctx.require(ctx.genNeq(a.value, b.value, source), msg.value, source).return(SVBool.create(true, source));
    }

    export function require_broadcastable(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.guard.require_broadcastable': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSetWith(SVBool.create(true, source));
        }

        const heap = ctx.heap;
        const [leftAddr, rightAddr, messageAddr] = params;

        const left = fetchAddr(leftAddr, heap);
        const right = fetchAddr(rightAddr, heap);
        const msg = fetchAddr(messageAddr, heap);

        if (!isSize(left)) {
            return ctx
                .warnWithMsg(`from 'LibCall.guard.require_broadcastable': left is not a Size type`, source)
                .toSet();
        }

        if (!isSize(right)) {
            return ctx
                .warnWithMsg(`from 'LibCall.guard.require_broadcastable': right is not a Size type`, source)
                .toSet();
        }

        if (!(msg?.type === SVType.String && typeof msg.value === 'string')) {
            return ctx
                .warnWithMsg(`from 'LibCall.guard.require_broadcastable: message is not a constant string`, source)
                .toSetWith(SVBool.create(true, source));
        }

        return ctx
            .require(ctx.genBroadcastable(left.shape, right.shape, source), msg.value, source)
            .return(SVBool.create(true, source));
    }

    export function require_shape_eq(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.guard.require_shape_eq': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSetWith(SVBool.create(true, source));
        }

        const heap = ctx.heap;
        const [leftAddr, rightAddr, messageAddr] = params;

        const left = fetchAddr(leftAddr, heap);
        const right = fetchAddr(rightAddr, heap);
        const msg = fetchAddr(messageAddr, heap);

        if (!isSize(left)) {
            return ctx.warnWithMsg(`from 'LibCall.guard.require_shape_eq': left is not a Size type`, source).toSet();
        }

        if (!isSize(right)) {
            return ctx.warnWithMsg(`from 'LibCall.guard.require_shape_eq': right is not a Size type`, source).toSet();
        }

        if (!(msg?.type === SVType.String && typeof msg.value === 'string')) {
            return ctx
                .warnWithMsg(`from 'LibCall.guard.require_shape_eq: message is not a constant string`, source)
                .toSetWith(SVBool.create(true, source));
        }

        return ctx
            .require(ctx.genEq(left.shape, right.shape, source), msg.value, source)
            .return(SVBool.create(true, source));
    }

    // return new symbolic variable that is equal with given value.
    // if given value is constant, return that constant
    export function new_symbol_int(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.guard.new_symbol_int': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }

        const heap = ctx.heap;
        const [nameAddr, valueAddr] = params;

        const name = fetchAddr(nameAddr, heap);
        const value = fetchAddr(valueAddr, heap);

        if (name?.type !== SVType.String || typeof name.value !== 'string') {
            return ctx
                .warnWithMsg(`from 'LibCall.guard.new_symbol_int': name is not a constant string`, source)
                .toSet();
        }

        if (value?.type !== SVType.Int) {
            return ctx.warnWithMsg(`from 'LibCall.guard.new_symbol_int': value is not an integer type`, source).toSet();
        }

        const valueRng = ctx.getCachedRange(value.value);
        if (valueRng?.isConst()) {
            return ctx.toSetWith(SVInt.create(valueRng.start, value.source));
        }

        const newCtx = ctx.genIntEq(name.value, value.value, source);
        const exp = SVInt.create(newCtx.retVal, source);

        return newCtx.toSetWith(exp);
    }

    export const libCallImpls: { [key: string]: LCImpl } = {
        require_lt,
        require_lte,
        require_eq,
        require_neq,
        require_broadcastable,
        require_shape_eq,
        new_symbol_int,
    };
}

export const libCallMap: Map<string, LCImpl> = new Map([...Object.entries(GuardLCImpl.libCallImpls)]);
