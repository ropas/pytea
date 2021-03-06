/*
 * guard.ts
 * Copyright (c) Seoul National University.
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo (mersshs@gmail.com)
 *
 * require guard in pylib side
 */
import { fetchAddr } from '../backend/backUtils';
import { Context, ContextSet } from '../backend/context';
import { fetchSize } from '../backend/expUtils';
import { CodeSource, ShValue, SVBool, SVType } from '../backend/sharpValues';
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

    // shapeConcat(T[1, 2, 3], T[4, 5, 6], obj):
    //     set size of 'obj' to be T[1, 2, 3, 4, 5, 6].
    export function require_broadcastable(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.shape.require_broadcastable': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSetWith(SVBool.create(true, source));
        }

        const heap = ctx.heap;
        const [leftAddr, rightAddr, messageAddr] = params;

        const left = fetchSize(leftAddr, heap);
        const right = fetchSize(rightAddr, heap);
        const msg = fetchAddr(messageAddr, heap);

        if (typeof left === 'string') {
            return ctx
                .warnWithMsg(`from 'LibCall.shape.require_broadcastable': left is not a Size type`, source)
                .toSet();
        }

        if (typeof right === 'string') {
            return ctx
                .warnWithMsg(`from 'LibCall.shape.require_broadcastable': right is not a Size type`, source)
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

    export const libCallImpls: { [key: string]: LCImpl } = {
        require_lt,
        require_lte,
        require_eq,
        require_neq,
        require_broadcastable,
    };
}

export const libCallMap: Map<string, LCImpl> = new Map([...Object.entries(GuardLCImpl.libCallImpls)]);
