/*
 * shape.ts
 * Copyright (c) Seoul National University.
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo
 *
 * Direct call of shape operations in context.
 */
import { ParseNode } from 'pyright-internal/parser/parseNodes';

import { fetchAddr } from '../../backend/backUtils';
import { Context, ContextSet } from '../../backend/context';
import { fetchSize, genTensor, simplifyNum } from '../../backend/expUtils';
import { ShValue, SVInt, SVSize, SVType } from '../../backend/sharpValues';
import { ExpNum, ExpShape, NumBopType } from '../../backend/symExpressions';
import { LCImpl } from '.';
import { LCBase } from './libcall';

export namespace ShapeLCImpl {
    // get (tensor, axis, repeat_count). returns new tensor repeated by repeat_count through axis.
    export function repeat(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.shape.repeat': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [tensorAddr, axisAddr, countAddr] = params;

        const tensorSize = fetchSize(tensorAddr, heap);
        const axis = fetchAddr(axisAddr, heap);
        const count = fetchAddr(countAddr, heap);

        if (typeof tensorSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.shape.repeat: ${tensorSize}`, source);
        }
        if (axis?.type !== SVType.Int) {
            return ctx.warnTensorWithMsg(`from 'LibCall.shape.repeat: axis value is not an integer`, source);
        } else if (count?.type !== SVType.Int) {
            return ctx.warnTensorWithMsg(`from 'LibCall.shape.repeat: count value is not an integer`, source);
        }

        const shape = tensorSize.shape;
        const rank = ExpShape.getRank(shape);
        const axisVal = axis.value;
        const countVal = count.value;

        const [axisPos, axisNeg] = ctx.ifThenElse(ctx.genLte(0, axisVal), source);
        const posPath = axisPos
            .flatMap((ctx) => ctx.shRepeat(shape, axisVal, countVal, source))
            .flatMap((ctx) => genTensor(ctx, ctx.retVal, source));

        const negPath = axisNeg
            .flatMap((ctx) => ctx.shRepeat(shape, ExpNum.bop(NumBopType.Sub, rank, axisVal, source), countVal, source))
            .flatMap((ctx) => genTensor(ctx, ctx.retVal, source));

        return posPath.join(negPath);
    }

    export function size_getitem(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.shape.size_getitem': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }

        const heap = ctx.heap;
        const [sizeAddr, itemAddr] = params;

        const size = fetchAddr(sizeAddr, heap);
        const axis = fetchAddr(itemAddr, heap);

        if (!(size instanceof SVSize)) {
            return ctx
                .warnWithMsg(`from 'LibCall.shape.size_getitem: ${size?.toString()} is not an SVSize`, source)
                .toSet();
        } else if (axis?.type !== SVType.Int) {
            return ctx.warnWithMsg(`from 'LibCall.shape.size_getitem: axis value is not an integer`, source).toSet();
        }

        const idx = simplifyNum(ctx.ctrSet, ExpNum.index(size.shape, axis.value, source));
        return ctx.toSetWith(SVInt.create(idx, source));
    }

    export function size_len(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 1) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.shape.size_len': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }

        const heap = ctx.heap;
        const [sizeAddr] = params;

        const size = fetchAddr(sizeAddr, heap);

        if (!(size instanceof SVSize)) {
            return ctx
                .warnWithMsg(`from 'LibCall.shape.size_len: ${size?.toString()} is not an SVSize`, source)
                .toSet();
        }

        return ctx.toSetWith(SVInt.create(ExpShape.getRank(size.shape), source));
    }

    export const libCallImpls: { [key: string]: LCImpl } = {
        repeat,
        size_getitem,
        size_len,
    };
}

export const libCallMap: Map<string, LCImpl> = new Map([...Object.entries(ShapeLCImpl.libCallImpls)]);
