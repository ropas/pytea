/*
 * shape.ts
 * Copyright (c) Seoul National University.
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo (mersshs@gmail.com)
 *
 * Direct call of shape operations in context.
 */
import { Constraint } from 'src/backend/constraintType';

import { ParseNode } from 'pyright-internal/parser/parseNodes';

import { fetchAddr, sanitizeAddr } from '../backend/backUtils';
import { Context, ContextSet } from '../backend/context';
import {
    absExpIndexByLen,
    fetchSize,
    genTensor,
    isInstanceOf,
    isSize,
    reluLen,
    simplifyNum,
    simplifyShape,
} from '../backend/expUtils';
import { ShValue, SVInt, SVSize, SVType } from '../backend/sharpValues';
import { ExpNum, ExpShape, NumBopType, NumUopType } from '../backend/symExpressions';
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

    // implementation slice of torch.Tensor.__getitem__
    // axis range is already checked from tensor.__getitem__
    // params: [inputShape, axis, index]
    export function tensorGetItem(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.shape.tensorGetItem': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const { env, heap } = ctx;
        const [sizeAddr, axisAddr, indexAddr] = params;

        const size = fetchAddr(sizeAddr, heap);
        const axis = fetchAddr(axisAddr, heap);
        const index = fetchAddr(indexAddr, heap);
        const mayTensorIndex = fetchSize(indexAddr, heap);

        if (!(size && size instanceof SVSize)) {
            return ctx.warnWithMsg(`from 'LibCall.shape.tensorGetItem': input is not a Size type`, source).toSet();
        }
        if (!(axis && axis.type === SVType.Int && typeof axis.value === 'number')) {
            return ctx.warnWithMsg(`from 'LibCall.shape.tensorGetItem': axis is not a number`, source).toSet();
        }
        if (!index) {
            return ctx.warnWithMsg(`from 'LibCall.shape.tensorGetItem': index is undefined`, source).toSet();
        }

        const slice = sanitizeAddr(ctx.env.getId('slice'), heap);
        const shape = size.shape;
        const axisValue = axis.value;

        if (index.type === SVType.Int) {
            // index is contant int
            const indexNum = index.value;
            const indexDim = ExpNum.index(shape, indexNum, source);

            return ctx
                .require(
                    [
                        ctx.genLte(ExpNum.bop(NumBopType.Sub, 0, indexDim, source), indexNum, source),
                        ctx.genLte(indexNum, ExpNum.bop(NumBopType.Sub, indexDim, 1, source), source),
                    ],
                    `from 'LibCall.shape.tensorGetItem': index out of range`,
                    source
                )
                .map((ctx) => {
                    const left = ExpShape.slice(shape, undefined, axisValue, source);
                    const right = ExpShape.slice(shape, ExpNum.bop(NumBopType.Add, axisValue, 1));
                    const result = simplifyShape(ctx.ctrSet, ExpShape.concat(left, right, source));
                    return ctx.setRetVal(SVSize.createSize(ctx, result, source));
                });
        } else if (index.type === SVType.Object) {
            if (slice && isInstanceOf(index, slice, env, heap) === true) {
                const start = fetchAddr(index.getAttr('start'), heap);
                const end = fetchAddr(index.getAttr('stop'), heap);
                const step = fetchAddr(index.getAttr('step'), heap);

                const currDim = ExpNum.index(shape, axisValue, source);
                let hasError = false;
                let startN, endN, stepN: ExpNum | number | undefined;

                if (start?.type === SVType.Int) startN = start.value;
                else if (!(start?.type === SVType.None)) hasError = true;
                if (end?.type === SVType.Int) endN = end.value;
                else if (!(end?.type === SVType.None)) hasError = true;
                if (step?.type === SVType.Int) stepN = step.value;
                else if (!(step?.type === SVType.None)) hasError = true;

                if (hasError) {
                    return ctx
                        .warnWithMsg(
                            `from 'LibCall.shape.tensorGetItem: slice value is not an integer or None.`,
                            source
                        )
                        .toSet();
                }

                const ctrList: Constraint[] = [];
                if (startN !== undefined) {
                    startN = absExpIndexByLen(startN, currDim, source, ctx.ctrSet);
                    ctrList.push(ctx.genLte(0, startN, source));
                } else {
                    startN = 0;
                }
                if (endN !== undefined) {
                    endN = absExpIndexByLen(endN, currDim, source, ctx.ctrSet);
                    ctrList.push(ctx.genLte(0, endN, source));
                } else {
                    endN = currDim;
                }

                // return ceil((endN - startN) // stepN)
                let newDim = reluLen(startN, endN, source, ctx.ctrSet);
                if (stepN === undefined) stepN = 1;
                if (typeof stepN !== 'number' || stepN !== 1) {
                    newDim = ExpNum.uop(NumUopType.Ceil, ExpNum.bop(NumBopType.TrueDiv, newDim, stepN, source), source);
                }
                const newShape = simplifyShape(ctx.ctrSet, ExpShape.setDim(shape, axisValue, newDim, source));
                if (ctrList.length === 0) {
                    return ctx.setRetVal(SVSize.createSize(ctx, newShape, source)).toSet();
                }
                return ctx
                    .require(ctrList, 'index out of range', source)
                    .map((ctx) => ctx.setRetVal(SVSize.createSize(ctx, newShape, source)));
            } else if (mayTensorIndex && mayTensorIndex instanceof SVSize) {
                // TOOD: distinguish dtype of tensor
                // TODO: Implement other advanced tensor indexing
                //       https://numpy.org/doc/stable/reference/arrays.indexing.html

                // mask indexing
                const sizeNumel = ExpNum.numel(shape, source);
                const mask = mayTensorIndex;
                const maskCtx = ctx.genIntGte('maskIndex', 0, source);
                const maskNum = maskCtx.retVal;

                return maskCtx
                    .require([maskCtx.genLte(maskNum, sizeNumel, source), maskCtx.genEq(shape, mask.shape, source)])
                    .flatMap((ctx) => {
                        return genTensor(ctx, ExpShape.fromConst(1, [maskNum], source));
                    });
            }
        }

        return ctx
            .warnWithMsg(
                `from 'LibCall.tensor.getItem: only indexing by integer, slice or bool tensor is supported currently.`,
                source
            )
            .toSet();
    }

    export const libCallImpls: { [key: string]: LCImpl } = {
        repeat,
        size_getitem,
        size_len,
        tensorGetItem,
    };
}

export const libCallMap: Map<string, LCImpl> = new Map([...Object.entries(ShapeLCImpl.libCallImpls)]);
