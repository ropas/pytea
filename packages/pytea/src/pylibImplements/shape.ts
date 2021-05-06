/*
 * shape.ts
 * Copyright (c) Seoul National University.
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo (mersshs@gmail.com)
 *
 * Direct call of shape operations in context.
 */
import { fetchAddr, sanitizeAddr } from '../backend/backUtils';
import { Constraint } from '../backend/constraintType';
import { Context, ContextSet } from '../backend/context';
import {
    absExpIndexByLen,
    fetchSize,
    genTensor,
    isInstanceOf,
    reluLen,
    simplifyNum,
    simplifyShape,
} from '../backend/expUtils';
import { CodeSource, ShValue, SVInt, SVObject, SVSize, SVType } from '../backend/sharpValues';
import { ExpNum, ExpShape, NumBopType, NumUopType } from '../backend/symExpressions';
import { LCImpl } from '.';
import { LCBase } from './libcall';

export namespace ShapeLCImpl {
    export function setSize(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length < 2) {
            return ctx.warnWithMsg(
                `from 'LibCall.shape.setSize': got insufficient number of argument: ${params.length}`,
                source
            ).toSet();
        }

        const heap = ctx.heap;
        const [selfAddr, argsAddr] = params;

        const self = fetchAddr(selfAddr, heap)! as SVObject;
        const args = fetchAddr(argsAddr, heap);

        if (args && args instanceof SVSize) {
            const sizedSelf = SVSize.toSize(ctx, self, args.shape);
            const newHeap = ctx.heap.setVal(self.addr, sizedSelf);

            return ctx.setHeap(newHeap).toSetWith(self);
        }

        return ctx.parseSize(argsAddr, source).map((ctx) => {
            let shape: ExpShape;
            let newCtx: Context<any> = ctx;
            if (typeof ctx.retVal === 'string') {
                newCtx = ctx.warnWithMsg(ctx.retVal, source).genIntGte('tempRank', 0, source);
                shape = ExpShape.fromSymbol(newCtx.genSymShape('tempShape', newCtx.retVal, source));
            } else {
                shape = ctx.retVal;
            }

            const sizedSelf = SVSize.toSize(ctx, self, shape);
            const newHeap = ctx.heap.setVal(self.addr, sizedSelf);

            return newCtx.setHeap(newHeap).setRetVal(self);
        });
    }

    // get (tensor, axis, repeat_count). returns new tensor repeated by repeat_count through axis.
    export function repeat(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
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

        const [axisPos, axisNeg] = ctx.ifThenElse(ctx.genLte(0, axisVal, source), source);
        const posPath = axisPos
            .flatMap((ctx) => ctx.shRepeat(shape, axisVal, countVal, source))
            .flatMap((ctx) => genTensor(ctx, ctx.retVal, source));

        const negPath = axisNeg
            .flatMap((ctx) => ctx.shRepeat(shape, ExpNum.bop(NumBopType.Sub, rank, axisVal, source), countVal, source))
            .flatMap((ctx) => genTensor(ctx, ctx.retVal, source));

        return posPath.join(negPath);
    }


    // shapeConcat(T[1, 2, 3], T[4, 5, 6]):
    //     set size of 'obj' to be T[1, 2, 3, 4, 5, 6].
    export function concat(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return _randShape(ctx,
                `from 'LibCall.shape.shapeConcat': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [leftAddr, rightAddr, objAddr] = params;

        const left = fetchSize(leftAddr, heap);
        const right = fetchSize(rightAddr, heap);
        const obj = fetchAddr(objAddr, heap);

        if (typeof left === 'string') {
            return ctx.warnWithMsg(`from 'LibCall.shape.shapeConcat': left is not a Size type`, source).toSet();
        }
        if (typeof right === 'string') {
            return ctx.warnWithMsg(`from 'LibCall.shape.shapeConcat': right is not a Size type`, source).toSet();
        }
        if (objAddr.type !== SVType.Addr || obj?.type !== SVType.Object) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.shape.shapeConcat': not an object type:\n\t${objAddr.toString()} -> ${obj?.toString()}`,
                    source
                )
                .toSet();
        }

        const newShape = ExpShape.concat(left.shape, right.shape, source);
        const sizeObj = SVSize.toSize(ctx, obj, newShape);
        const newHeap = heap.setVal(objAddr, obj.setAttr('shape', sizeObj));
        return ctx.setHeap(newHeap).toSetWith(objAddr);
    }

    // extract shape from list of list of ... list of integer/tensor/ndarray
    //
    // TODO: we look only the first items of list to infer a shape.
    //       consistency of each item's shape is not checked for now.
    export function extractShape(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const params = ctx.retVal.params;

        if (params.length !== 1) {
            return _randSize(
                ctx,
                `from 'LibCall.shape.extractShape': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const valueAddr = params[0];

        return buildShape(ctx, valueAddr, source).map((ctx) =>
            ctx.setRetVal(SVSize.createSize(ctx, ctx.retVal, source))
        );
    }

    function buildShape(
        ctx: Context<unknown>,
        maySized: ShValue,
        source: CodeSource | undefined
    ): ContextSet<ExpShape> {
        const value = fetchAddr(maySized, ctx.heap);
        if (!value) {
            return _randShape(ctx, `from 'LibCall.shape.extractShape': got undefined value`, source);
        }

        switch (value.type) {
            case SVType.Bool:
            case SVType.Int:
            case SVType.Float:
                return ctx.toSetWith(ExpShape.fromConst(0, [], source));
            case SVType.Object: {
                // already has size
                const size = fetchSize(value, ctx.heap);
                if (typeof size === 'object') {
                    return ctx.toSetWith(size.shape);
                }

                // has length
                const length = value.getAttr('$length');
                if (length?.type === SVType.Int) {
                    return ctx.getIndiceDeep(value, 0, source).flatMap((ctx) => {
                        const inner = ctx.retVal;
                        return buildShape(ctx, inner, source).map((ctx) => {
                            const shape = simplifyShape(
                                ctx.ctrSet,
                                ExpShape.concat(ExpShape.fromConst(1, [length.value], source), ctx.retVal, source)
                            );
                            return ctx.setRetVal(shape);
                        });
                    });
                }

                // fall back to unknown
                return _randShape(
                    ctx,
                    `from 'LibCall.shape.extractShape': failed to infer size. return temp shape.`,
                    source
                );
            }
            default:
                return _randShape(
                    ctx,
                    `from 'LibCall.shape.extractShape': extract shape from invalid type. return temp shape.`,
                    source
                );
        }
    }

    function _randSize(ctx: Context<unknown>, message: string, source: CodeSource | undefined): ContextSet<ShValue> {
        return _randShape(ctx, message, source).map((ctx) =>
            ctx.setRetVal(SVSize.createSize(ctx, ctx.retVal, source))
        );
    }

    function _randNewSize(ctx: Context<unknown>, baseSize: SVObject, message: string, source: CodeSource | undefined): ContextSet<ShValue> {
        return _randShape(ctx, message, source).map((ctx) => {
            const [addr, heap] = ctx.heap.malloc(source);
            const size = SVSize.()
        }


        );
    }

    function _randShape(ctx: Context<unknown>, msg: string, source: CodeSource | undefined): ContextSet<ExpShape> {
        const rank = ctx.genSymInt('WarnTempRank', source);
        const sym = ctx.genSymShape('WarnTempShape', ExpNum.fromSymbol(rank), source);
        const shape = ExpShape.fromSymbol(sym);
        return ctx.warnWithMsg(msg, source).toSetWith(shape);
    }

    export function randShape(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const rank = ctx.genSymInt('WarnTempRank', source);
        const sym = ctx.genSymShape('WarnTempShape', ExpNum.fromSymbol(rank), source);
        const shape = ExpShape.fromSymbol(sym);
        return ctx.toSetWith(SVSize.createSize(ctx, shape, source));
    }

    export const libCallImpls: { [key: string]: LCImpl } = {
        repeat,
        getItem,
        slice,
        concat,
        extractShape,
        randShape,
        setSize
    };
}

export const libCallMap: Map<string, LCImpl> = new Map([...Object.entries(ShapeLCImpl.libCallImpls)]);
