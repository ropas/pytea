/*
 * shape.ts
 * Copyright (c) Seoul National University.
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo (mersshs@gmail.com)
 *
 * Direct call of shape operations in context.
 * Every arguments should be SVSize objects. Also returns SVSize.
 */
import { fetchAddr, fetchSize, isSize } from '../backend/backUtils';
import { Context, ContextSet } from '../backend/context';
import { absExpIndexByLen, simplifyNum } from '../backend/expUtils';
import { simplifyShape } from '../backend/expUtils';
import { CodeSource, ShValue, SVInt, SVObject, SVSize, SVType } from '../backend/sharpValues';
import { ExpNum, ExpShape } from '../backend/symExpressions';
import { LCImpl } from '.';
import { LCBase } from './libcall';

export namespace ShapeLCImpl {
    // parse shape (args), copy it to new SVSize.
    export function setShape(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length < 2) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.shape.setSize': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }

        const heap = ctx.heap;
        const [selfAddr, argsAddr] = params;

        const self = fetchAddr(selfAddr, heap)! as SVObject;
        const args = fetchAddr(argsAddr, heap);

        if (args && args instanceof SVSize) {
            const sizedSelf = SVSize.toSize(self, args.shape);
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

            const sizedSelf = SVSize.toSize(self, shape);
            const newHeap = newCtx.heap.setVal(self.addr, sizedSelf);

            return newCtx.setHeap(newHeap).setRetVal(self);
        });
    }

    // get (shape, axis, repeat_count). returns new shape repeated by repeat_count through axis.
    export function repeat(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx
                .warnSizeWithMsg(
                    `from 'LibCall.shape.repeat': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }

        const heap = ctx.heap;
        const [sizeAddr, axisAddr, countAddr] = params;

        const size = fetchAddr(sizeAddr, heap);
        const axis = fetchAddr(axisAddr, heap);
        const count = fetchAddr(countAddr, heap);

        if (!isSize(size)) {
            return ctx.warnSizeWithMsg(`from 'LibCall.shape.repeat: shape value is not a sized type`, source).toSet();
        }
        if (axis?.type !== SVType.Int) {
            return ctx.warnSizeWithMsg(`from 'LibCall.shape.repeat: axis value is not an integer`, source).toSet();
        } else if (count?.type !== SVType.Int) {
            return ctx.warnSizeWithMsg(`from 'LibCall.shape.repeat: count value is not an integer`, source).toSet();
        }

        const shape = size.shape;
        const rank = ExpShape.getRank(shape);
        const axisVal = axis.value;
        const countVal = count.value;

        const absAxis = absExpIndexByLen(axisVal, rank, source, ctx.ctrSet);

        return ctx.shRepeat(shape, absAxis, countVal, source).map((ctx) => {
            const [newSize, heap] = size.clone(ctx.heap, source);
            return ctx.setHeap(heap).setRetVal(SVSize.toSize(newSize, ctx.retVal));
        });
    }

    // index from shape
    export function index(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.shape.index': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }

        const heap = ctx.heap;
        const [sizeAddr, axisAddr] = params;

        const size = fetchAddr(sizeAddr, heap);
        const axis = fetchAddr(axisAddr, heap);

        if (!isSize(size)) {
            return ctx.warnSizeWithMsg(`from 'LibCall.shape.repeat: shape value is not a sized type`, source).toSet();
        }

        if (axis?.type !== SVType.Int) {
            return ctx.warnSizeWithMsg(`from 'LibCall.shape.repeat: axis value is not an integer`, source).toSet();
        }

        const shape = size.shape;
        const rank = ExpShape.getRank(shape);
        const axisVal = axis.value;
        const absAxis = absExpIndexByLen(axisVal, rank, source, ctx.ctrSet);

        const retVal = simplifyNum(ctx.ctrSet, ExpNum.index(shape, absAxis, source));

        return ctx.toSetWith(SVInt.create(retVal, source));
    }

    // slice from shape
    export function slice(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx
                .warnSizeWithMsg(
                    `from 'LibCall.shape.slice': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }

        const heap = ctx.heap;
        const [sizeAddr, startAddr, endAddr] = params;

        const size = fetchAddr(sizeAddr, heap);
        const start = fetchAddr(startAddr, heap);
        const end = fetchAddr(endAddr, heap);

        if (!isSize(size)) {
            return ctx.warnSizeWithMsg(`from 'LibCall.shape.slice: shape value is not a sized type`, source).toSet();
        }

        if (!(start?.type === SVType.Int || start?.type === SVType.None)) {
            return ctx.warnSizeWithMsg(`from 'LibCall.shape.slice: start value is not an integer`, source).toSet();
        }
        if (!(end?.type === SVType.Int || end?.type === SVType.None)) {
            return ctx.warnSizeWithMsg(`from 'LibCall.shape.slice: end value is not an integer`, source).toSet();
        }

        const shape = size.shape;
        const rank = ExpShape.getRank(shape);
        const startNum = start.type === SVType.Int ? start.value : 0;
        const endNum = end.type === SVType.Int ? end.value : rank;

        const startAbs = absExpIndexByLen(startNum, rank, source, ctx.ctrSet);
        const endAbs = absExpIndexByLen(endNum, rank, source, ctx.ctrSet);

        let finalShape: ExpShape = ExpShape.slice(shape, startAbs, endAbs, source);
        finalShape = simplifyShape(ctx.ctrSet, finalShape);

        const [newSize, newHeap] = size.clone(heap, source);
        return ctx.setHeap(newHeap).toSetWith(SVSize.toSize(newSize, finalShape));
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
        const value = fetchAddr(valueAddr, ctx.heap);
        if (value?.type === SVType.Object) {
            if (value instanceof SVSize) {
                return ctx.toSetWith(value);
            }
            const shape = fetchAddr(value.getAttr('shape'), ctx.heap);
            if (shape?.type === SVType.Object && shape instanceof SVSize) {
                return ctx.toSetWith(shape);
            }
        }

        return buildShape(ctx, valueAddr, source).map((ctx) => SVSize.createSize(ctx, ctx.retVal, source));
    }

    export function buildShape(
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
        return _randShape(ctx, message, source).map((ctx) => SVSize.createSize(ctx, ctx.retVal, source));
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
        return SVSize.createSize(ctx, shape, source).toSet();
    }

    export const libCallImpls: { [key: string]: LCImpl } = {
        setShape,
        repeat,
        index,
        slice,
        extractShape,
        randShape,
    };
}

export const libCallMap: Map<string, LCImpl> = new Map([...Object.entries(ShapeLCImpl.libCallImpls)]);
