import { ParseNode } from 'pyright-internal/parser/parseNodes';

import { LCImpl } from '..';
import { fetchAddr } from '../../backend/backUtils';
import { Context, ContextSet } from '../../backend/context';
import { fetchSize, simplifyShape } from '../../backend/expUtils';
import { ShValue, SVSize, SVNone, SVType } from '../../backend/sharpValues';
import { ExpNum, ExpShape } from '../../backend/symExpressions';
import { LCBase } from '../libcall';

export namespace PILLCImpl {
    export function blend(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.PIL.blend': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [im1Addr, im2Addr, alphaAddr] = params;

        const im1Size = fetchSize(im1Addr, heap);
        const im2Size = fetchSize(im2Addr, heap);
        const alpha = fetchAddr(alphaAddr, heap);

        if (typeof im1Size === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.PIL.blend: ${im1Size}`, source);
        } else if (typeof im2Size === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.PIL.blend: ${im2Size}`, source);
        }
        if (alpha?.type !== SVType.Int && alpha?.type !== SVType.Float) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.PIL.blend: alpha value is not a number.` +
                    ` got ${alpha ? ShValue.toString(alpha) : undefined}`,
                source
            );
        }

        const im1Shape = im1Size.shape;
        const im2Shape = im2Size.shape;

        // Just add constraint.
        // New image returned by Image.blend() is created in python interface(pytea/pylib/PIL/Image.py).
        return ctx
            .require(
                [ctx.genEq(im1Shape, im2Shape, source)],
                `from 'LibCall.PIL.blend: shapes of images must be equal.`,
                source
            )
            .return(SVNone.create());
    }

    export function fromarray(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.PIL.fromarray': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [imageAddr, objAddr, modeAddr] = params;

        const image = fetchAddr(imageAddr, heap);
        const objSize = fetchSize(objAddr, heap);

        if (imageAddr.type !== SVType.Addr || image?.type !== SVType.Object) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.PIL.fromarray': not an object type:\n\t${imageAddr.toString()} -> ${image?.toString()}`,
                    source
                )
                .toSet();
        }
        if (typeof objSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.PIL.fromarray: ${objSize}`, source);
        }

        const shape = objSize.shape;
        const rank = ExpShape.getRank(shape);

        const isRankTwo = ctx.genEq(2, rank, source);
        const [rankTwoPath, rankThreePath] = ctx.ifThenElse(isRankTwo, source);

        const leftPath = rankTwoPath.flatMap((ctx) => {
            // [H, W] -> [1, H, W]
            const newShape = simplifyShape(
                ctx.ctrSet,
                ExpShape.concat(ExpShape.fromConst(1, [1], source), shape, source)
            );

            const newImage = SVSize.fromObject(ctx, image, newShape);
            return ctx.setHeap(ctx.heap.setVal(imageAddr, newImage)).toSetWith(newImage);
        });

        const rightPath = rankThreePath.flatMap((ctx) => {
            // [H, W, C] -> [C, H, W]
            const channel = ExpNum.index(shape, 2, source);
            const shapeC = ExpShape.fromConst(1, [channel], source);
            const shapeHW = ExpShape.slice(shape, 0, 2, source);
            const newShape = simplifyShape(ctx.ctrSet, ExpShape.concat(shapeC, shapeHW));

            return ctx
                .require(
                    [ctx.genEq(3, rank, source)],
                    `from 'LibCall.PIL.fromarray: rank must be 2 or 3. got ${
                        typeof rank === 'number' ? rank : ExpNum.toString(rank)
                    }`,
                    source
                )
                .require(
                    [ctx.genLte(1, channel, source), ctx.genLte(channel, 4, source)],
                    `from 'LibCall.PIL.fromarray: channel must be 1 ~ 4. got ${
                        typeof channel === 'number' ? channel : ExpNum.toString(channel)
                    }`,
                    source
                )
                .map((ctx) => {
                    const newImage = SVSize.fromObject(ctx, image, newShape);
                    return ctx.setHeap(ctx.heap.setVal(imageAddr, newImage)).setRetVal(newImage);
                });
        });

        return leftPath.join(rightPath);
    }

    export const libCallImpls: { [key: string]: LCImpl } = {
        blend,
        fromarray,
    };
}

export const libCallMap: Map<string, LCImpl> = new Map([...Object.entries(PILLCImpl.libCallImpls)]);
