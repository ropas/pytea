import { ParseNode } from 'pyright-internal/parser/parseNodes';

import { LCImpl } from '..';
import { fetchAddr } from '../../backend/backUtils';
import { Context, ContextSet } from '../../backend/context';
import { fetchSize, genTensor } from '../../backend/expUtils';
import { ShValue, SVType, SVNone } from '../../backend/sharpValues';
import { ExpNum, ExpShape, NumBopType } from '../../backend/symExpressions';
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

    export function crop(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torchvision.crop': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [tensorAddr, heightAddr, widthAddr] = params;

        const tensorSize = fetchSize(tensorAddr, heap);
        const height = fetchAddr(heightAddr, heap);
        const width = fetchAddr(widthAddr, heap);

        if (typeof tensorSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torchvision.crop: ${tensorSize}`, source);
        }

        // TODO: height or width is warning?
        if (height?.type !== SVType.Int) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torchvision.crop: height value is not an integer.` +
                    ` got ${height ? ShValue.toString(height) : undefined}`,
                source
            );
        } else if (width?.type !== SVType.Int) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torchvision.crop: width value is not an integer.` +
                    ` got ${width ? ShValue.toString(width) : undefined}`,
                source
            );
        }

        const shape = tensorSize.shape;
        const rank = ExpShape.getRank(shape);

        return ctx
            .require(
                [ctx.genLte(0, height.value, source), ctx.genLte(0, width.value, source)],
                `from 'LibCall.torchvision.crop: height or width value can be negative.`,
                source
            )
            .require(
                ctx.genLte(2, rank, source),
                `from 'LibCall.torchvision.crop: rank is less than 2. got ${
                    typeof rank === 'number' ? rank : ExpNum.toString(rank)
                }`,
                source
            )
            .flatMap((ctx) => {
                let newShape: ExpShape = ExpShape.slice(
                    shape,
                    undefined,
                    ExpNum.bop(NumBopType.Sub, rank, 2, source),
                    source
                );
                newShape = ExpShape.concat(
                    newShape,
                    ExpShape.fromConst(2, [height.value, width.value], source),
                    source
                );
                return genTensor(ctx, newShape, source);
            });
    }

    export const libCallImpls: { [key: string]: LCImpl } = {
        crop,
    };
}

export const libCallMap: Map<string, LCImpl> = new Map([...Object.entries(PILLCImpl.libCallImpls)]);
