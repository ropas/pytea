import { ParseNode } from 'pyright-internal/parser/parseNodes';

import { LCImpl } from '..';
import { fetchAddr } from '../../../backend/backUtils';
import { Context, ContextSet } from '../../../backend/context';
import { fetchSize, genTensor } from '../../../backend/expUtils';
import { ShValue, SVType } from '../../../backend/sharpValues';
import { ExpNum, ExpShape, NumBopType } from '../../../backend/symExpressions';
import { LCBase } from '../libcall';

export namespace TorchvisionLCImpl {
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

export const libCallMap: Map<string, LCImpl> = new Map([...Object.entries(TorchvisionLCImpl.libCallImpls)]);
