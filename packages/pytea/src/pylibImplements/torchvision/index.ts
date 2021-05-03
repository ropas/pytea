import { fetchAddr } from '../../backend/backUtils';
import { Context, ContextSet } from '../../backend/context';
import { fetchSize, genTensor, simplifyShape } from '../../backend/expUtils';
import { CodeSource, ShValue, SVSize, SVType } from '../../backend/sharpValues';
import { ExpNum, ExpShape, NumBopType } from '../../backend/symExpressions';
import { LCImpl } from '..';
import { LCBase } from '../libcall';

export namespace TorchvisionLCImpl {
    export function crop(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
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

    export function to_pil_image(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torchvision.to_pil_image': got insufficient number of argument: ${params.length}`,
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
                    `from 'LibCall.torchvision.to_pil_image': not an object type:\n\t${imageAddr.toString()} -> ${image?.toString()}`,
                    source
                )
                .toSet();
        }
        if (typeof objSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torchvision.to_pil_image: ${objSize}`, source);
        }

        const shape = objSize.shape;
        const rank = ExpShape.getRank(shape);
        const channel = ExpNum.index(shape, 0, source);

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
            // [C, H, W] -> [C, H, W]
            return ctx
                .require(
                    [ctx.genEq(3, rank, source)],
                    `from 'LibCall.torchvision.to_pil_image: rank must be 2 or 3. got ${
                        typeof rank === 'number' ? rank : ExpNum.toString(rank)
                    }`,
                    source
                )
                .require(
                    [ctx.genLte(1, channel, source), ctx.genLte(channel, 4, source)],
                    `from 'LibCall.torchvision.to_pil_image: channel must be 1 ~ 4. got ${
                        typeof channel === 'number' ? channel : ExpNum.toString(channel)
                    }`,
                    source
                )
                .map((ctx) => {
                    const newImage = SVSize.fromObject(ctx, image, shape);
                    return ctx.setHeap(ctx.heap.setVal(imageAddr, newImage)).setRetVal(newImage);
                });
        });

        return leftPath.join(rightPath);
    }

    export function normalize(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torchvision.normalize': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [tensorAddr, meanLenAddr, stdLenAddr] = params;

        const size = fetchSize(tensorAddr, heap);
        const meanLen = fetchAddr(meanLenAddr, heap);
        const stdLen = fetchAddr(stdLenAddr, heap);
        if (typeof size === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torchvision.normalize': ${size}`, source);
        }
        if (meanLen?.type !== SVType.Int) {
            return ctx
                .warnWithMsg(`from 'LibCall.torchvision.normalize': incorrect len of mean ${meanLen}`, source)
                .toSet();
        }
        if (stdLen?.type !== SVType.Int) {
            return ctx
                .warnWithMsg(`from 'LibCall.torchvision.normalize': incorrect len of std ${stdLen}`, source)
                .toSet();
        }
        const shape = size.shape;
        const rank = size.rank();
        const channel = ExpNum.index(shape, 0, source);

        return ctx
            .require(
                [ctx.genEq(3, rank, source)],
                `from 'LibCall.torchvision.normalize: Expected tensor to be a tensor image of size (C, H, W). Got a tensor whose rank is ${rank}.`,
                source
            )
            .require(
                [ctx.genOr(ctx.genEq(1, meanLen.value, source), ctx.genEq(channel, meanLen.value, source), source)],
                `from 'LibCall.torchvision.normalize: Lengths of mean must be 1 or equal to channel. Got channel: ${channel}, mean: ${meanLen.value}.`,
                source
            )
            .require(
                [ctx.genOr(ctx.genEq(1, stdLen.value, source), ctx.genEq(channel, stdLen.value, source), source)],
                `from 'LibCall.torchvision.normalize: Lengths of std must be 1 or equal to channel. Got channel: ${channel}, std: ${stdLen.value}.`,
                source
            )
            .flatMap((ctx) => genTensor(ctx, shape, source));
    }

    export const libCallImpls: { [key: string]: LCImpl } = {
        crop,
        to_pil_image,
        normalize,
    };
}

export const libCallMap: Map<string, LCImpl> = new Map([...Object.entries(TorchvisionLCImpl.libCallImpls)]);
