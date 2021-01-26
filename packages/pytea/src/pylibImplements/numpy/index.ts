import { ParseNode } from 'pyright-internal/parser/parseNodes';

import { LCImpl } from '..';
import { fetchAddr } from '../../backend/backUtils';
import { Constraint } from '../../backend/constraintType';
import { ConstraintSet } from '../../backend/constraintSet';
import { Context, ContextSet } from '../../backend/context';
import { absExpIndexByLen, fetchSize, isSize, simplifyBool, simplifyNum, simplifyShape } from '../../backend/expUtils';
import { ShValue, SVAddr, SVObject, SVSize, SVNone, SVType } from '../../backend/sharpValues';
import { BoolOpType, ExpBool, ExpNum, ExpShape, NumBopType, NumOpType } from '../../backend/symExpressions';
import { TorchBackend } from '../../backend/torchBackend';
import { LCBase } from '../libcall';

export namespace NumpyLCImpl {
    export function ndarrayInit(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 7) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.numpy.ndarrayInit': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }

        const heap = ctx.heap;
        const [selfAddr, shapeAddr, dtypeAddr] = params;

        // TODO: use dtype

        // ndarrayInit is always used in ndarray.__init__ -> force casting
        const addr = selfAddr as SVAddr;
        const self = fetchAddr(selfAddr, heap)! as SVObject;
        const shape = fetchAddr(shapeAddr as SVAddr, heap)! as SVObject;

        // if args is object that has 'shape'
        if (shape?.type === SVType.Object) {
            let size: ShValue | undefined = shape;
            if (size.shape === undefined) {
                size = fetchAddr(shape.getAttr('shape'), heap);
            }

            if (size && size.type === SVType.Object && size?.shape !== undefined) {
                const newHeap = heap.setVal(addr, self.setAttr('shape', size));
                return ctx.setHeap(newHeap).setRetVal(SVNone.create()).toSet();
            }
        }

        // if args is list of integer
        return ctx.parseSize(shape, source).map((ctx) => {
            let shape: ExpShape;
            let newCtx: Context<any> = ctx;
            if (typeof ctx.retVal === 'string') {
                newCtx = ctx.addLog(ctx.retVal, source).genIntGte('tempRank', 0, source);
                shape = ExpShape.fromSymbol(newCtx.genSymShape('tempShape', newCtx.retVal, source));
            } else {
                shape = ctx.retVal;
            }

            const size = SVSize.createSize(ctx, shape, source);
            const newHeap = heap.setVal(addr, self.setAttr('shape', size));

            return newCtx.setHeap(newHeap);
        });
    }

    // get `(arrAddr: ndarray, imgAddr: PIL Image)`, set shape of `arr` to SVSize with shape of `img`
    export function fromImage(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.numpy.fromImage': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }
        const heap = ctx.heap;
        const [arrAddr, imgAddr] = params;
        const arrObj = fetchAddr(arrAddr, heap);
        const arrSize = fetchSize(arrAddr, heap);
        const imgSize = fetchAddr(imgAddr, heap);

        if (arrAddr.type !== SVType.Addr || arrObj?.type !== SVType.Object) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.numpy.fromImage': not an object type:\n\t${arrAddr.toString()} -> ${arrObj?.toString()}`,
                    source
                )
                .toSet();
        } else if (typeof arrSize === 'string') {
            return ctx.warnWithMsg(`from 'LibCall.numpy.fromImage': ${arrSize}`, source).toSet();
        } else if (imgAddr.type !== SVType.Addr || !isSize(imgSize)) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.numpy.fromImage': not a size type:\n\t${imgAddr.toString()} -> ${imgSize?.toString()}`,
                    source
                )
                .toSet();
        }

        const newSize = SVSize.fromObject(ctx, arrSize, imgSize.shape);
        const newArr = arrObj.setAttr('shape', newSize);

        const newHeap = heap.setVal(arrAddr, newArr);
        return ctx.setHeap(newHeap).toSetWith(SVNone.create());
    }

    // Assumption: "tensors" is a constantRanked sequence, and each element is available.
    // TODO: handle empty tensor.
    export function concatenate(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.numpy.concatenate': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }

        const heap = ctx.heap;
        const [seqAddr, axisAddr] = params;

        const seq = fetchAddr(seqAddr, heap);
        const axis = fetchAddr(axisAddr, heap);

        if (seq?.type !== SVType.Object) {
            return ctx.warnWithMsg(`from 'LibCall.numpy.concatenate': array sequence is not iterable`, source).toSet();
        } else if (axis?.type !== SVType.Int) {
            return ctx.warnWithMsg(`from 'LibCall.numpy.concatenate': axis is not an integer`, source).toSet();
        }

        // Assumption: length of "tensors" is constant.
        const seqLen_ = fetchAddr(seq.getAttr('$length'), heap);
        if (!(seqLen_?.type === SVType.Int && typeof seqLen_.value === 'number')) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.numpy.concatenate': length of array sequence is unknown, cannot iterate.`,
                    source
                )
                .toSet();
        } else if (seqLen_.value === 0) {
            return ctx
                .warnWithMsg(`from 'LibCall.numpy.concatenate': length of array sequence is zero.`, source)
                .toSet();
        }
        const seqLen = seqLen_.value;

        const size0 = fetchSize(seq.getIndice(0), heap);
        if (typeof size0 === 'string') {
            return ctx.warnWithMsg(`from 'LibCall.numpy.concatenate': ${size0}`, source).toSet();
        }
        const size0shape = size0.shape;
        const size0rank = size0.rank();
        const shape0Front = ExpShape.slice(size0shape, 0, axis.value, source);
        const shape0Back = ExpShape.slice(size0shape, ExpNum.bop(NumBopType.Add, axis.value, 1), size0rank, source);

        // TODO: handle negative index.
        const ctrs: Constraint[] = [ctx.genLte(0, axis.value, source), ctx.genLt(axis.value, size0rank)];
        let thickness: ExpNum = ExpNum.index(size0shape, axis.value, source);

        for (let i = 1; i < seqLen; i++) {
            const sizeI = fetchSize(seq.getIndice(i), heap);
            if (typeof sizeI === 'string') {
                return ctx.warnTensorWithMsg(`from 'LibCall.numpy.concatenate': ${sizeI}`, source);
            }
            const sizeIshape = sizeI.shape;
            const shapeIFront = ExpShape.slice(sizeIshape, 0, axis.value, source);
            const shapeIBack = ExpShape.slice(sizeIshape, ExpNum.bop(NumBopType.Add, axis.value, 1), size0rank, source);

            ctrs.push(ctx.genEq(shape0Front, shapeIFront, source));
            ctrs.push(ctx.genEq(shape0Back, shapeIBack, source));
            thickness = ExpNum.bop(NumBopType.Add, thickness, ExpNum.index(sizeIshape, axis.value, source));
        }

        const shapeThick = ExpShape.fromConst(1, [thickness], source);
        const returnShape_ = ExpShape.concat(shape0Front, shapeThick, source);
        const returnShape = ExpShape.concat(returnShape_, shape0Back, source);

        return ctx
            .require(ctrs, `from 'LibCall.numpy.concatenate': shapes must match, axis must be within rank`, source)
            .flatMap((ctx) => {
                return genNdarray(ctx, returnShape);
            });
    }

    export function copyOut(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.numpy.copyOUt': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }

        const heap = ctx.heap;
        const [arrayAddr, outAddr] = params;
        const array = fetchSize(arrayAddr, heap);

        if (outAddr.type === SVType.None) {
            return ctx.toSetWith(outAddr);
        }
        const out = fetchSize(outAddr, heap);
        if (typeof array === 'string') {
            return ctx.warnWithMsg(`from 'LibCall.numpy.copyOut': ${array}`, source).toSet();
        }
        if (typeof out === 'string') {
            return ctx.warnWithMsg(`from 'LibCall.numpy.copyOut': ${out}`, source).toSet();
        }

        return ctx
            .require(
                ctx.genEq(array.shape, out.shape, source),
                `from 'LibCall.numpy.copyOut': shapes must be equal`,
                source
            )
            .return(SVNone.create(source));
    }

    export function max(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 4) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.numpy.max': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [selfAddr, axisAddr, outAddr, keepdimsAddr] = params;

        const selfSize = fetchSize(selfAddr, heap);
        if (typeof selfSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.numpy.max': ${selfSize}`, source);
        }
        const selfShape = selfSize.shape;
        const selfRank = selfSize.rank();
        let rankValue: number | undefined = undefined;

        if (typeof selfRank === 'number') {
            rankValue = selfRank;
        } else {
            const selfRank_ = simplifyNum(ctx.ctrSet, selfRank);
            if (selfRank_.opType === NumOpType.Const) {
                rankValue = selfRank_.value;
            }
        }

        const axis = fetchAddr(axisAddr, heap);
        const keepdims = fetchAddr(keepdimsAddr, heap);

        if (
            axis === undefined ||
            (axis.type !== SVType.None && axis.type !== SVType.Int && axis.type !== SVType.Object)
        ) {
            return ctx.failWithMsg(`from 'LibCall.numpy.max': invalid type of axis ${axis?.type}`, source).toSet();
        } else if (keepdims === undefined || keepdims.type !== SVType.Bool) {
            return ctx
                .failWithMsg(`from 'LibCall.numpy.max': invalid type of keepdims ${keepdims?.type}`, source)
                .toSet();
        }

        let keepdimsVal: boolean;
        if (typeof keepdims.value === 'boolean') {
            keepdimsVal = keepdims.value;
        } else {
            const keepdims_: ExpBool = simplifyBool(ctx.ctrSet, keepdims.value);
            if (keepdims_.opType !== BoolOpType.Const) {
                return ctx.warnTensorWithMsg(
                    `from 'LibCall.numpy.max': cannot infer value of keepdims ${keepdims.value}`,
                    source
                );
            }
            keepdimsVal = keepdims_.value;
        }

        // 1) axis is None. return a scalar value.
        if (axis.type === SVType.None) {
            if (keepdimsVal) {
                if (rankValue !== undefined) {
                    let dims: number[] = [];
                    for (let i = 0; i < rankValue; i++) {
                        dims.push(1);
                    }
                    const shape = ExpShape.fromConst(rankValue, dims, source);
                    return genNdarray(ctx, shape, source);
                } else {
                    const dim = ctx.genSymInt('dim', source);
                    const ctrEq = ctx.genEq(ExpNum.fromSymbol(dim), 1, source);
                    return ctx.require(ctx.genForall(dim, [0, selfRank], ctrEq, source)).flatMap((ctx) => {
                        return ctx.genRankedShape(selfRank, source).flatMap((ctx) => {
                            const rankedShape = ctx.retVal;
                            return genNdarray(ctx, rankedShape, source);
                        });
                    });
                }
            } else {
                // TODO: data type
                return genNdarray(ctx, ExpShape.fromConst(0, [], source), source);
            }
        }
        // 2) axis is an integer.
        else if (axis.type === SVType.Int) {
            const axisVal = absExpIndexByLen(axis.value, selfRank, source, ctx.ctrSet);

            const shapeFront = ExpShape.slice(selfShape, 0, axisVal, source);
            const shapeBack = ExpShape.slice(
                selfShape,
                ExpNum.bop(NumBopType.Add, axisVal, 1, source),
                selfRank,
                source
            );

            let newShape: ExpShape;
            if (keepdimsVal) {
                const newDim = ExpShape.fromConst(1, [1], source);
                const newShape_ = ExpShape.concat(shapeFront, newDim, source);
                newShape = ExpShape.concat(newShape_, shapeBack, source);
            } else {
                newShape = ExpShape.concat(shapeFront, shapeBack, source);
            }
            return ctx
                .require(
                    [ctx.genLte(0, axisVal, source), ctx.genLt(axisVal, selfRank, source)],
                    `from 'LibCall.numpy.max': axis must be within rank`,
                    source
                )
                .flatMap((ctx) => {
                    return genNdarray(ctx, newShape, source);
                });
        }
        // 3) axis is a tuple of ints.
        else {
            let axes = getExpNumTuple(axis, ctx.ctrSet);
            if (typeof axes === 'string') {
                return ctx.failWithMsg(`from 'LibCall.numpy.max': ${axes}`, source).toSet();
            }

            const constAxes: number[] = [];
            axes.forEach((axis) => {
                const axis_ = absExpIndexByLen(axis, selfRank, source);
                if (typeof axis_ === 'number') {
                    constAxes.push(axis_);
                }
            });
            if (axes.length !== constAxes.length) {
                return ctx.failWithMsg(`from 'LibCall.numpy.max': ${axes} has non-const axis`, source).toSet();
            }
            constAxes.sort();

            const shapes: ExpShape[] = [];
            let lastDim: number | ExpNum = -1;
            for (let i = 0; i < constAxes.length; i++) {
                let dim = constAxes[i];
                shapes.push(ExpShape.slice(selfShape, ExpNum.bop(NumBopType.Add, lastDim, 1, source), dim, source));
                if (keepdimsVal) {
                    shapes.push(ExpShape.fromConst(1, [1], source));
                }
                lastDim = dim;
            }
            shapes.push(ExpShape.slice(selfShape, ExpNum.bop(NumBopType.Add, lastDim, 1, source), selfRank, source));
            const shape = shapes.reduce((left, right) => ExpShape.concat(left, right, source));
            return ctx
                .require(
                    [ctx.genLte(0, constAxes[0], source), ctx.genLt(constAxes[constAxes.length - 1], selfRank, source)],
                    `axes must be within rank`,
                    source
                )
                .flatMap((ctx) => {
                    return genNdarray(ctx, shape, source);
                });
        }
    }

    function genNdarray<T>(ctx: Context<T>, shape: ExpShape, source?: ParseNode): ContextSet<ShValue> {
        const newShape = simplifyShape(ctx.ctrSet, shape);

        return TorchBackend.libClassInit(ctx, 'numpy.ndarray', [SVSize.createSize(ctx, newShape, source)], source);
    }

    // return tuple of ExpNums from SVObject.
    function getExpNumTuple(obj: SVObject, ctrSet: ConstraintSet): (number | ExpNum)[] | string {
        let length = obj.getAttr('$length');
        if (length === undefined || !(length.type === SVType.Int)) {
            return `attribute '$length' is not an int`;
        }

        let len = typeof length.value === 'number' ? length.value : simplifyNum(ctrSet, length.value);
        if (typeof len !== 'number' && !(len.opType === NumOpType.Const)) {
            return `length is not const`;
        }

        len = typeof len === 'number' ? len : len.value;
        const intTuple: (number | ExpNum)[] = [];
        for (let i = 0; i < len; i++) {
            const elem = obj.getIndice(i);
            if (elem === undefined || elem.type !== SVType.Int) {
                return `an element of tuple is not an int`;
            }
            let num = elem.value;
            intTuple.push(num);
        }
        return intTuple;
    }

    export const libCallImpls: { [key: string]: LCImpl } = {
        ndarrayInit,
        fromImage,
        concatenate,
        copyOut,
        max,
    };
}

export const libCallMap: Map<string, LCImpl> = new Map([...Object.entries(NumpyLCImpl.libCallImpls)]);
