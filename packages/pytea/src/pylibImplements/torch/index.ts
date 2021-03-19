import { fetchAddr } from '../../backend/backUtils';
import { Constraint } from '../../backend/constraintType';
import { Context, ContextSet } from '../../backend/context';
import { ceilDiv, fetchSize, genTensor, simplifyNum } from '../../backend/expUtils';
import {
    CodeSource,
    ShValue,
    SVAddr,
    SVBool,
    SVFloat,
    SVInt,
    SVNone,
    SVNotImpl,
    SVObject,
    SVSize,
    SVString,
    SVType,
} from '../../backend/sharpValues';
import {
    ExpBool,
    ExpNum,
    ExpShape,
    ExpString,
    NumBopType,
    NumOpType,
    NumUopType,
    ShapeOpType,
} from '../../backend/symExpressions';
import { TorchBackend } from '../../backend/torchBackend';
import { LCImpl } from '..';
import { LCBase } from '../libcall';

export namespace TorchLCImpl {
    export function tensorInit(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.tensorInit': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [selfAddr, argsAddr] = params;

        // TODO: use kwargs info.

        // tensorInit is always used in Tensor.__init__ -> force casting
        const addr = selfAddr as SVAddr;
        const self = fetchAddr(selfAddr, heap)! as SVObject;
        const args = fetchAddr(argsAddr as SVAddr, heap)! as SVObject;

        // if first argument is object that has 'shape'
        const firstArg = fetchAddr(args.getIndice(0), heap);
        if (firstArg?.type === SVType.Object) {
            // if first argument is shaped value, cast it to Tensor
            let mayShaped: ShValue | undefined = firstArg;
            if (!mayShaped.shape) {
                mayShaped = fetchAddr(firstArg.getAttr('shape'), heap);
            }
            if (mayShaped?.type === SVType.Object && mayShaped?.shape !== undefined) {
                const size = SVSize.createSize(ctx, mayShaped.shape, source);
                const newHeap = heap.setVal(addr, self.setAttr('shape', size));
                return ctx.setHeap(newHeap).setRetVal(SVNone.create()).toSet();
            }

            // else, check value is list of ... list of number
            const structure: (number | ExpNum)[] = [];
            let obj: ShValue | undefined = firstArg;
            let length = fetchAddr(firstArg.getAttr('$length'), heap);
            let shaped = true;

            if (length && length.type === SVType.Int) {
                // if argument is list of ... list of number, return that shape
                structure.push(length.value);
                obj = fetchAddr(obj.getIndice(0), heap);

                // simply fetch only first values
                while (obj?.type === SVType.Object) {
                    length = fetchAddr(obj.getAttr('$length'), heap);
                    if (length?.type === SVType.Int) {
                        structure.push(length.value);
                        obj = fetchAddr(obj.getIndice(0), heap);
                    } else {
                        shaped = false;
                        break;
                    }
                }

                // traversed list and ends with integer or float
                if (shaped && (obj?.type === SVType.Int || obj?.type === SVType.Float)) {
                    const size = SVSize.createSize(ctx, ExpShape.fromConst(structure.length, structure, source));
                    const newHeap = heap.setVal(addr, self.setAttr('shape', size));
                    return ctx.setHeap(newHeap).setRetVal(SVNone.create()).toSet();
                }
            }
        }

        // if varargs is list of integer
        return ctx.parseSize(args, source).map((ctx) => {
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

    export function identityShape(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 1) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.identityShape': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const inputAddr = params[0];

        const inputSize = fetchSize(inputAddr, heap);

        if (typeof inputSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.identityShape': ${inputSize}`, source);
        }

        const inputShape = inputSize.shape;

        return genTensor(ctx, inputShape, source);
    }

    // return broadcasted tensor
    export function broadcast(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.broadcast': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [leftAddr, rightAddr] = params;

        const leftSize = fetchSize(leftAddr, heap);
        const rightSize = fetchSize(rightAddr, heap);

        if (typeof leftSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.broadcast': ${leftSize}`, source);
        } else if (typeof rightSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.broadcast': ${rightSize}`, source);
        }

        const leftShape = leftSize.shape;
        const rightShape = rightSize.shape;

        return ctx.shBroadcast(leftShape, rightShape, source).flatMap((ctx) => genTensor(ctx, ctx.retVal, source));
    }

    // implementation of torch.matmul
    export function matmul(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.matmul': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [leftAddr, rightAddr] = params;

        const leftSize = fetchSize(leftAddr, heap);
        const rightSize = fetchSize(rightAddr, heap);

        if (typeof leftSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.matmul': ${leftSize}`, source);
        } else if (typeof rightSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.matmul': ${rightSize}`, source);
        }

        const leftShape = leftSize.shape;
        const rightShape = rightSize.shape;
        const leftRank = leftSize.rank();
        const rightRank = rightSize.rank();

        return ctx
            .require(
                [ctx.genLte(1, leftRank, source), ctx.genLte(1, rightRank, source)],
                `from 'LibCall.torch.matmul': cannot do matmul with scalar tensor`,
                source
            )
            .flatMap((ctx) => {
                const isLeftRankOne = ctx.genEq(1, leftRank, source);
                const [leftOnePath, leftTwoPath] = ctx.ifThenElse(isLeftRankOne, source);

                const leftPath = leftOnePath.flatMap((ctx) => {
                    const isRightRankOne = ctx.genEq(1, rightRank, source);
                    const [rightOnePath, rightTwoPath] = ctx.ifThenElse(isRightRankOne, source);

                    const lr11 = rightOnePath
                        .flatMap((ctx) => {
                            const sameDim = ctx.genEq(ExpNum.index(leftShape, 0), ExpNum.index(rightShape, 0), source);
                            return ctx.require(
                                [sameDim],
                                `from 'LibCall.torch.matmul': dimension mismatch between rank-1 tensors`,
                                source
                            );
                        })
                        .flatMap((ctx) => genTensor(ctx, ExpShape.fromConst(0, [], source), source));

                    const rightAxis = ExpNum.bop(NumBopType.Sub, rightRank, 2);
                    const lr12 = rightTwoPath
                        .flatMap((ctx) => {
                            const sameDim = ctx.genEq(
                                ExpNum.index(leftShape, 0),
                                ExpNum.index(rightShape, rightAxis),
                                source
                            );
                            return ctx.require(
                                [sameDim],
                                `from 'LibCall.torch.matmul: dimension mismatch between rank-1 @ rank-n`
                            );
                        })
                        .flatMap((ctx) => ctx.shReduce(rightShape, rightAxis, source))
                        .flatMap((ctx) => genTensor(ctx, ctx.retVal, source));

                    return lr11.join(lr12);
                });

                const rightPath = leftTwoPath.flatMap((ctx) => {
                    const isRightRankOne = ctx.genEq(1, rightRank, source);
                    const [rightOnePath, rightTwoPath] = ctx.ifThenElse(isRightRankOne, source);

                    const leftAxis = ExpNum.bop(NumBopType.Sub, leftRank, 1, source);
                    const lr21 = rightOnePath
                        .flatMap((ctx) => {
                            const sameDim = ctx.genEq(
                                ExpNum.index(leftShape, leftAxis),
                                ExpNum.index(rightShape, 0),
                                source
                            );
                            return ctx.require(
                                [sameDim],
                                `from 'LibCall.torch.matmul': dimension mismatch between rank-n @ rank-1`,
                                source
                            );
                        })
                        .flatMap((ctx) => ctx.shReduce(leftShape, leftAxis, source))
                        .flatMap((ctx) => genTensor(ctx, ctx.retVal, source));

                    const lr22 = rightTwoPath
                        .flatMap((ctx) => ctx.shMatmul(leftShape, rightShape, source))
                        .flatMap((ctx) => genTensor(ctx, ctx.retVal, source));

                    return lr21.join(lr22);
                });

                return leftPath.join(rightPath);
            });
    }

    // implementation of torch.mm
    export function mm(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.mm': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [leftAddr, rightAddr] = params;

        const leftSize = fetchSize(leftAddr, heap);
        const rightSize = fetchSize(rightAddr, heap);

        if (typeof leftSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.mm': ${leftSize}`, source);
        } else if (typeof rightSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.mm': ${rightSize}`, source);
        }

        const leftShape = leftSize.shape;
        const rightShape = rightSize.shape;
        const leftRank = leftSize.rank();
        const rightRank = rightSize.rank();

        return ctx
            .require(
                [ctx.genEq(2, leftRank, source), ctx.genEq(2, rightRank, source)],
                `from 'LibCall.torch.mm': input must be 2-D tensors`,
                source
            )
            .require(
                ctx.genEq(ExpNum.index(leftShape, 1, source), ExpNum.index(rightShape, 0, source)),
                `from 'LibCall.torch.mm': dimension mismatch`
            )
            .flatMap((ctx) => {
                const newShape = ExpShape.fromConst(
                    2,
                    [ExpNum.index(leftShape, 0, source), ExpNum.index(rightShape, 1, source)],
                    source
                );
                return genTensor(ctx, newShape, source);
            });
    }

    // implementation of torch.mm
    export function bmm(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.bmm': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [leftAddr, rightAddr] = params;

        const leftSize = fetchSize(leftAddr, heap);
        const rightSize = fetchSize(rightAddr, heap);

        if (typeof leftSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.bmm': ${leftSize}`, source);
        } else if (typeof rightSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.bmm': ${rightSize}`, source);
        }

        const leftShape = leftSize.shape;
        const rightShape = rightSize.shape;
        const leftRank = leftSize.rank();
        const rightRank = rightSize.rank();
        const leftBatch = ExpNum.index(leftShape, 0, source);
        const rightBatch = ExpNum.index(rightShape, 0, source);

        return ctx
            .require(
                [ctx.genEq(3, leftRank, source), ctx.genEq(3, rightRank, source)],
                `from 'LibCall.torch.bmm': input must be 3-D tensors`,
                source
            )
            .require(
                [ctx.genEq(leftBatch, rightBatch, source)],
                `from 'LibCall.torch.bmm': batch size mismatch`,
                source
            )
            .require(
                ctx.genEq(ExpNum.index(leftShape, 2, source), ExpNum.index(rightShape, 1, source)),
                `from 'LibCall.torch.mm': dimension mismatch`
            )
            .flatMap((ctx) => {
                const simplerBatch = leftShape.opType === ShapeOpType.Const ? leftBatch : rightBatch;
                const newShape = ExpShape.fromConst(
                    3,
                    [simplerBatch, ExpNum.index(leftShape, 1, source), ExpNum.index(rightShape, 2, source)],
                    source
                );
                return genTensor(ctx, newShape, source);
            });
    }

    export function item(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 1) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.item': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const selfAddr = params[0];

        const self = fetchAddr(selfAddr, heap);
        const selfSize = fetchSize(selfAddr, heap);

        if (self?.type !== SVType.Object) {
            return ctx.failWithMsg(`from 'LibCall.torch.item': not a tensor object`, source).toSet();
        }
        if (typeof selfSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.item': ${selfSize}`, source);
        }

        const selfShape = selfSize.shape;
        const selfRank = selfSize.rank();

        // Tensor.dtype is always given. force casting.
        const dtypeClass = fetchAddr(self.getAttr('dtype'), heap) as SVObject;
        const dtypeName = dtypeClass.getAttr('__name__') as SVString;
        const dtype = dtypeName.value as string;

        const isFloat = dtype === 'float16' || dtype === 'float32' || dtype === 'float64';
        const isInt =
            dtype === 'int8' || dtype === 'int16' || dtype === 'int32' || dtype === 'int64' || dtype === 'uint8';
        const isBool = dtype === 'bool';

        const ctxSet = ctx.require(
            [ctx.genOr(ctx.genEq(0, selfRank, source), ctx.genEq(1, ExpNum.numel(selfShape, source)), source)],
            `from 'LibCall.torch.item': tensor must have exacly one element`,
            source
        );

        if (isFloat) {
            return ctxSet.return(SVFloat.create(ExpNum.fromSymbol(ctx.genSymFloat('torchItem', source)), source));
        } else if (isInt) {
            return ctxSet.return(SVInt.create(ExpNum.fromSymbol(ctx.genSymInt('torchItem', source)), source));
        } else if (isBool) {
            return ctxSet.return(SVBool.create(ExpBool.fromSymbol(ctx.genSymBool('torchItem', source)), source));
        } else {
            return ctx.failWithMsg(`from 'LibCall.torch.item': unknown dtype of tensor`, source).toSet();
        }
    }

    // implementation of torch.Tensor.repeat
    export function repeat(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.repeat': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [selfAddr, sizes] = params;

        const selfSize = fetchSize(selfAddr, heap);
        const repeatSizes = fetchAddr(sizes, heap);

        if (typeof selfSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.repeat': ${selfSize}`, source);
        } else if (repeatSizes?.type !== SVType.Object) {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.repeat': sizes is not iterable`, source);
        }

        const selfShape = selfSize.shape;
        const selfRank = selfSize.rank();

        const tupleLenFetch = fetchAddr(repeatSizes.getAttr('$length'), heap);
        let sizeObj = fetchAddr(repeatSizes.getIndice(0), heap); // temporarily get first value

        if (!(sizeObj && tupleLenFetch && tupleLenFetch.type === SVType.Int)) {
            return ctx.failWithMsg(`from 'LibCall.torch.repeat': sizes is not iterable`, source).toSet();
        }

        let tupleLenObj = tupleLenFetch.value;
        if (sizeObj.type === SVType.Object) {
            // first argument is object
            tupleLenObj = -1;
        } else if (typeof tupleLenObj === 'number' && tupleLenObj >= 2) {
            // size is given as vararg
            sizeObj = repeatSizes;
        } else if (sizeObj.type === SVType.Int) {
            // single dimension repeat
            const [size, sizeAddr, newHeap] = SVObject.create(heap, source);
            sizeObj = size.setIndice(0, sizeObj).setAttr('$length', SVInt.create(1, source));
            ctx = ctx.setHeap(newHeap.setVal(sizeAddr, sizeObj));
        }

        return ctx.parseSize(sizeObj, source).flatMap((ctx) => {
            const sizes = ctx.retVal;
            if (typeof sizes === 'string') {
                return ctx.warnTensorWithMsg(sizes, source);
            }

            const sizeRank = ExpShape.getRank(sizes);

            return ctx
                .require(
                    ctx.genLte(selfRank, sizeRank, source),
                    `from 'LibCall.torch.repeat': Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor`,
                    source
                )
                .flatMap((ctx) => {
                    const selfRankRng = ctx.getCachedRange(selfRank);

                    if (!selfRankRng) {
                        return ctx.failWithMsg(`from 'LibCall.torch.repeat': invalid 'self' rank`, source).toSet();
                    }
                    if (selfRankRng.isConst()) {
                        const selfRank = selfRankRng.end;
                        let shape = sizes;
                        for (let i = 0; i < selfRank; i++) {
                            const targetAxis = ExpNum.bop(NumBopType.Sub, sizeRank, i + 1, source);
                            const newDim = ExpNum.bop(
                                NumBopType.Mul,
                                ExpNum.index(sizes, targetAxis, source),
                                ExpNum.index(selfShape, selfRank - 1 - i, source),
                                source
                            );
                            shape = ExpShape.setDim(shape, targetAxis, newDim);
                        }

                        return genTensor(ctx, shape, source);
                    } else {
                        // if rank of self is not inferable, we have no choice but to make symbolic shape like broadcasting
                        // that is right-aligned to self.shape and sizes, all the dimensions is gte than sizes
                        return ctx.genRankedShape(sizeRank, source).flatMap((ctx) => {
                            const symShape = ctx.retVal;
                            const rankRng = ctx.getCachedRange(sizeRank);

                            if (!rankRng) {
                                return ctx.failWithMsg(`from 'LibCall.torch.repeat': invalid length of sizes`).toSet();
                            }

                            let newCtx: Context<any> = ctx;
                            // set dimension lower bound
                            for (let i = 0; i < rankRng.start; i++) {
                                const dim = ExpNum.index(sizes, i, source);
                                newCtx = newCtx.guarantee(
                                    newCtx.genLte(dim, ExpNum.index(symShape, i, source), source)
                                );
                            }

                            return genTensor(newCtx, symShape, source);
                        });
                    }
                });
        });
    }

    export function copyOut(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx
                .warnWithMsg(`LibCall.torch.copyOut got insufficient number of argument: ${params.length}`, source)
                .toSet();
        }

        const [tensorAddr, out] = params;
        if (out.type === SVType.None) {
            return ctx.toSetWith(out);
        }

        if (out.type !== SVType.Addr) {
            return ctx
                .warnWithMsg(`LibCall.torch.copyOut type error: out is not an address - got ${out.type}`, source)
                .toSet();
        }

        const heap = ctx.heap;
        const tensor = fetchAddr(tensorAddr, heap);

        return (tensor ? ctx.setHeap(heap.setVal(out, tensor)) : ctx).setRetVal(out).toSet();
    }

    export function callTensor(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
        return TorchBackend.libClassInit(ctx, 'torch.Tensor', ctx.retVal.params, source);
    }

    export function transpose(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.transpose': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [selfAddr, dim0Addr, dim1Addr] = params;

        const selfSize = fetchSize(selfAddr, heap);

        if (typeof selfSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.transpose': ${selfSize}`, source);
        }

        const selfShape = selfSize.shape;
        const dim0 = fetchAddr(dim0Addr, heap);
        const dim1 = fetchAddr(dim1Addr, heap);

        if (dim0?.type !== SVType.Int) {
            return ctx.failWithMsg(`from 'LibCall.torch.transpose': cannot infer dim0 as integer`, source).toSet();
        }
        if (dim1?.type !== SVType.Int) {
            return ctx.failWithMsg(`from 'LibCall.torch.transpose': cannot infer dim1 as integer`, source).toSet();
        }

        const selfRank = ExpShape.getRank(selfShape);
        const negRank = ExpNum.bop(NumBopType.Sub, 0, selfRank);

        const [dim0Pos, dim0Neg] = ctx
            .require(
                [
                    ctx.genLte(negRank, dim0.value, source),
                    ctx.genLt(dim0.value, selfRank, source),
                    ctx.genLte(negRank, dim1.value, source),
                    ctx.genLt(dim1.value, selfRank, source),
                ],
                `from 'LibCall.torch.transpose': dimension out of range`,
                source
            )
            .ifThenElse(ctx.genLte(0, dim0.value, source), source);

        const [dimPP, dimPN] = dim0Pos.ifThenElse(ctx.genLte(0, dim1.value, source), source);
        const [dimNP, dimNN] = dim0Neg.ifThenElse(ctx.genLte(0, dim1.value, source), source);

        const dimPPNext = dimPP.flatMap((ctx) =>
            genTensor(
                ctx,
                ExpShape.setDim(
                    ExpShape.setDim(selfShape, dim0.value, ExpNum.index(selfShape, dim1.value, source), source),
                    dim1.value,
                    ExpNum.index(selfShape, dim0.value, source),
                    source
                ),
                source
            )
        );
        const dimPNNext = dimPN.flatMap((ctx) => {
            const ndim1 = ExpNum.bop(NumBopType.Add, selfRank, dim1.value, source);
            return genTensor(
                ctx,
                ExpShape.setDim(
                    ExpShape.setDim(selfShape, dim0.value, ExpNum.index(selfShape, ndim1, source), source),
                    ndim1,
                    ExpNum.index(selfShape, dim0.value, source),
                    source
                ),
                source
            );
        });
        const dimNPNext = dimNP.flatMap((ctx) => {
            const ndim0 = ExpNum.bop(NumBopType.Add, selfRank, dim0.value, source);
            return genTensor(
                ctx,
                ExpShape.setDim(
                    ExpShape.setDim(selfShape, ndim0, ExpNum.index(selfShape, dim1.value, source), source),
                    dim1.value,
                    ExpNum.index(selfShape, ndim0, source),
                    source
                ),
                source
            );
        });
        const dimNNNext = dimNN.flatMap((ctx) => {
            const ndim0 = ExpNum.bop(NumBopType.Add, selfRank, dim0.value, source);
            const ndim1 = ExpNum.bop(NumBopType.Add, selfRank, dim1.value, source);
            return genTensor(
                ctx,
                ExpShape.setDim(
                    ExpShape.setDim(selfShape, ndim0, ExpNum.index(selfShape, ndim1, source), source),
                    ndim1,
                    ExpNum.index(selfShape, ndim0, source),
                    source
                ),
                source
            );
        });

        return dimPPNext.join(dimPNNext).join(dimNPNext).join(dimNNNext);
    }

    export function reduce(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.reduce': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [inputAddr, dimAddr, keepdimAddr] = params;

        const inputSize = fetchSize(inputAddr, heap);
        if (typeof inputSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.reduce': ${inputSize}`, source);
        }

        const dim = fetchAddr(dimAddr, heap);
        let keepdim = fetchAddr(keepdimAddr, heap);
        if (dim === undefined || (dim.type !== SVType.Int && dim.type !== SVType.None)) {
            return ctx.failWithMsg(`from 'LibCall.torch.reduce': cannot infer dim as integer`, source).toSet();
        }
        if (keepdim === undefined || (keepdim.type !== SVType.Bool && keepdim.type !== SVType.None)) {
            return ctx.failWithMsg(`from 'LibCall.torch.reduce': cannot infer keepdim as boolean`, source).toSet();
        }

        // if dim == None, return argmax of the flattend input.(scalar tensor)
        if (dim.type === SVType.None) {
            return ctx.genConstRankedShape(0, source).flatMap((ctx) => {
                const symShape = ctx.retVal;
                return genTensor(ctx, symShape, source);
            });
        }

        // default value of keepdim is false.
        if (keepdim.type === SVType.None) {
            keepdim = SVBool.create(false, source);
        }

        const inputShape = inputSize.shape;
        const inputRank = inputSize.rank();

        const shape1 = ExpShape.slice(inputShape, 0, dim.value, source);
        const shape2 = ExpShape.slice(inputShape, ExpNum.bop(NumBopType.Add, dim.value, 1), inputRank, source);

        return ctx
            .require(
                // TODO: handle negative index
                [ctx.genLte(0, dim.value, source), ctx.genLt(dim.value, inputRank, source)],
                `from 'LibCall.torch.reduce': dim must be within rank`,
                source
            )
            .flatMap((ctx) => {
                const isKeepdimSet = ctx.genBool((keepdim as SVBool).value, source);
                const [keepdimPath, notKeepdimPath] = ctx.ifThenElse(isKeepdimSet, source);

                // keepdim=True : [..., d, ...] -> [..., 1, ...]
                const leftPath = keepdimPath.flatMap((ctx) => {
                    const newDimShape = ExpShape.fromConst(1, [1], source);
                    const newShape_ = ExpShape.concat(shape1, newDimShape, source);
                    const newShape = ExpShape.concat(newShape_, shape2, source);

                    return genTensor(ctx, newShape, source);
                });

                // keepdim=False : [..., d, ...] -> [..., ...]
                const rightPath = notKeepdimPath.flatMap((ctx) => {
                    const newShape = ExpShape.concat(shape1, shape2, source);

                    return genTensor(ctx, newShape, source);
                });

                return leftPath.join(rightPath);
            });
    }

    // TODO: currently, assumed -1 is given only via constant rank tuple.
    export function view(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.view': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [selfAddr, shapeAddr] = params;

        const selfSize = fetchSize(selfAddr, heap);
        if (typeof selfSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.view': ${selfSize}`, source);
        }
        const selfShape = selfSize.shape;
        const selfNumel = ExpNum.numel(selfShape, source);

        let shape: ExpShape = ExpShape.fromConst(0, []);
        const shapeObj = fetchAddr(shapeAddr, heap);
        if (shapeObj === undefined) {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.view': ${shapeObj}`, source);
        }

        // if size is an integer. TODO: check integer input size go through this.
        else if (shapeObj.type === SVType.Int) {
            shape = ExpShape.fromConst(1, [shapeObj.value], source);
        } else if (shapeObj.type === SVType.Object) {
            // else if size is a tensor.size().
            const firstArg = shapeObj.getIndice(0);
            const firstArgObj = fetchAddr(firstArg, heap);
            if (firstArgObj && firstArgObj.type === SVType.Object && firstArgObj?.shape !== undefined) {
                shape = firstArgObj.shape;
            } else if (firstArgObj && firstArgObj.type === SVType.Object) {
                return ctx.warnTensorWithMsg(`from 'LibCall.torch.view': tensor size not found`, source);
            }
            // else size is an tuple of integers
            else {
                const shapeRank_ = fetchAddr(shapeObj.getAttr('$length'), heap);
                if (shapeRank_ === undefined) {
                    return ctx.warnTensorWithMsg(`from 'LibCall.torch.view': ${shapeRank_}`, source);
                }

                // tuple must have constant rank.
                if (shapeRank_.type !== SVType.Int) {
                    return ctx.warnTensorWithMsg(`from 'LibCall.torch.view': ${shapeRank_}`, source);
                } else {
                    let shapeRank = -1;
                    if (typeof shapeRank_.value === 'number') {
                        shapeRank = shapeRank_.value;
                    } else if (shapeRank_.value.opType === NumOpType.Const) {
                        shapeRank = shapeRank_.value.value;
                    } else {
                        return ctx.warnTensorWithMsg(`from 'LibCall.torch.view': size must have constant rank`, source);
                    }

                    const dimsMap = shapeObj.extractIndexedNumber(heap);
                    const dims: ExpNum[] = [];
                    let wildCardIdx = -1; // index of -1

                    for (let i = 0; i < shapeRank; i++) {
                        const dim = dimsMap.get(i);
                        if (dim === undefined) {
                            return ctx.warnTensorWithMsg(
                                `from 'LibCall.torch.view': input size must involve every dim ${shapeObj.toString()}`,
                                source
                            );
                        }
                        dims.push(dim);
                        if (dim.opType === NumOpType.Const && dim.value === -1) {
                            wildCardIdx = i;
                        }
                    }

                    // Special case: input size includes a -1.
                    if (wildCardIdx !== -1) {
                        const shapeL = ExpShape.fromConst(wildCardIdx, dims.slice(0, wildCardIdx));
                        const shapeR = ExpShape.fromConst(shapeRank - wildCardIdx - 1, dims.slice(wildCardIdx + 1));
                        const numelL = ExpNum.numel(shapeL, source);
                        const numelR = ExpNum.numel(shapeR, source);

                        const isLeftRankZero = ctx.genEq(0, shapeL.rank, source);
                        const isRightRankZero = ctx.genEq(0, shapeR.rank, source);
                        const [leftZeroPath, leftNonzeroPath] = ctx.ifThenElse(isLeftRankZero, source);
                        const pathL = leftZeroPath.flatMap((ctx) => {
                            const [rightZeroPath, rightNonzeroPath] = ctx.ifThenElse(isRightRankZero, source);
                            // path0: shape argument is [-1]
                            const pathLL = rightZeroPath.flatMap((ctx) => {
                                const newShape = ExpShape.fromConst(1, [selfNumel], source);
                                return genTensor(ctx, newShape, source);
                            });
                            // path1: shape argument is [-1, ...]
                            const pathLR = rightNonzeroPath.flatMap((ctx) => {
                                const wildCardDim = ExpNum.bop(NumBopType.FloorDiv, selfNumel, numelR, source);
                                const wildCardDimShape = ExpShape.fromConst(1, [wildCardDim], source);
                                const newShape = ExpShape.concat(wildCardDimShape, shapeR, source);
                                const mod = ExpNum.bop(NumBopType.Mod, selfNumel, numelR, source);
                                return ctx
                                    .require(
                                        [ctx.genEq(mod, 0, source)],
                                        `from 'LibCall.torch.view': numel mismatch. selfSize: ${selfNumel} must be dividable by ${numelR}`,
                                        source
                                    )
                                    .flatMap((ctx) => genTensor(ctx, newShape, source));
                            });
                            return pathLL.join(pathLR);
                        });
                        const pathR = leftNonzeroPath.flatMap((ctx) => {
                            const [rightZeroPath, rightNonzeroPath] = ctx.ifThenElse(isRightRankZero, source);
                            // path2: shape argument is [..., -1]
                            const pathRL = rightZeroPath.flatMap((ctx) => {
                                const wildCardDim = ExpNum.bop(NumBopType.FloorDiv, selfNumel, numelL, source);
                                const wildCardDimShape = ExpShape.fromConst(1, [wildCardDim], source);
                                const newShape = ExpShape.concat(shapeL, wildCardDimShape, source);
                                const mod = ExpNum.bop(NumBopType.Mod, selfNumel, numelL, source);
                                return ctx
                                    .require(
                                        [ctx.genEq(mod, 0, source)],
                                        `from 'LibCall.torch.view': numel mismatch. selfSize: ${selfNumel} must be dividable by ${numelL}`,
                                        source
                                    )
                                    .flatMap((ctx) => genTensor(ctx, newShape, source));
                            });
                            // path3: shape argument is [..., -1, ...]
                            const pathRR = rightNonzeroPath.flatMap((ctx) => {
                                const numelLR = ExpNum.bop(NumBopType.Mul, numelL, numelR, source);
                                const wildCardDim = ExpNum.bop(NumBopType.FloorDiv, selfNumel, numelLR, source);
                                const wildCardDimShape = ExpShape.fromConst(1, [wildCardDim], source);
                                const newShape_ = ExpShape.concat(shapeL, wildCardDimShape, source);
                                const newShape = ExpShape.concat(newShape_, shapeR);
                                const mod = ExpNum.bop(NumBopType.Mod, selfNumel, numelLR, source);
                                return ctx
                                    .require(
                                        [ctx.genEq(mod, 0, source)],
                                        `from 'LibCall.torch.view': numel mismatch. selfSize: ${selfNumel} must be dividable by ${numelLR}`,
                                        source
                                    )
                                    .flatMap((ctx) => genTensor(ctx, newShape, source));
                            });
                            return pathRL.join(pathRR);
                        });
                        return pathL.join(pathR);
                    }

                    // input size doesn't include -1.
                    else {
                        shape = ExpShape.fromConst(shapeRank, dims, source);
                    }
                }
            }
        } else {
            shape = ExpShape.fromConst(1, [100], source);
        }
        const shapeNumel = ExpNum.numel(shape, source);

        return ctx
            .require(
                [
                    // TODO: Commented out to avoid call stack excess
                    // ctx.genForall(ctx.genSymInt('i', source), [0, shapeRank], ctx.genLt(0, i)),
                    ctx.genEq(selfNumel, shapeNumel, source),
                ],
                `from 'LibCall.torch.view': numel mismatch`,
                source
            )
            .flatMap((ctx) => genTensor(ctx, shape, source));
    }

    export function conv2d(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 7) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.conv2d': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [inputAddr, weightAddr, biasAddr, strideAddr, paddingAddr, dilationAddr, groupsAddr] = params;

        const inputSize = fetchSize(inputAddr, heap);
        const weightSize = fetchSize(weightAddr, heap);

        if (typeof inputSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'Libcall.torch.conv2d: ${inputSize}`, source);
        }
        if (typeof weightSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'Libcall.torch.conv2d: ${weightSize}`, source);
        }

        const inputShape = inputSize.shape;
        const inputRank = inputSize.rank();
        const weightShape = weightSize.shape;
        const weightRank = weightSize.rank();

        const out_channels = SVInt.create(ExpNum.index(weightShape, 0), source);
        const in_channels = SVInt.create(ExpNum.index(weightShape, 1), source);
        const kernel_size_0 = SVInt.create(ExpNum.index(weightShape, 2), source);
        const kernel_size_1 = SVInt.create(ExpNum.index(weightShape, 3), source);

        // bias can be either None or (out_channels) shaped tensor
        let bias_channels: ExpNum | number;
        let biasRank: ExpNum | number;
        if (biasAddr.type === SVType.None) {
            bias_channels = -1;
            biasRank = 1;
        } else {
            const biasSize = fetchSize(biasAddr, heap);
            if (typeof biasSize === 'string') {
                return ctx.warnTensorWithMsg(`from 'Libcall.torch.conv2d: ${biasSize}`, source);
            }
            const biasShape = biasSize.shape;
            bias_channels = ExpNum.index(biasShape, 0, source);
            biasRank = biasSize.rank();
        }

        const stride = fetchAddr(strideAddr, heap);
        let stride_0, stride_1: SVInt;
        if (stride === undefined) {
            return ctx.warnTensorWithMsg(`from 'Libcall.torch.conv2d: stride is undefined`, source);
        } else if (stride.type !== SVType.Object && stride.type !== SVType.Int) {
            return ctx.warnTensorWithMsg(`from 'Libcall.torch.conv2d: stride is not an int nor int pair`, source);
        } else if (stride.type === SVType.Object) {
            stride_0 = stride.getIndice(0) as SVInt;
            stride_1 = stride.getIndice(1) as SVInt;
        } else {
            stride_0 = stride;
            stride_1 = stride;
        }

        const padding = fetchAddr(paddingAddr, heap);
        let padding_0, padding_1: SVInt;
        if (padding === undefined) {
            return ctx.warnTensorWithMsg(`from 'Libcall.torch.conv2d: padding is undefined`, source);
        } else if (padding.type !== SVType.Object && padding.type !== SVType.Int) {
            return ctx.warnTensorWithMsg(`from 'Libcall.torch.conv2d: padding is not an int nor int pair`, source);
        } else if (padding.type === SVType.Object) {
            padding_0 = padding.getIndice(0) as SVInt;
            padding_1 = padding.getIndice(1) as SVInt;
        } else {
            padding_0 = padding;
            padding_1 = padding;
        }

        const dilation = fetchAddr(dilationAddr, heap);
        let dilation_0, dilation_1: SVInt;
        if (dilation === undefined) {
            return ctx.warnTensorWithMsg(`from 'Libcall.torch.conv2d: dilation is undefined`, source);
        } else if (dilation.type !== SVType.Object && dilation.type !== SVType.Int) {
            return ctx.warnTensorWithMsg(`from 'Libcall.torch.conv2d: dilation is not an int nor int pair`, source);
        } else if (dilation.type === SVType.Object) {
            dilation_0 = dilation.getIndice(0) as SVInt;
            dilation_1 = dilation.getIndice(1) as SVInt;
        } else {
            dilation_0 = dilation;
            dilation_1 = dilation;
        }

        const groups = fetchAddr(groupsAddr, heap) as SVInt;

        // size1 = input[2] + 2 * padding - dilation * (kernel_size - 1) - 1
        const exp1 = ExpNum.bop(NumBopType.Mul, 2, padding_0.value, source);
        const exp2 = ExpNum.bop(NumBopType.Sub, kernel_size_0.value, 1, source);
        const exp3 = ExpNum.bop(NumBopType.Mul, dilation_0.value, exp2, source);
        const exp4 = ExpNum.bop(NumBopType.Add, ExpNum.index(inputShape, 2, source), exp1, source);
        const exp5 = ExpNum.bop(NumBopType.Sub, exp4, exp3, source);
        const size1 = ExpNum.bop(NumBopType.Sub, exp5, 1, source);

        // size2 = input[3] + 2 * padding - dilation * (kernel_size - 1) - 1
        const exp6 = ExpNum.bop(NumBopType.Mul, 2, padding_1.value, source);
        const exp7 = ExpNum.bop(NumBopType.Sub, kernel_size_1.value, 1, source);
        const exp8 = ExpNum.bop(NumBopType.Mul, dilation_1.value, exp7, source);
        const exp9 = ExpNum.bop(NumBopType.Add, ExpNum.index(inputShape, 3, source), exp6, source);
        const exp10 = ExpNum.bop(NumBopType.Sub, exp9, exp8, source);
        const size2 = ExpNum.bop(NumBopType.Sub, exp10, 1, source);

        // dims of return shape
        const dim0 = ExpNum.index(inputShape, 0, source);
        const dim1 = out_channels.value;
        const dim2 = ExpNum.bop(
            NumBopType.Add,
            ExpNum.bop(NumBopType.FloorDiv, size1, stride_0.value, source),
            1,
            source
        );
        const dim3 = ExpNum.bop(
            NumBopType.Add,
            ExpNum.bop(NumBopType.FloorDiv, size2, stride_1.value, source),
            1,
            source
        );

        return ctx
            .require(
                [ctx.genEq(4, inputRank, source), ctx.genEq(4, weightRank, source), ctx.genEq(1, biasRank, source)],
                `from 'LibCall.torch.conv2d': (input, weight, bias) rank is not (4, 4, 1)`,
                source
            )
            .require(
                [ctx.genOr(ctx.genEq(-1, bias_channels), ctx.genEq(out_channels.value, bias_channels))],
                `from 'LibCall.torch.conv2d': bias channel mismatch`,
                source
            )
            .require(
                [ctx.genEq(ExpNum.index(inputShape, 1, source), in_channels.value, source)],
                `from 'LibCall.torch.conv2d': in-channel mismatch`,
                source
            )
            .require(
                [
                    ctx.genLte(0, size1, source),
                    ctx.genLte(0, size2, source),
                    ctx.genEq(0, ExpNum.bop(NumBopType.Mod, in_channels.value, groups.value, source), source),
                    ctx.genEq(0, ExpNum.bop(NumBopType.Mod, out_channels.value, groups.value, source), source),
                ],
                `from 'LibCall.torch.conv2d': output size should be non-negative`,
                source
            )
            .flatMap((ctx) => genTensor(ctx, ExpShape.fromConst(4, [dim0, dim1, dim2, dim3], source), source));
    }

    export function pool2d(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 6) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.pool2d': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [inputAddr, kernel_sizeAddr, strideAddr, paddingAddr, dilationAddr, ceil_modeAddr] = params;

        const inputSize = fetchSize(inputAddr, heap);

        const kernel_size = fetchAddr(kernel_sizeAddr, heap);
        let kernel_size_0: SVInt;
        let kernel_size_1: SVInt;
        if (kernel_size === undefined) {
            return ctx.warnTensorWithMsg(`from 'Libcall.torch.pool2d: kernel_size is undefined`, source);
        } else if (kernel_size.type !== SVType.Object && kernel_size.type !== SVType.Int) {
            return ctx.warnTensorWithMsg(`from 'Libcall.torch.pool2d: kernel_size is not an int nor int pair`, source);
        } else if (kernel_size.type === SVType.Object) {
            kernel_size_0 = kernel_size.getIndice(0) as SVInt;
            kernel_size_1 = kernel_size.getIndice(1) as SVInt;
        } else {
            kernel_size_0 = kernel_size;
            kernel_size_1 = kernel_size;
        }

        const stride = fetchAddr(strideAddr, heap);
        let stride_0: SVInt;
        let stride_1: SVInt;
        if (stride === undefined) {
            return ctx.warnTensorWithMsg(`from 'Libcall.torch.pool2d: stride is undefined`, source);
        } else if (stride.type !== SVType.Object && stride.type !== SVType.Int) {
            return ctx.warnTensorWithMsg(`from 'Libcall.torch.pool2d: stride is not an int nor int pair`, source);
        } else if (stride.type === SVType.Object) {
            stride_0 = stride.getIndice(0) as SVInt;
            stride_1 = stride.getIndice(1) as SVInt;
        } else {
            stride_0 = stride;
            stride_1 = stride;
        }

        const padding = fetchAddr(paddingAddr, heap);
        let padding_0: SVInt;
        let padding_1: SVInt;
        if (padding === undefined) {
            return ctx.warnTensorWithMsg(`from 'Libcall.torch.pool2d: padding is undefined`, source);
        } else if (padding.type !== SVType.Object && padding.type !== SVType.Int) {
            return ctx.warnTensorWithMsg(`from 'Libcall.torch.pool2d: padding is not an int nor int pair`, source);
        } else if (padding.type === SVType.Object) {
            padding_0 = padding.getIndice(0) as SVInt;
            padding_1 = padding.getIndice(1) as SVInt;
        } else {
            padding_0 = padding;
            padding_1 = padding;
        }

        const dilation = fetchAddr(dilationAddr, heap);
        let dilation_0: SVInt;
        let dilation_1: SVInt;
        if (dilation === undefined) {
            return ctx.warnTensorWithMsg(`from 'Libcall.torch.pool2d: dilation is undefined`, source);
        } else if (dilation.type !== SVType.Object && dilation.type !== SVType.Int) {
            return ctx.warnTensorWithMsg(`from 'Libcall.torch.pool2d: dilation is not an int nor int pair`, source);
        } else if (dilation.type === SVType.Object) {
            dilation_0 = dilation.getIndice(0) as SVInt;
            dilation_1 = dilation.getIndice(1) as SVInt;
        } else {
            dilation_0 = dilation;
            dilation_1 = dilation;
        }

        const ceil_mode = fetchAddr(ceil_modeAddr, heap) as SVBool;

        if (typeof inputSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.pool2d': ${inputSize}`, source);
        }

        const inputShape = inputSize.shape;
        const inputRank = inputSize.rank();
        const height = ExpNum.index(inputShape, ExpNum.bop(NumBopType.Sub, inputRank, 2, source), source);
        const width = ExpNum.index(inputShape, ExpNum.bop(NumBopType.Sub, inputRank, 1, source), source);

        // size1 = height + 2 * padding - dilation * (kernel_size - 1) - 1
        const exp1 = ExpNum.bop(NumBopType.Mul, 2, padding_0.value, source);
        const exp2 = ExpNum.bop(NumBopType.Sub, kernel_size_0.value, 1, source);
        const exp3 = ExpNum.bop(NumBopType.Mul, dilation_0.value, exp2, source);
        const exp4 = ExpNum.bop(NumBopType.Add, height, exp1, source);
        const exp5 = ExpNum.bop(NumBopType.Sub, exp4, exp3, source);
        const size1 = ExpNum.bop(NumBopType.Sub, exp5, 1, source);

        // size2 = width + 2 * padding - dilation * (kernel_size - 1) - 1
        const exp6 = ExpNum.bop(NumBopType.Mul, 2, padding_1.value, source);
        const exp7 = ExpNum.bop(NumBopType.Sub, kernel_size_1.value, 1, source);
        const exp8 = ExpNum.bop(NumBopType.Mul, dilation_1.value, exp7, source);
        const exp9 = ExpNum.bop(NumBopType.Add, width, exp6, source);
        const exp10 = ExpNum.bop(NumBopType.Sub, exp9, exp8, source);
        const size2 = ExpNum.bop(NumBopType.Sub, exp10, 1, source);

        // return shape
        const frontShape = ExpShape.slice(inputShape, 0, ExpNum.bop(NumBopType.Sub, inputRank, 2, source), source);

        // ceil_mode option; common constraint
        const commonBranch = ctx.require([
            ctx.genOr(ctx.genEq(3, inputRank, source), ctx.genEq(4, inputRank, source), source),
            ctx.genLte(ExpNum.bop(NumBopType.Mul, 2, padding_0.value), kernel_size_0.value),
            ctx.genLte(ExpNum.bop(NumBopType.Mul, 2, padding_1.value), kernel_size_1.value),
            ctx.genLte(0, size1, source),
            ctx.genLte(0, size2, source),
        ]);

        const [ceilPath, floorPath] = commonBranch.ifThenElse(ctx.genBool(ceil_mode.value));

        // when ceil_mode=true
        const ceilOnePath = ceilPath.flatMap((ctx) => {
            const h_out = ExpNum.bop(NumBopType.Add, ceilDiv(size1, stride_0.value, source), 1, source);
            const w_out = ExpNum.bop(NumBopType.Add, ceilDiv(size2, stride_1.value, source), 1, source);
            const backShape = ExpShape.fromConst(2, [h_out, w_out], source);
            const returnShape = ExpShape.concat(frontShape, backShape, source);

            return genTensor(ctx, returnShape, source);
        });
        // when ceil_mode=false
        const floorOnePath = floorPath.flatMap((ctx) => {
            const h_out = ExpNum.bop(
                NumBopType.Add,
                ExpNum.bop(NumBopType.FloorDiv, size1, stride_0.value, source),
                1,
                source
            );
            const w_out = ExpNum.bop(
                NumBopType.Add,
                ExpNum.bop(NumBopType.FloorDiv, size2, stride_1.value, source),
                1,
                source
            );
            const backShape = ExpShape.fromConst(2, [h_out, w_out], source);
            const returnShape = ExpShape.concat(frontShape, backShape, source);

            return genTensor(ctx, returnShape, source);
        });

        return ceilOnePath.join(floorOnePath);
    }

    // TODO: Implement batch_norm and let BatchNormNd call it.
    export function batchnorm2d(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length < 2) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.batchnorm2d': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [inputAddr, num_featuresAddr] = params;

        const inputSize = fetchSize(inputAddr, heap);
        const num_features = fetchAddr(num_featuresAddr, heap);

        if (typeof inputSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.batchnorm2d': ${inputSize}`, source);
        }
        if (num_features?.type !== SVType.Int) {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.batchnorm2d': ${num_features}`, source);
        }

        const inputShape = inputSize.shape;
        const inputRank = inputSize.rank();

        return ctx
            .require(
                [
                    ctx.genEq(4, inputRank, source),
                    ctx.genEq(num_features.value, ExpNum.index(inputShape, 1, source), source),
                ],
                `from 'LibCall.torch.batchnorm2d': rank must be 4, num_features must match`,
                source
            )
            .flatMap((ctx) => genTensor(ctx, inputShape, source));
    }

    // ctr = [ -x1.rank <= dim < x1.rank, -x2.rank <= dim < x2.rank]
    // if (dim < 0)
    //     ctr += [broadcastable(x1, x2)]
    // else
    //     ctr += [x1.rank == x2.rank && broadcastable(x1, x2)]
    export function cosine_similarity(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length < 3) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.cosine_similarity': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [x1Addr, x2Addr, dimAddr] = params;

        const x1Size = fetchSize(x1Addr, heap);
        const x2Size = fetchSize(x2Addr, heap);
        const dimVal = fetchAddr(dimAddr, heap);

        if (typeof x1Size === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.cosine_similarity': ${x1Size}`, source);
        }
        if (typeof x2Size === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.cosine_similarity': ${x2Size}`, source);
        }
        if (dimVal?.type !== SVType.Int) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.cosine_similarity': dim must be an int ${dimVal}`,
                source
            );
        }
        const x1Shape = x1Size.shape;
        const x2Shape = x2Size.shape;
        const x1Rank = x1Size.rank();
        const x2Rank = x2Size.rank();

        // Assume dim is always constant
        let dim: ExpNum;
        if (typeof dimVal.value === 'number') {
            dim = ExpNum.fromConst(dimVal.value, source);
        } else {
            dim = simplifyNum(ctx.ctrSet, dimVal.value);
            if (dim.opType !== NumOpType.Const) {
                return ctx.warnTensorWithMsg(
                    `from 'LibCall.torch.cosine_similarity': dim must be a constant number ${dimVal}`,
                    source
                );
            }
        }

        const x1x2Bc = ExpShape.broadcast(x1Shape, x2Shape, source);
        // negative index handling
        let dim_: ExpNum;
        if (dim.value < 0) {
            dim_ = ExpNum.bop(NumBopType.Add, ExpShape.getRank(x1x2Bc), dim, source);
        } else {
            dim_ = dim;
        }
        const shape1 = ExpShape.slice(x1x2Bc, 0, dim_, source);
        const shape2 = ExpShape.slice(x1x2Bc, ExpNum.bop(NumBopType.Add, dim_, 1), ExpShape.getRank(x1x2Bc), source);
        const returnShape = ExpShape.concat(shape1, shape2, source);

        if (dim.value < 0) {
            return ctx
                .require(
                    [
                        ctx.genLte(ExpNum.uop(NumUopType.Neg, x1Rank, source), dim, source),
                        ctx.genLt(dim, x1Rank, source),
                        ctx.genLte(ExpNum.uop(NumUopType.Neg, x2Rank, source), dim, source),
                        ctx.genLt(dim, x2Rank, source),
                        ctx.genBroadcastable(x1Shape, x2Shape, source),
                    ],
                    `from 'LibCall.torch.cosine_similarity': shapes must be broadcastable, dim must be within rank`,
                    source
                )
                .flatMap((ctx) => genTensor(ctx, returnShape, source));
        } else {
            return ctx
                .require(
                    [
                        ctx.genLte(ExpNum.uop(NumUopType.Neg, x1Rank, source), dim, source),
                        ctx.genLt(dim, x1Rank, source),
                        ctx.genLte(ExpNum.uop(NumUopType.Neg, x2Rank, source), dim, source),
                        ctx.genLt(dim, x2Rank, source),
                        ctx.genEq(x1Rank, x2Rank, source),
                        ctx.genBroadcastable(x1Shape, x2Shape, source),
                    ],
                    `from 'LibCall.torch.cosine_similarity': shapes must be broadcastable, dim must be within rank`,
                    source
                )
                .flatMap((ctx) => genTensor(ctx, returnShape, source));
        }
    }

    // conditions of elements in "target" is not considered.
    export function cross_entropy(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.cross_entropy': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [inputAddr, targetAddr, reductionAddr] = params;

        const inputSize = fetchSize(inputAddr, heap);
        if (typeof inputSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.cross_entropy': ${inputSize}`, source);
        }
        const inputShape = inputSize.shape;
        const inputRank = inputSize.rank();

        const targetSize = fetchSize(targetAddr, heap);
        if (typeof targetSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.cross_entropy': ${targetSize}`, source);
        }
        const targetShape = targetSize.shape;
        const targetRank = ExpShape.getRank(targetShape);

        const reduction = fetchAddr(reductionAddr, heap);
        if (reduction === undefined || reduction.type !== SVType.Bool) {
            return ctx
                .failWithMsg(`from 'LibCall.torch.cross_entropy': cannot infer reduction as boolean`, source)
                .toSet();
        }

        return ctx
            .require(
                [
                    ctx.genLte(2, inputRank, source),
                    ctx.genEq(ExpNum.bop(NumBopType.Sub, inputRank, 1, source), targetRank, source),
                    ctx.genEq(ExpNum.index(inputShape, 0, source), ExpNum.index(targetShape, 0, source), source),
                    ctx.genEq(
                        ExpShape.slice(inputShape, 2, inputRank, source),
                        ExpShape.slice(targetShape, 1, targetRank, source),
                        source
                    ),
                ],
                `from 'LibCall.torch.cross_entropy': input target shapes mismatch`,
                source
            )
            .flatMap((ctx) => {
                const isReductionSet = ctx.genBool((reduction as SVBool).value, source);
                const [reductionPath, noReductionPath] = ctx.ifThenElse(isReductionSet, source);

                // not(reduction='none') : output is a scalar tensor.
                const leftPath = reductionPath.flatMap((ctx) => {
                    const newShape = ExpShape.fromConst(0, [], source);
                    return genTensor(ctx, newShape, source);
                });

                // reduction='none' : output shape is equal to target shape.
                const rightpath = noReductionPath.flatMap((ctx) => {
                    return genTensor(ctx, targetShape, source);
                });

                return leftPath.join(rightpath);
            });
    }

    export function checkSameShape(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.checkSameShape': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [inputAddr, targetAddr] = params;

        const inputSize = fetchSize(inputAddr, heap);
        if (typeof inputSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.checkSameShape': ${inputSize}`, source);
        }
        const inputShape = inputSize.shape;

        const targetSize = fetchSize(targetAddr, heap);
        if (typeof targetSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.checkSameShape': ${targetSize}`, source);
        }
        const targetShape = targetSize.shape;

        return ctx
            .setRetVal(SVNone.create(source))
            .require(
                ctx.genEq(inputShape, targetShape, source),
                "from 'LibCall.torch.checkSameShape': got different shape",
                source
            );
    }

    // Assumption: "tensors" is a constantRanked sequence, and each element is available.
    // TODO: handle empty tensor.
    export function cat(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.cat': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [tensorsAddr, dimAddr] = params;

        const tensors = fetchAddr(tensorsAddr, heap);
        const dim = fetchAddr(dimAddr, heap);

        if (tensors?.type !== SVType.Object) {
            if (tensors?.type === SVType.Error) {
                // propagate error
                return ctx.warnTensorWithMsg(`from 'LibCall.torch.cat': 'tensors' is not iterable`, source);
            }
            return ctx.failWithMsg(`from 'LibCall.torch.cat': 'tensors' is not iterable`, source).toSet();
        } else if (dim?.type !== SVType.Int) {
            if (dim?.type === SVType.Error) {
                // propagate error
                return ctx.warnTensorWithMsg(`from 'LibCall.torch.cat': 'dim' is not an integer`, source);
            }
            return ctx.failWithMsg(`from 'LibCall.torch.cat': 'dim' is not an integer`, source).toSet();
        }

        // Assumption: length of "tensors" is constant.
        const tensorsLen_ = fetchAddr(tensors.getAttr('$length'), heap);
        if (!(tensorsLen_?.type === SVType.Int && typeof tensorsLen_.value === 'number')) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.cat': length of tensors is unknown, cannot iterate.`,
                source
            );
        }
        const tensorsLen = tensorsLen_.value;

        const tensor0 = fetchAddr(tensors.getIndice(0), heap);
        const size0 = fetchSize(tensor0, heap);
        if (typeof size0 === 'string') {
            if (tensor0?.type === SVType.Error) {
                return ctx.warnTensorWithMsg(`from 'LibCall.torch.cat': ${size0}`, source);
            }
            return ctx.failWithMsg(`from 'LibCall.torch.cat': ${size0}`, source).toSet();
        }
        const size0shape = size0.shape;
        const size0rank = size0.rank();
        const shape0Front = ExpShape.slice(size0shape, 0, dim.value, source);
        const shape0Back = ExpShape.slice(size0shape, ExpNum.bop(NumBopType.Add, dim.value, 1), size0rank, source);

        // TODO: handle negative index.
        const ctrs: Constraint[] = [ctx.genLte(0, dim.value, source), ctx.genLt(dim.value, size0rank)];
        let thickness: ExpNum = ExpNum.index(size0shape, dim.value, source);

        for (let i = 1; i < tensorsLen; i++) {
            const tensorI = fetchAddr(tensors.getIndice(i), heap);
            const sizeI = fetchSize(tensorI, heap);
            if (typeof sizeI === 'string') {
                if (tensorI?.type === SVType.Error) {
                    return ctx.warnTensorWithMsg(`from 'LibCall.torch.cat': ${sizeI}`, source);
                }
                return ctx.failWithMsg(`from 'LibCall.torch.cat': ${sizeI}`, source).toSet();
            }
            const sizeIshape = sizeI.shape;
            const shapeIFront = ExpShape.slice(sizeIshape, 0, dim.value, source);
            const shapeIBack = ExpShape.slice(sizeIshape, ExpNum.bop(NumBopType.Add, dim.value, 1), size0rank, source);

            ctrs.push(ctx.genEq(shape0Front, shapeIFront, source));
            ctrs.push(ctx.genEq(shape0Back, shapeIBack, source));
            thickness = ExpNum.bop(NumBopType.Add, thickness, ExpNum.index(sizeIshape, dim.value, source), source);
        }

        const shapeThick = ExpShape.fromConst(1, [thickness], source);
        const returnShape_ = ExpShape.concat(shape0Front, shapeThick, source);
        const returnShape = ExpShape.concat(returnShape_, shape0Back, source);

        return ctx
            .require(ctrs, `from 'LibCall.torch.cat': tensor shapes must match, dim must be within rank`, source)
            .flatMap((ctx) => {
                return genTensor(ctx, returnShape, source);
            });
    }

    export function unsqueeze(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length < 2) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.unsqueeze': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [inputAddr, dimAddr] = params;

        const inputSize = fetchSize(inputAddr, heap);
        const dim = fetchAddr(dimAddr, heap);

        if (typeof inputSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.unsqueeze': ${inputSize}`, source);
        }
        if (dim?.type !== SVType.Int) {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.unsqueeze': ${dim}`, source);
        }

        const inputShape = inputSize.shape;
        const inputRank = inputSize.rank();

        const shapeFront = ExpShape.slice(inputShape, 0, dim.value, source);
        const shapeBack = ExpShape.slice(inputShape, dim.value, inputRank, source);
        const shapeMid = ExpShape.fromConst(1, [1], source);
        const returnShape_ = ExpShape.concat(shapeFront, shapeMid, source);
        const returnShape = ExpShape.concat(returnShape_, shapeBack, source);

        return ctx
            .require(
                // TODO: handle negative index
                [ctx.genLte(0, dim.value, source), ctx.genLte(dim.value, inputRank, source)],
                `from 'LibCall.torch.unsqueeze': dim must be within rank`,
                source
            )
            .flatMap((ctx) => genTensor(ctx, returnShape, source));
    }

    export function diag(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length < 2) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.diag': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [inputAddr, diagonalAddr] = params;

        const inputSize = fetchSize(inputAddr, heap);
        const diagonal = fetchAddr(diagonalAddr, heap);

        if (typeof inputSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.diag': ${inputSize}`, source);
        }
        if (diagonal?.type !== SVType.Int) {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.diag': ${diagonal}`, source);
        }

        const inputShape = inputSize.shape;
        const inputRank = inputSize.rank();

        return ctx
            .require(
                [ctx.genOr(ctx.genEq(inputRank, 1, source), ctx.genEq(inputRank, 2, source), source)],
                `from 'LibCall.torch.diag': rank must be 1 or 2`,
                source
            )
            .flatMap((ctx) => {
                const isRankOne = ctx.genEq(inputRank, 1, source);
                const [rankOnePath, rankTwoPath] = ctx.ifThenElse(isRankOne, source);

                const leftPath = rankOnePath.flatMap((ctx) => {
                    const dim0 = ExpNum.index(inputShape, 0, source);
                    const diagAbs = ExpNum.uop(NumUopType.Abs, diagonal.value, source);
                    const newDim = ExpNum.bop(NumBopType.Add, dim0, diagAbs, source);
                    const returnShape = ExpShape.fromConst(2, [newDim, newDim], source);

                    return genTensor(ctx, returnShape, source);
                });

                const rightPath = rankTwoPath.flatMap((ctx) => {
                    const dim0 = ExpNum.index(inputShape, 0, source);
                    const dim1 = ExpNum.index(inputShape, 1, source);
                    const negDim0 = ExpNum.uop(NumUopType.Neg, dim0, source);
                    const dim0diag = ExpNum.bop(NumBopType.Add, dim0, diagonal.value, source);
                    const dim1diag = ExpNum.bop(NumBopType.Sub, dim1, diagonal.value, source);

                    const minDim = ExpNum.min([dim0, dim1, dim0diag, dim1diag], source);
                    const returnShape = ExpShape.fromConst(1, [minDim], source);

                    return ctx
                        .require(
                            [ctx.genLte(negDim0, diagonal.value, source), ctx.genLte(diagonal.value, dim1, source)],
                            `from 'LibCall.torch.diag': diagonal must be gte -d1 and lte d2`,
                            source
                        )
                        .flatMap((ctx) => {
                            return genTensor(ctx, returnShape, source);
                        });
                });

                return leftPath.join(rightPath);
            });
    }

    export function flatten(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length < 1) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.flatten': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [inputAddr, startDimAddr, endDimAddr] = params;

        // TODO: use kwargs info.
        // TODO: handle negative indexing

        const inputSize = fetchSize(inputAddr, heap);
        if (typeof inputSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.flatten': ${inputSize}`, source);
        }
        const inputShape = inputSize.shape;
        const inputRank = inputSize.rank();

        let startDim = fetchAddr(startDimAddr, heap);
        if (startDim === undefined) {
            startDim = SVInt.create(ExpNum.fromConst(0), source);
        } else if (startDim.type !== SVType.Int) {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.flatten': ${startDim}`, source);
        }

        let endDim = fetchAddr(endDimAddr, heap);
        if (endDim === undefined) {
            endDim = SVInt.create(ExpNum.bop(NumBopType.Sub, inputRank, 1, source), source);
        } else if (endDim.type !== SVType.Int) {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.flatten': ${endDim}`, source);
        } else if (endDim.value === -1 || endDim.value === ExpNum.fromConst(-1)) {
            endDim = SVInt.create(ExpNum.bop(NumBopType.Sub, inputRank, 1, source), source);
        }

        const frontShape = ExpShape.slice(inputShape, 0, startDim.value, source);
        const endShape = ExpShape.slice(
            inputShape,
            ExpNum.bop(NumBopType.Add, endDim.value, 1, source),
            inputRank,
            source
        );
        const middleShape = ExpShape.slice(
            inputShape,
            startDim.value,
            ExpNum.bop(NumBopType.Add, endDim.value, 1, source),
            source
        );

        const middleDim = ExpNum.numel(middleShape, source);
        const returnShape = ExpShape.concat(
            ExpShape.concat(frontShape, ExpShape.fromConst(1, [middleDim], source), source),
            endShape,
            source
        );

        return ctx
            .require(
                [ctx.genLte(0, startDim.value, source), ctx.genLt(endDim.value, inputRank, source)],
                `from 'LibCall.torch.flatten': start_dim, end_dim range error`,
                source
            )
            .flatMap((ctx) => genTensor(ctx, returnShape, source));
    }

    export function embedding(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length < 2) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.embedding': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [inputAddr, weightAddr] = params;

        const inputSize = fetchSize(inputAddr, heap);
        if (typeof inputSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.embedding': ${inputSize}`, source);
        }
        const inputShape = inputSize.shape;

        const weightSize = fetchSize(weightAddr, heap);
        if (typeof weightSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.embedding': ${weightSize}`, source);
        }
        const weightShape = weightSize.shape;
        const weightRank = weightSize.rank();

        const weightLastShape = ExpShape.slice(weightShape, 1, undefined, source);
        const returnShape = ExpShape.concat(inputShape, weightLastShape, source);

        return ctx.require([ctx.genEq(2, weightRank, source)]).flatMap((ctx) => genTensor(ctx, returnShape, source));
    }

    // TODO: `broadcastable` is not the sufficient condition for this code
    export function layer_norm(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length < 4) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.layer_norm': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [inputAddr, normAddr, weightAddr, biasAddr] = params;

        const inputSize = fetchSize(inputAddr, heap);
        if (typeof inputSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.embedding': ${inputSize}`, source);
        }
        const inputShape = inputSize.shape;

        const normSize = fetchSize(normAddr, heap);
        if (typeof normSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.embedding': ${normSize}`, source);
        }
        const normShape = normSize.shape;

        return ctx.shBroadcast(inputShape, normShape, source).flatMap((ctx) => genTensor(ctx, ctx.retVal, source));
    }

    export function pad(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.pad': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [selfAddr, padsAddr] = params;

        const selfSize = fetchSize(selfAddr, heap);
        const pads = fetchAddr(padsAddr, heap);

        if (typeof selfSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.pad': ${selfSize}`, source);
        } else if (pads?.type !== SVType.Object) {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.pad': pad is not iterable`, source);
        }

        const selfShape = selfSize.shape;
        const selfRank = selfSize.rank();

        const tupleLenObj = fetchAddr(pads.getAttr('$length'), heap);

        if (tupleLenObj?.type !== SVType.Int || typeof tupleLenObj.value !== 'number') {
            return ctx.failWithMsg(`from 'LibCall.torch.pad': pad is not iterable`, source).toSet();
        }

        const tupleLen = tupleLenObj.value;
        if (tupleLen % 2 === 1 || tupleLen <= 0) {
            return ctx.failWithMsg(`from 'LibCall.torch.pad': pad has an odd or invalid length`, source).toSet();
        }

        let newCtx: Context<any> = ctx;
        const padSizes: (ExpNum | number)[] = [];
        for (let i = 0; i < tupleLen; i++) {
            const padDim = fetchAddr(pads.getIndice(i), heap);
            if (padDim?.type === SVType.Int) {
                padSizes.push(padDim.value);
            } else {
                const dimCtx = newCtx.genIntGte(`pad_dim${i}`, 0, source);
                newCtx = dimCtx;
                padSizes.push(dimCtx.retVal);
            }
        }

        return newCtx
            .require(
                newCtx.genLte(tupleLen / 2, selfRank, source),
                "from 'LibCall.torch.pad': input shape has rank shorter than pad count / 2",
                source
            )
            .flatMap((ctx) => {
                let shape = selfShape;
                const rankN = tupleLen / 2;
                for (let i = 0; i < rankN; i++) {
                    const padLeft = padSizes[i * 2];
                    const padRight = padSizes[i * 2 + 1];
                    shape = ExpShape.setDim(
                        shape,
                        ExpNum.bop(NumBopType.Sub, selfRank, i + 1, source),
                        ExpNum.bop(
                            NumBopType.Add,
                            ExpNum.bop(
                                NumBopType.Add,
                                ExpNum.index(shape, ExpNum.bop(NumBopType.Sub, selfRank, i + 1, source), source),
                                padLeft,
                                source
                            ),
                            padRight,
                            source
                        ),
                        source
                    );
                }
                return genTensor(ctx, shape, source);
            });
    }

    export function adaptive(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.adaptive': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [selfAddr, padsAddr] = params;

        const selfSize = fetchSize(selfAddr, heap);
        const outputSize = fetchAddr(padsAddr, heap);

        if (typeof selfSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.adaptive': ${selfSize}`, source);
        } else if (outputSize?.type !== SVType.Object) {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.adaptive': output size is not iterable`, source);
        }

        const selfShape = selfSize.shape;
        const selfRank = selfSize.rank();

        const tupleLenObj = fetchAddr(outputSize.getAttr('$length'), heap);

        if (tupleLenObj?.type !== SVType.Int || typeof tupleLenObj.value !== 'number') {
            return ctx.failWithMsg(`from 'LibCall.torch.adaptive': output size is not iterable`, source).toSet();
        }

        const tupleLen = tupleLenObj.value;
        if (tupleLen <= 0) {
            return ctx.failWithMsg(`from 'LibCall.torch.adaptive': invalid length of output size`, source).toSet();
        }

        let newCtx: Context<any> = ctx;
        const outSizes: (ExpNum | number)[] = [];
        for (let i = 0; i < tupleLen; i++) {
            const outDim = fetchAddr(outputSize.getIndice(i), heap);
            if (outDim?.type === SVType.Int) {
                outSizes.push(outDim.value);
            } else {
                const dimCtx = newCtx.genIntGte(`out_dim${i}`, 0, source);
                newCtx = dimCtx;
                outSizes.push(dimCtx.retVal);
            }
        }

        return newCtx
            .require(
                newCtx.genLte(tupleLen, selfRank, source),
                "from 'LibCall.torch.adaptive': input shape has rank shorter than output size",
                source
            )
            .flatMap((ctx) => {
                let shape = selfShape;
                const rankN = tupleLen;
                for (let i = 0; i < rankN; i++) {
                    const outDim = outSizes[rankN - i - 1];
                    shape = ExpShape.setDim(shape, ExpNum.bop(NumBopType.Sub, selfRank, i + 1, source), outDim, source);
                }
                return genTensor(ctx, shape, source);
            });
    }

    export function genDatasetLen(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 1) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.genDatasetLen': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        return ctx.setRetVal(SVNotImpl.create('not implemented', source)).toSet();
    }

    export function datasetGetItem(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.genDatasetLen': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        return ctx.setRetVal(SVNotImpl.create('not implemented', source)).toSet();
    }

    export function warnTensorWithMsg(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 1) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.warnTensorWithMsg': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const msg = fetchAddr(params[0], heap);

        if (msg?.type !== SVType.String) {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.warnTensorWithMsg': invalid message`, source);
        }

        return ctx.warnTensorWithMsg(typeof msg.value === 'string' ? msg.value : ExpString.toString(msg.value), source);
    }

    // implementation of torch.nn.functional.interpolate
    export function interpolate(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.interpolate': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [inputAddr, sizeAddr, scaleFactorAddr] = params;

        const inputSize = fetchSize(inputAddr, heap);
        const outSizeObj = fetchAddr(sizeAddr, heap);
        const scaleFactorObj = fetchAddr(scaleFactorAddr, heap);

        if (typeof inputSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.interpolate': ${inputSize}`, source);
        }

        if (outSizeObj?.type === SVType.None && scaleFactorObj?.type === SVType.None) {
            return ctx
                .failWithMsg(
                    `from 'LibCall.torch.interpolate': only one of size or scale_factor should be defined`,
                    source
                )
                .toSet();
        }

        const outSize: (ExpNum | number)[] = [];
        let outSizeConst: ExpNum | number | undefined;
        if (outSizeObj?.type === SVType.Int) {
            outSizeConst = outSizeObj.value;
            outSize.push(outSizeConst);
        } else if (outSizeObj?.type === SVType.Object) {
            const dims = outSizeObj.extractIndexedNumber(ctx.heap);
            if (dims.has(0)) outSize.push(dims.get(0)!);
            if (dims.has(1)) outSize.push(dims.get(1)!);
            if (dims.has(2)) outSize.push(dims.get(2)!);
        }

        const scaleFactor: (ExpNum | number)[] = [];
        let scaleFactorConst: ExpNum | number | undefined;
        if (scaleFactorObj?.type === SVType.Float || scaleFactorObj?.type === SVType.Int) {
            scaleFactorConst = scaleFactorObj.value;
            scaleFactor.push(scaleFactorConst);
        } else if (scaleFactorObj?.type === SVType.Object) {
            const factors = scaleFactorObj.extractIndexedNumber(ctx.heap);
            if (factors.has(0)) scaleFactor.push(factors.get(0)!);
            if (factors.has(1)) scaleFactor.push(factors.get(1)!);
            if (factors.has(2)) scaleFactor.push(factors.get(2)!);
        }

        if (outSize.length === 0 && scaleFactor.length === 0) {
            return ctx
                .failWithMsg(
                    `from 'LibCall.torch.interpolate': only one of size or scale_factor should be defined`,
                    source
                )
                .toSet();
        }

        const inputShape = inputSize.shape;
        const inputRank = inputSize.rank();

        if (outSizeConst !== undefined || scaleFactorConst !== undefined) {
            return ctx
                .require(
                    [ctx.genLte(3, inputRank, source), ctx.genLte(inputRank, 5, source)],
                    `from 'LibCall.torch.interpolate': expected inputs are 3, 4, or 5-D shape`,
                    source
                )
                .flatMap((ctx: Context<unknown>) => {
                    const [rank3, rank45] = ctx.ifThenElse(ctx.genEq(3, inputRank, source));
                    const [rank4, rank5] = rank45.ifThenElse(ctx.genEq(4, inputRank, source));

                    const shapes: ExpShape[] = [];
                    let newShape: ExpShape = inputShape;
                    for (let dim = 2; dim <= 4; dim++) {
                        if (outSizeConst !== undefined) {
                            newShape = ExpShape.setDim(newShape, dim, outSizeConst, source);
                        } else {
                            newShape = ExpShape.setDim(
                                newShape,
                                dim,
                                ExpNum.uop(
                                    NumUopType.Floor,
                                    ExpNum.bop(
                                        NumBopType.Mul,
                                        ExpNum.index(newShape, dim, source),
                                        scaleFactorConst!,
                                        source
                                    ),
                                    source
                                ),
                                source
                            );
                        }
                        shapes.push(newShape);
                    }

                    return rank3
                        .flatMap((ctx) => genTensor(ctx, shapes[0], source))
                        .join(rank4.flatMap((ctx) => genTensor(ctx, shapes[1], source)))
                        .join(rank5.flatMap((ctx) => genTensor(ctx, shapes[2], source)));
                });
        } else if (outSize.length > 0) {
            return ctx
                .require(
                    ctx.genEq(inputRank, 2 + outSize.length, source),
                    `from 'LibCall.torch.interpolate': size shape must match with input shape`,
                    source
                )
                .flatMap((ctx) => {
                    let newShape: ExpShape = inputShape;
                    outSize.forEach((size, dim) => {
                        newShape = ExpShape.setDim(newShape, dim + 2, size, source);
                    });

                    return genTensor(ctx, newShape, source);
                });
        } else {
            return ctx
                .require(
                    ctx.genEq(inputRank, 2 + scaleFactor.length, source),
                    `from 'LibCall.torch.interpolate': scale_factor shape must match with input shape`,
                    source
                )
                .flatMap((ctx) => {
                    let newShape: ExpShape = inputShape;
                    scaleFactor.forEach((factor, dim) => {
                        newShape = ExpShape.setDim(
                            newShape,
                            dim + 2,
                            ExpNum.uop(
                                NumUopType.Floor,
                                ExpNum.bop(NumBopType.Mul, ExpNum.index(inputShape, dim + 2, source), factor, source),
                                source
                            ),
                            source
                        );
                    });

                    return genTensor(ctx, newShape, source);
                });
        }
    }

    export const libCallImpls: { [key: string]: LCImpl } = {
        tensorInit,
        identityShape,
        matmul,
        mm,
        bmm,
        item,
        copyOut,
        repeat,
        callTensor,
        transpose,
        reduce,
        view,
        conv2d,
        broadcast,
        pool2d,
        batchnorm2d,
        interpolate,
        cosine_similarity,
        cross_entropy,
        checkSameShape,
        cat,
        unsqueeze,
        diag,
        flatten,
        embedding,
        layer_norm,
        pad,
        adaptive,
        genDatasetLen,
        datasetGetItem,
        warnTensorWithMsg,
    };
}

export const libCallMap: Map<string, LCImpl> = new Map([...Object.entries(TorchLCImpl.libCallImpls)]);
