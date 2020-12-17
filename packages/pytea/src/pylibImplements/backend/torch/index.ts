import { ParseNode } from 'pyright-internal/parser/parseNodes';

import { LCImpl } from '..';
import { fetchAddr } from '../../../backend/backUtils';
import { Constraint } from '../../../backend/constraintType';
import { Context, ContextSet } from '../../../backend/context';
import { ceilDiv, fetchSize, genTensor } from '../../../backend/expUtils';
import {
    ShValue,
    SVAddr,
    SVBool,
    SVFloat,
    SVInt,
    SVNone,
    SVNotImpl,
    SVObject,
    SVSize,
    SVType,
} from '../../../backend/sharpValues';
import { ExpNum, ExpShape, ExpString, NumBopType, NumOpType, NumUopType } from '../../../backend/symExpressions';
import { TorchBackend } from '../../../backend/torchBackend';
import { LCBase } from '../libcall';

export namespace TorchLCImpl {
    export function tensorInit(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.tensorInit': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [selfAddr, argsAddr, kwargs] = params;

        // TODO: use kwargs info.

        // tensorInit is always used in Tensor.__init__ -> force casting
        const addr = selfAddr as SVAddr;
        const self = fetchAddr(selfAddr, heap)! as SVObject;
        const args = fetchAddr(argsAddr as SVAddr, heap)! as SVObject;

        // if args is object that has 'shape'
        let firstArg = fetchAddr(args.getIndice(0), heap);
        if (firstArg?.type === SVType.Object) {
            if (firstArg.shape === undefined) {
                firstArg = fetchAddr(firstArg.getAttr('shape'), heap);
            }

            if (firstArg && firstArg.type === SVType.Object && firstArg?.shape !== undefined) {
                const newHeap = heap.setVal(addr, self.setAttr('shape', firstArg));
                return ctx.setHeap(newHeap).setRetVal(SVNone.create()).toSet();
            }
        }

        // if args is list of integer
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

    export function identityShape(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
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
    export function broadcast(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
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
    export function matmul(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
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

    export function item(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 1) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.item': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const selfAddr = params[0];

        const selfSize = fetchSize(selfAddr, heap);

        if (typeof selfSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.item': ${selfSize}`, source);
        }

        const selfShape = selfSize.shape;
        const selfRank = selfSize.rank();

        return ctx
            .require(
                [ctx.genOr(ctx.genEq(0, selfRank, source), ctx.genEq(1, ExpNum.numel(selfShape, source)), source)],
                `from 'LibCall.torch.item': tensor must have exacly one element`,
                source
            ) // TODO: match return value type with tensor dtype.
            .return(SVFloat.create(ExpNum.fromSymbol(ctx.genSymFloat('torchItemElem', source)), source));
    }

    // implementation of torch.Tensor.__getitem__
    export function getItem(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.getItem': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [selfAddr, item] = params;

        const selfSize = fetchSize(selfAddr, heap);
        const indices = fetchAddr(item, heap);

        if (typeof selfSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.getItem': ${selfSize}`, source);
        } else if (indices?.type !== SVType.Int) {
            // TODO: index by tuple
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.getItem: index by non-integer value is not supported currently.`,
                source
            );
        }

        const selfShape = selfSize.shape;
        const rank = selfSize.rank();
        const firstDim = ExpNum.index(selfShape, 0, source);
        const index = indices.value;

        return ctx
            .require([
                ctx.genLte(1, rank, source),
                ctx.genLte(ExpNum.bop(NumBopType.Sub, 0, firstDim, source), index, source),
                ctx.genLte(index, ExpNum.bop(NumBopType.Sub, firstDim, 1, source), source),
            ])
            .flatMap((ctx) => {
                return genTensor(ctx, ExpShape.slice(selfShape, 1, undefined, source));
            });
    }

    // implementation of torch.Tensor.repeat
    export function repeat(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
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

        let tupleLen = tupleLenFetch.value;
        if (sizeObj.type === SVType.Object) {
            // first argument is object
            tupleLen = -1;
        } else if (typeof tupleLen === 'number' && tupleLen >= 2) {
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
                    `from 'LibCall.torch.repeat': Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor`
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

    export function copyOut(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx
                .warnWithMsg(`LibCall.torch copyOut got insufficient number of argument: ${params.length}`, source)
                .toSet();
        }

        const [tensorAddr, out] = params;
        if (out.type === SVType.None) {
            return ctx.toSetWith(out);
        }

        if (out.type !== SVType.Addr) {
            return ctx
                .warnWithMsg(`LibCall.torch copyOut type error: out is not an address - got ${out.type}`, source)
                .toSet();
        }

        const heap = ctx.heap;
        const tensor = fetchAddr(tensorAddr, heap);

        return (tensor ? ctx.setHeap(heap.setVal(out, tensor)) : ctx).setRetVal(out).toSet();
    }

    export function callTensor(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
        return TorchBackend.libClassInit(ctx, 'torch.Tensor', ctx.retVal.params, source);
    }

    export function transpose(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
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

    export function reduce(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
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
    export function view(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
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
                        const shape1 = ExpShape.fromConst(wildCardIdx, dims.slice(0, wildCardIdx));
                        const shape2 = ExpShape.fromConst(shapeRank - wildCardIdx - 1, dims.slice(wildCardIdx + 1));
                        const numel1 = ExpNum.numel(shape1, source);
                        const numel2 = ExpNum.numel(shape2, source);
                        const numel12 = ExpNum.bop(NumBopType.Mul, numel1, numel2, source);
                        const wildCardDim = ExpNum.bop(NumBopType.FloorDiv, selfNumel, numel12, source);
                        const wildCardDimShape = ExpShape.fromConst(1, [wildCardDim], source);
                        const newShape_ = ExpShape.concat(shape1, wildCardDimShape, source);
                        const newShape = ExpShape.concat(newShape_, shape2);

                        const mod = ExpNum.bop(NumBopType.Mod, selfNumel, numel12, source);

                        return ctx
                            .require([
                                // TODO: Commented out to avoid call stack excess
                                // ctx.genForall(ctx.genSymInt('i', source), [0, wildCardIdx], ctx.genLt(0, i)),
                                ctx.genEq(mod, 0, source),
                            ])
                            .flatMap((ctx) => genTensor(ctx, newShape, source));
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
            .require([
                // TODO: Commented out to avoid call stack excess
                // ctx.genForall(ctx.genSymInt('i', source), [0, shapeRank], ctx.genLt(0, i)),
                ctx.genEq(selfNumel, shapeNumel, source),
            ])
            .flatMap((ctx) => genTensor(ctx, shape, source));
    }

    export function conv2d(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
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

        // stride can be either None or (int, int) tuple
        let stride_0, stride_1: SVInt;
        if (strideAddr.type === SVType.None) {
            [stride_0, stride_1] = [kernel_size_0, kernel_size_1];
        } else {
            const stride_tuple = fetchAddr(strideAddr, heap) as SVObject;
            stride_0 = stride_tuple.getIndice(0) as SVInt;
            stride_1 = stride_tuple.getIndice(1) as SVInt;
        }
        const padding_tuple = fetchAddr(paddingAddr, heap) as SVObject;
        const padding_0 = padding_tuple.getIndice(0) as SVInt;
        const padding_1 = padding_tuple.getIndice(1) as SVInt;
        const dilation_tuple = fetchAddr(dilationAddr, heap) as SVObject;
        const dilation_0 = dilation_tuple.getIndice(0) as SVInt;
        const dilation_1 = dilation_tuple.getIndice(1) as SVInt;

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
            .require([
                ctx.genEq(4, inputRank, source),
                ctx.genEq(4, weightRank, source),
                ctx.genEq(1, biasRank, source),
                ctx.genOr(ctx.genEq(-1, bias_channels), ctx.genEq(out_channels.value, bias_channels)),
                ctx.genEq(ExpNum.index(inputShape, 1, source), in_channels.value, source),
                ctx.genLte(0, size1, source),
                ctx.genLte(0, size2, source),
                ctx.genEq(0, ExpNum.bop(NumBopType.Mod, in_channels.value, groups.value, source), source),
                ctx.genEq(0, ExpNum.bop(NumBopType.Mod, out_channels.value, groups.value, source), source),
            ])
            .flatMap((ctx) => genTensor(ctx, ExpShape.fromConst(4, [dim0, dim1, dim2, dim3], source), source));
    }

    export function pool2d(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
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

        const kernel_size_tuple = fetchAddr(kernel_sizeAddr, heap) as SVObject;
        const kernel_size_0 = kernel_size_tuple.getIndice(0) as SVInt;
        const kernel_size_1 = kernel_size_tuple.getIndice(1) as SVInt;

        const stride_tuple = fetchAddr(strideAddr, heap) as SVObject;
        const stride_0 = stride_tuple.getIndice(0) as SVInt;
        const stride_1 = stride_tuple.getIndice(1) as SVInt;

        const padding_tuple = fetchAddr(paddingAddr, heap) as SVObject;
        const padding_0 = padding_tuple.getIndice(0) as SVInt;
        const padding_1 = padding_tuple.getIndice(1) as SVInt;

        const dilation_tuple = fetchAddr(dilationAddr, heap) as SVObject;
        const dilation_0 = dilation_tuple.getIndice(0) as SVInt;
        const dilation_1 = dilation_tuple.getIndice(1) as SVInt;

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
    export function batchnorm2d(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
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

    export function cosine_similarity(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length < 3) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.batchnorm2d': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [x1Addr, x2Addr, dimAddr] = params;

        const x1Size = fetchSize(x1Addr, heap);
        const x2Size = fetchSize(x2Addr, heap);
        const dim = fetchAddr(dimAddr, heap);

        if (typeof x1Size === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.cosine_similarity': ${x1Size}`, source);
        }
        if (typeof x2Size === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.cosine_similarity': ${x2Size}`, source);
        }
        if (dim?.type !== SVType.Int) {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.cosine_similarity': ${dim}`, source);
        }

        const x1Shape = x1Size.shape;
        const x1Rank = x1Size.rank();
        const x2hape = x2Size.shape;

        const shape1 = ExpShape.slice(x1Shape, 0, dim.value, source);
        const shape2 = ExpShape.slice(x1Shape, ExpNum.bop(NumBopType.Add, dim.value, 1), x1Rank, source);
        const returnShape = ExpShape.concat(shape1, shape2, source);

        return ctx
            .require(
                // TODO: handle negative index
                [
                    ctx.genEq(x1Shape, x2hape, source),
                    ctx.genLte(0, dim.value, source),
                    ctx.genLt(dim.value, x1Rank, source),
                ],
                `from 'LibCall.torch.cosine_similarity': shapes must be equal, dim must be within rank`,
                source
            )
            .flatMap((ctx) => genTensor(ctx, returnShape, source));
    }

    // conditions of elements in "target" is not considered.
    export function loss(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.loss': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [inputAddr, targetAddr, reductionAddr] = params;

        const inputSize = fetchSize(inputAddr, heap);
        if (typeof inputSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.loss': ${inputSize}`, source);
        }
        const inputShape = inputSize.shape;
        const inputRank = inputSize.rank();

        const targetSize = fetchSize(targetAddr, heap);
        if (typeof targetSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.loss': ${targetSize}`, source);
        }
        const targetShape = targetSize.shape;
        const targetRank = ExpShape.getRank(targetShape);

        const reduction = fetchAddr(reductionAddr, heap);
        if (reduction === undefined || reduction.type !== SVType.Bool) {
            return ctx.failWithMsg(`from 'LibCall.torch.loss': cannot infer reduction as boolean`, source).toSet();
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
                `from 'LibCall.torch.loss': input target shapes mismatch`,
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

    // Assumption: "tensors" is a constantRanked sequence, and each element is available.
    // TODO: handle empty tensor.
    export function cat(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
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
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.cat': tensors is not iterable`, source);
        } else if (dim?.type !== SVType.Int) {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.cat': dim is not an integer`, source);
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

        const size0 = fetchSize(tensors.getIndice(0), heap);
        if (typeof size0 === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.cat': ${size0}`, source);
        }
        const size0shape = size0.shape;
        const size0rank = size0.rank();
        const shape0Front = ExpShape.slice(size0shape, 0, dim.value, source);
        const shape0Back = ExpShape.slice(size0shape, ExpNum.bop(NumBopType.Add, dim.value, 1), size0rank, source);

        // TODO: handle negative index.
        const ctrs: Constraint[] = [ctx.genLte(0, dim.value, source), ctx.genLt(dim.value, size0rank)];
        let thickness: ExpNum = ExpNum.index(size0shape, dim.value, source);

        for (let i = 1; i < tensorsLen; i++) {
            const sizeI = fetchSize(tensors.getIndice(i), heap);
            if (typeof sizeI === 'string') {
                return ctx.warnTensorWithMsg(`from 'LibCall.torch.cat': ${sizeI}`, source);
            }
            const sizeIshape = sizeI.shape;
            const shapeIFront = ExpShape.slice(sizeIshape, 0, dim.value, source);
            const shapeIBack = ExpShape.slice(sizeIshape, ExpNum.bop(NumBopType.Add, dim.value, 1), size0rank, source);

            ctrs.push(ctx.genEq(shape0Front, shapeIFront, source));
            ctrs.push(ctx.genEq(shape0Back, shapeIBack, source));
            thickness = ExpNum.bop(NumBopType.Add, thickness, ExpNum.index(sizeIshape, dim.value, source));
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

    export function unsqueeze(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
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

    export function diag(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
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

    export function flatten(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length < 1) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.flatten': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [inputAddr, start_dimAddr, end_dimAddr] = params;

        // TODO: use kwargs info.
        // TODO: handle negative indexing

        const inputSize = fetchSize(inputAddr, heap);
        if (typeof inputSize === 'string') {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.flatten': ${inputSize}`, source);
        }
        const inputShape = inputSize.shape;
        const inputRank = inputSize.rank();

        let start_dim = fetchAddr(start_dimAddr, heap);
        if (start_dim === undefined) {
            start_dim = SVInt.create(ExpNum.fromConst(0), source);
        } else if (start_dim.type !== SVType.Int) {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.flatten': ${start_dim}`, source);
        }

        let end_dim = fetchAddr(end_dimAddr, heap);
        if (end_dim === undefined) {
            end_dim = SVInt.create(ExpNum.bop(NumBopType.Sub, inputRank, 1, source), source);
        } else if (end_dim.type !== SVType.Int) {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.flatten': ${end_dim}`, source);
        } else if (end_dim.value === -1 || end_dim.value === ExpNum.fromConst(-1)) {
            end_dim = SVInt.create(ExpNum.bop(NumBopType.Sub, inputRank, 1, source), source);
        }

        const frontShape = ExpShape.slice(inputShape, 0, start_dim.value, source);
        const endShape = ExpShape.slice(
            inputShape,
            ExpNum.bop(NumBopType.Add, end_dim.value, 1, source),
            inputRank,
            source
        );
        const middleShape = ExpShape.slice(
            inputShape,
            start_dim.value,
            ExpNum.bop(NumBopType.Add, end_dim.value, 1, source),
            source
        );

        const middleDim = ExpNum.numel(middleShape, source);
        const returnShape = ExpShape.concat(
            ExpShape.concat(frontShape, ExpShape.fromConst(1, [middleDim], source), source),
            endShape,
            source
        );

        return ctx
            .require([ctx.genLte(0, start_dim.value, source), ctx.genLt(end_dim.value, inputRank, source)])
            .flatMap((ctx) => genTensor(ctx, returnShape, source));
    }

    export function genDatasetLen(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 1) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.genDatasetLen': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        return ctx.setRetVal(SVNotImpl.create('not implemented', source)).toSet();
    }

    export function datasetGetItem(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.genDatasetLen': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        return ctx.setRetVal(SVNotImpl.create('not implemented', source)).toSet();
    }

    export function warnTensorWithMsg(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
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

    export const libCallImpls: { [key: string]: LCImpl } = {
        tensorInit,
        identityShape,
        matmul,
        item,
        getItem,
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
        cosine_similarity,
        loss,
        cat,
        unsqueeze,
        diag,
        flatten,
        genDatasetLen,
        datasetGetItem,
        warnTensorWithMsg,
    };
}

export const libCallMap: Map<string, LCImpl> = new Map([...Object.entries(TorchLCImpl.libCallImpls)]);
