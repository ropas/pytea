import { ParseNode } from 'pyright-internal/parser/parseNodes';

import { LCImpl } from '..';
import { fetchAddr } from '../../backend/backUtils';
import { Context, ContextSet } from '../../backend/context';
import { fetchSize, genTensor, isSize } from '../../backend/expUtils';
import { ShValue, SVAddr, SVObject, SVSize, SVNone, SVType } from '../../backend/sharpValues';
import { ExpNum, ExpShape, NumBopType } from '../../backend/symExpressions';
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

    export const libCallImpls: { [key: string]: LCImpl } = {
        ndarrayInit,
        fromImage,
    };
}

export const libCallMap: Map<string, LCImpl> = new Map([...Object.entries(NumpyLCImpl.libCallImpls)]);
