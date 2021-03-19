import { fetchAddr } from '../../backend/backUtils';
import { Context, ContextSet } from '../../backend/context';
import { CodeSource, ShValue, SVInt, SVType } from '../../backend/sharpValues';
import { ExpNum, NumUopType } from '../../backend/symExpressions';
import { LCImpl } from '..';
import { LCBase } from '../libcall';

export namespace MathLCImpl {
    export function ceil(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 1) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.math.ceil': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [numAddr] = params;

        const num = fetchAddr(numAddr, heap);

        if (num?.type !== SVType.Int && num?.type !== SVType.Float) {
            return ctx.failWithMsg(`from 'LibCall.math.ceil': input should be a number: ${num}`, source).toSet();
        }

        const numExp = num.value;
        const ceiled = ExpNum.uop(NumUopType.Ceil, numExp, source);

        return ctx.toSetWith(SVInt.create(ceiled, source));
    }

    export const libCallImpls: { [key: string]: LCImpl } = {
        ceil,
    };
}

export const libCallMap: Map<string, LCImpl> = new Map([...Object.entries(MathLCImpl.libCallImpls)]);
