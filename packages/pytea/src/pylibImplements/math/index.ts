import { TorchBackend } from 'src/backend/torchBackend';

import { fetchAddr } from '../../backend/backUtils';
import { Context, ContextSet } from '../../backend/context';
import { CodeSource, ShValue, SVFloat, SVInt, SVType, svTypeToString } from '../../backend/sharpValues';
import { ExpNum, NumUopType } from '../../backend/symExpressions';
import { LCImpl } from '..';
import { LCBase } from '../libcall';

export namespace MathLCImpl {
    export function ceil(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 1) {
            return ctx
                .warnWithMsg(`from 'LibCall.math.ceil': got insufficient number of argument: ${params.length}`, source)
                .toSet();
        }

        const heap = ctx.heap;
        const [numAddr] = params;

        const num = fetchAddr(numAddr, heap);

        if (!num) {
            return ctx.failWithMsg(`from 'LibCall.math.ceil: got undefined value`, source).toSet();
        }

        switch (num.type) {
            case SVType.Int:
                return ctx.toSetWith(num);
            case SVType.Float: {
                const numExp = num.value;
                if (typeof numExp === 'number') {
                    return ctx.toSetWith(SVInt.create(Math.ceil(numExp), source));
                }
                const exp = ExpNum.uop(NumUopType.Ceil, numExp, source);
                return ctx.toSetWith(SVInt.create(exp, source));
            }
            case SVType.Object:
                return ctx.getAttrDeep(num, '__ceil__', source).flatMap((ctx) => {
                    const ceil = ctx.retVal;
                    if (ceil.type === SVType.Func) {
                        return TorchBackend.functionCall(ctx, ceil, [], source);
                    }
                    return ctx
                        .warnWithMsg(`from 'LibCall.math.ceil: object does not have __ceil__ method`, source)
                        .toSet();
                });
            case SVType.Error:
                return ctx.warnWithMsg(`from 'LibCall.math.ceil: got unknown value`, source).toSet();
            default:
                return ctx
                    .failWithMsg(`from 'LibCall.math.ceil: invalid type - ${svTypeToString(num.type)}`, source)
                    .toSet();
        }
    }

    export function floor(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 1) {
            return ctx
                .warnWithMsg(`from 'LibCall.math.floor': got insufficient number of argument: ${params.length}`, source)
                .toSet();
        }

        const heap = ctx.heap;
        const [numAddr] = params;

        const num = fetchAddr(numAddr, heap);

        if (!num) {
            return ctx.failWithMsg(`from 'LibCall.math.floor: got undefined value`, source).toSet();
        }

        switch (num.type) {
            case SVType.Int:
                return ctx.toSetWith(num);
            case SVType.Float: {
                const numExp = num.value;
                if (typeof numExp === 'number') {
                    return ctx.toSetWith(SVInt.create(Math.floor(numExp), source));
                }
                const exp = ExpNum.uop(NumUopType.Floor, numExp, source);
                return ctx.toSetWith(SVInt.create(exp, source));
            }
            case SVType.Object:
                return ctx.getAttrDeep(num, '__floor__', source).flatMap((ctx) => {
                    const ceil = ctx.retVal;
                    if (ceil.type === SVType.Func) {
                        return TorchBackend.functionCall(ctx, ceil, [], source);
                    }
                    return ctx
                        .warnWithMsg(`from 'LibCall.math.floor: object does not have __floor__ method`, source)
                        .toSet();
                });
            case SVType.Error:
                return ctx.warnWithMsg(`from 'LibCall.math.floor: got unknown value`, source).toSet();
            default:
                return ctx
                    .failWithMsg(`from 'LibCall.math.floor: invalid type - ${svTypeToString(num.type)}`, source)
                    .toSet();
        }
    }

    export function abs(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 1) {
            return ctx
                .warnWithMsg(`from 'LibCall.math.abs': got insufficient number of argument: ${params.length}`, source)
                .toSet();
        }

        const heap = ctx.heap;
        const [numAddr] = params;

        const num = fetchAddr(numAddr, heap);

        if (!num) {
            return ctx.failWithMsg(`from 'LibCall.math.abs: got undefined value`, source).toSet();
        }

        switch (num.type) {
            case SVType.Int:
            case SVType.Float: {
                const numExp = num.value;
                if (typeof numExp === 'number') {
                    return ctx.toSetWith(SVInt.create(Math.abs(numExp), source));
                }
                const exp = ExpNum.uop(NumUopType.Abs, numExp, source);
                const retVal = num.type === SVType.Int ? SVInt.create(exp, source) : SVFloat.create(exp, source);
                return ctx.toSetWith(retVal);
            }
            case SVType.Object:
                return ctx.getAttrDeep(num, '__abs__', source).flatMap((ctx) => {
                    const abs = ctx.retVal;
                    if (abs.type === SVType.Func) {
                        return TorchBackend.functionCall(ctx, abs, [], source);
                    }
                    return ctx
                        .warnWithMsg(`from 'LibCall.math.abs: object does not have __abs__ method`, source)
                        .toSet();
                });
            case SVType.Error:
                return ctx.warnWithMsg(`from 'LibCall.math.abs: got unknown value`, source).toSet();
            default:
                return ctx
                    .failWithMsg(`from 'LibCall.math.abs: invalid type - ${svTypeToString(num.type)}`, source)
                    .toSet();
        }
    }

    export function log(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 1) {
            return ctx
                .warnWithMsg(`from 'LibCall.math.log': got insufficient number of argument: ${params.length}`, source)
                .toSet();
        }

        const heap = ctx.heap;
        const [numAddr] = params;

        const num = fetchAddr(numAddr, heap);

        if (!num) {
            return ctx.failWithMsg(`from 'LibCall.math.log': got undefined value`, source).toSet();
        }

        switch (num.type) {
            case SVType.Int:
            case SVType.Float: {
                let numExp = num.value;
                const numRng = ctx.getCachedRange(num.value);
                if (numRng?.isConst()) {
                    numExp = numRng.start;
                }

                if (typeof numExp === 'number') {
                    if (numExp <= 0) {
                        return ctx
                            .warnWithMsg(`from 'LibCall.math.log': math domain error. got ${numExp}`, source)
                            .toSet();
                    }
                    return ctx.toSetWith(SVFloat.create(Math.log(numExp), source));
                }

                const value = SVFloat.create(ExpNum.fromSymbol(ctx.genSymFloat('WarnMathLog', source)), source);
                return ctx
                    .warnWithMsg(`from 'LibCall.math.log': got non-constant value. return symbolic float.`, source)
                    .toSetWith(value);
            }
            case SVType.Object:
                return ctx.getAttrDeep(num, '__abs__', source).flatMap((ctx) => {
                    const abs = ctx.retVal;
                    if (abs.type === SVType.Func) {
                        return TorchBackend.functionCall(ctx, abs, [], source);
                    }
                    return ctx
                        .warnWithMsg(`from 'LibCall.math.abs: object does not have __abs__ method`, source)
                        .toSet();
                });
            case SVType.Error:
                return ctx.warnWithMsg(`from 'LibCall.math.abs: got unknown value`, source).toSet();
            default:
                return ctx
                    .failWithMsg(`from 'LibCall.math.abs: invalid type - ${svTypeToString(num.type)}`, source)
                    .toSet();
        }
    }

    export const libCallImpls: { [key: string]: LCImpl } = {
        floor,
        ceil,
        abs,
        log,
    };
}

export const libCallMap: Map<string, LCImpl> = new Map([...Object.entries(MathLCImpl.libCallImpls)]);
