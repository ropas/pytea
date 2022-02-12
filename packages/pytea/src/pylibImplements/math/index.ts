import { fetchAddr } from '../../backend/backUtils';
import { Context, ContextSet } from '../../backend/context';
import { CodeSource, ShValue, SVFloat, SVInt, SVType, svTypeToString } from '../../backend/sharpValues';
import { ExpNum, NumUopType } from '../../backend/symExpressions';
import { TorchBackend } from '../../backend/torchBackend';
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

    export function float_fun(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.math.float_fun': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }

        const heap = ctx.heap;
        const [numAddr, nameAddr] = params;

        const num = fetchAddr(numAddr, heap);
        const name = fetchAddr(nameAddr, heap);

        if (!num || name?.type !== SVType.String || typeof name.value !== 'string') {
            return ctx.warnWithMsg(`from 'LibCall.math.float_fun': got undefined value`, source).toSet();
        }

        const funName = name.value;

        switch (num.type) {
            case SVType.Int:
            case SVType.Float: {
                let numExp = num.value;
                const numRng = ctx.getCachedRange(num.value);
                if (numRng?.isConst()) {
                    numExp = numRng.start;
                }

                const oldNum = numExp;
                if (typeof numExp === 'number') {
                    switch (funName) {
                        case 'sqrt':
                            numExp = Math.sqrt(numExp);
                            break;
                        case 'log':
                            numExp = Math.log(numExp);
                            break;
                        case 'log2':
                            numExp = Math.log2(numExp);
                            break;
                        case 'log10':
                            numExp = Math.log10(numExp);
                            break;
                        case 'exp':
                            numExp = Math.exp(numExp);
                            break;
                        case 'expm1':
                            numExp = Math.expm1(numExp);
                            break;
                        default:
                            return ctx
                                .warnWithMsg(
                                    `from 'LibCall.math.float_fun': undefined math function (${funName})`,
                                    source
                                )
                                .toSet();
                    }
                    if (Number.isNaN(numExp)) {
                        return ctx
                            .failWithMsg(
                                `from 'LibCall.math.float_fun': math domain error (${funName}(${oldNum}))`,
                                source
                            )
                            .toSet();
                    }
                    return ctx.toSetWith(SVFloat.create(Math.log(numExp), source));
                }

                const value = SVFloat.create(ExpNum.fromSymbol(ctx.genSymFloat(`Math_${funName}`, source)), source);
                return ctx
                    .warnWithMsg(
                        `from 'LibCall.math.float_fun': got non-constant value. return symbolic float.`,
                        source
                    )
                    .toSetWith(value);
            }
            case SVType.Error:
                return ctx.warnWithMsg(`from 'LibCall.math.float_fun: got unknown value`, source).toSet();
            default:
                return ctx
                    .failWithMsg(`from 'LibCall.math.float_fun: invalid type - ${svTypeToString(num.type)}`, source)
                    .toSet();
        }
    }

    export const libCallImpls: { [key: string]: LCImpl } = {
        floor,
        ceil,
        abs,
        float_fun,
    };
}

export const libCallMap: Map<string, LCImpl> = new Map([...Object.entries(MathLCImpl.libCallImpls)]);
