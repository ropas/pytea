import * as BackUtils from '../backend/backUtils';
import { Context, ContextSet } from '../backend/context';
import { ShValue, SVAddr, SVFunc, SVInt, SVNone, SVNotImpl, SVString, SVType } from '../backend/sharpValues';
import { TorchBackend } from '../backend/torchBackend';
import { LibCallType, TELibCall } from '../frontend/torchStatements';
import { libCallMap } from './index';
import { LCBase } from './libcall';

export function evalLibCall<T>(ctxSet: ContextSet<T>, expr: TELibCall): ContextSet<ShValue> {
    // calculate E1, E2, ..., En part
    const paramSet = TorchBackend.evalAll(
        ctxSet,
        expr.params.map((p) => p[1])
    );
    const paramKeys = expr.params.map((value) => value[0]);

    const impl = LCBase.libCallImpls[expr.type];
    if (impl === undefined) {
        return paramSet.fail('invalid libcall type', expr.source);
    }

    switch (expr.type) {
        case LibCallType.explicit:
            return paramSet.flatMap((ctx) => {
                const params = ctx.retVal;
                if (params[0]?.type !== SVType.String) {
                    return ctx.failWithMsg(`libcall argument type mismatch: explicit`, expr.source).toSet();
                }
                const libCallName = params[0].value as string;
                const explicitImpl = libCallMap.get(libCallName);

                if (explicitImpl) {
                    const explicitParams = params.slice();
                    explicitParams.shift();
                    return explicitImpl(
                        ctx.setRetVal({ params: explicitParams } as LCBase.ExplicitParams),
                        expr.source
                    );
                } else {
                    return ctx.warnWithMsg(`unimplemented explicit libcall: ${libCallName}`, expr.source).toSet();
                }
            });
        case LibCallType.import:
            return paramSet.flatMap((ctx) => {
                const params = ctx.retVal;

                if (params[0]?.type !== SVType.String) {
                    return ctx.failWithMsg(`libcall argument type mismatch: import`, expr.source).toSet();
                }
                const implParam = { qualPath: params[0].value as string } as LCBase.ImportParams;
                if (params[1]?.type === SVType.String) {
                    implParam.assignTo = params[1].value as string;
                }
                return impl(ctx.setRetVal(implParam), expr.source);
            });
        case LibCallType.objectClass:
            return ctxSet.flatMap((ctx) => impl(ctx.setRetVal({ params: [] }), expr.source));
        case LibCallType.genList:
        case LibCallType.genDict:
            return paramSet.flatMap((ctx) => impl(ctx.setRetVal({ params: ctx.retVal }), expr.source));
        case LibCallType.setDefault:
            return paramSet.flatMap((ctx) => {
                const heap = ctx.heap;
                const params = ctx.retVal;
                const mp: Partial<LCBase.SetDefaultParams> = {};
                const defaults: { [paramName: string]: ShValue } = {};

                for (const i in paramKeys) {
                    const key = paramKeys[i];
                    switch (key) {
                        case '$func':
                            // TODO: check params[i] is not SVFunc
                            mp.$func = BackUtils.fetchAddr(params[i], heap) as SVFunc;
                            break;
                        case '$varargsName':
                            mp.$varargsName = (params[i] as SVString).value as string;
                            break;
                        case '$kwargsName':
                            mp.$kwargsName = (params[i] as SVString).value as string;
                            break;
                        case '$keyOnlyNum':
                            mp.$keyOnlyNum = (params[i] as SVInt).value as number;
                            break;
                        default:
                            defaults[key] = params[i];
                            break;
                    }
                }

                mp.defaults = defaults;

                return impl(ctx.setRetVal(mp as LCBase.SetDefaultParams), expr.source);
            });

        case LibCallType.callKV:
            return paramSet.flatMap((ctx) => {
                const heap = ctx.heap;
                const mp: Partial<LCBase.CallKVParams> = {};
                const params = ctx.retVal;
                const args: ShValue[] = [];
                const kwargs: { [paramName: string]: ShValue } = {};

                let ctxSet: ContextSet<ShValue> | undefined;

                for (const i in paramKeys) {
                    const key = paramKeys[i];
                    const addr = params[i];
                    const arg = BackUtils.fetchAddr(addr, heap);

                    switch (key) {
                        case '$func':
                            // TODO: check params[i] is not SVFunc
                            if (arg?.type === SVType.Func) {
                                mp.$func = arg;
                            } else if (arg?.type === SVType.Object) {
                                ctxSet = TorchBackend.getAttrDeep(ctx, arg, '__call__', expr.source).map((ctx) => {
                                    const func = BackUtils.fetchAddr(ctx.retVal, ctx.heap);
                                    if (func?.type === SVType.Func) {
                                        return ctx.setRetVal(func);
                                    } else {
                                        return ctx.warnWithMsg(
                                            `object is not callable`,
                                            expr.source
                                        ) as Context<ShValue>;
                                    }
                                });
                            }
                            break;
                        case '$kwargs':
                            if (arg?.type === SVType.Object) {
                                arg.keyValues.forEach((v, k) => {
                                    kwargs[k] = v;
                                });
                            }
                            // TODO: ignore?
                            break;
                        case '$varargs':
                            // TODO: tensor / getitem / toList / etc ...
                            if (arg?.type === SVType.Object) {
                                let maxLen = 0;
                                arg.indices.forEach((_, i) => {
                                    if (i > maxLen) {
                                        maxLen = i;
                                    }
                                });
                                const objs: ShValue[] = Array(maxLen).fill(SVNone.create());
                                arg.indices.forEach((v, i) => {
                                    objs[i] = v;
                                });
                                objs.forEach((o) => args.push(o));
                            }
                            // TODO: ignore?
                            break;
                        case '':
                            args.push(addr);
                            break;
                        default:
                            kwargs[key] = addr;
                            break;
                    }
                }

                mp.$args = args;
                mp.$kwargs = kwargs;

                if (ctxSet) {
                    return ctxSet.flatMap((ctx) => {
                        if (ctx.retVal.type === SVType.Func) {
                            return impl(
                                ctx.setRetVal({ ...mp, $func: ctx.retVal } as LCBase.CallKVParams),
                                expr.source
                            );
                        }
                        // propagate warning if not a function
                        return ctx.toSet();
                    });
                }

                return impl(ctx.setRetVal(mp as LCBase.CallKVParams), expr.source);
            });

        case LibCallType.exportGlobal:
            return paramSet.flatMap((ctx) => {
                const params = ctx.retVal;
                const mp: LCBase.ExportGlobalParams = {
                    $module: params[0] as SVAddr,
                    globalVar: paramKeys[1],
                };

                return impl(ctx.setRetVal(mp), expr.source);
            });
        case LibCallType.raise:
            return paramSet.flatMap((ctx) => {
                return impl(ctx.setRetVal({ value: ctx.retVal[0] as ShValue }), expr.source);
            });
        default:
            // TODO: resolve explicit.
            return paramSet.return(SVNotImpl.create(`not implemented libcall`, expr.source));
    }
}
