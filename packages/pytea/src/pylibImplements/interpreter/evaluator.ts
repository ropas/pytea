import { fetchAddr } from '../../frontend/evalUtils';
import { ThEnv, ThHeap } from '../../frontend/torchEnvironments';
import { TorchInterpreter } from '../../frontend/torchInterpreter';
import { LibCallType, TELibCall } from '../../frontend/torchStatements';
import { ThValue, TVAddr, TVError, TVFunc, TVNone, TVObject, TVString, TVType } from '../../frontend/torchValues';
import { EvalLCContext, EvalLCParamType } from './';
import { EvalLCBase } from './libcall';

export function evalLibCall(env: ThEnv, heap: ThHeap, expr: TELibCall): [ThValue, ThHeap] {
    let newHeap = heap;
    let metaParams: Partial<EvalLCParamType> = {};

    let argTypeMismatch = false;

    // calculate E1, E2, ..., En part
    const paramRaw = expr.params.map((value) => {
        const [calc, nextHeap] = TorchInterpreter.evaluate(env, newHeap, value[1]);
        newHeap = nextHeap;
        return calc;
    });

    // resolve addresses
    const params = paramRaw.map((val) => {
        const obj = fetchAddr(val, newHeap);
        if (!obj) {
            return TVError.create('PyTea Interpreter: Try to access non-existing object.');
        }
        return obj;
    });

    const paramKeys = expr.params.map((value) => value[0]);

    switch (expr.type) {
        case LibCallType.import:
            if (params[0].type !== TVType.String) {
                argTypeMismatch = true;
                break;
            }
            metaParams = {
                qualPath: (params[0] as TVString).value,
            } as Partial<EvalLCBase.EImportParams>;
            break;
        case LibCallType.importQualified:
            if (params[0].type !== TVType.String) {
                argTypeMismatch = true;
                break;
            }
            metaParams = {
                qualPath: (params[0] as TVString).value,
            } as Partial<EvalLCBase.EImportQualifiedParams>;
            break;
        case LibCallType.super:
            {
                const mp: Partial<EvalLCBase.ESuperParams> = {};
                paramKeys.forEach((key, i) => {
                    const arg = params[i];
                    switch (key) {
                        case 'self':
                            if (arg.type === TVType.Object) {
                                let addr: TVAddr | undefined = paramRaw[i] as TVAddr;
                                while (addr && addr.addr && newHeap.getVal(addr)?.type !== TVType.Object) {
                                    addr = newHeap.getVal(addr) as TVAddr;
                                }
                                if (addr) {
                                    mp.selfAddr = addr;
                                    mp.self = newHeap.getVal(addr) as TVObject;
                                } else {
                                    argTypeMismatch = true;
                                }
                            } else {
                                mp.self = TVNone.create();
                            }
                            break;
                        case 'baseClass':
                            if (arg.type === TVType.Object) {
                                mp.baseClass = arg;
                            } else {
                                argTypeMismatch = true;
                            }
                            break;
                        default:
                            argTypeMismatch = true;
                            break;
                    }
                });
                metaParams = mp;
            }
            break;
        case LibCallType.genList:
            // Do nothing.
            break;
        case LibCallType.setDefault:
            {
                const mp: Partial<EvalLCBase.ESetDefaultParams> = {};
                const defaults: { [paramName: string]: ThValue } = {};

                paramKeys.forEach((key, i) => {
                    switch (key) {
                        case '$func':
                            // TODO: check params[i] is not TVFunc
                            mp.$func = params[i] as TVFunc;
                            break;
                        case '$varargsName':
                            mp.$varargsName = (params[i] as TVString).value;
                            break;
                        case '$kwargsName':
                            mp.$kwargsName = (params[i] as TVString).value;
                            break;
                        default:
                            defaults[key] = params[i];
                            break;
                    }
                });

                mp.defaults = defaults;
                metaParams = mp;
            }
            break;
        case LibCallType.callKV:
            {
                const mp: Partial<EvalLCBase.ECallKVParams> = {};
                const args: ThValue[] = [];
                const kwargs: { [paramName: string]: ThValue } = {};

                paramKeys.forEach((key, i) => {
                    const arg = params[i];
                    const rawArg = paramRaw[i];

                    switch (key) {
                        case '$func':
                            // TODO: check params[i] is not TVFunc
                            if (params[0].type !== TVType.Func) {
                                argTypeMismatch = true;
                                break;
                            }
                            mp.$func = arg as TVFunc;
                            break;
                        case '$kwargs':
                            if (arg.type === TVType.Object) {
                                arg.keyValues.forEach((v, k) => {
                                    kwargs[k] = v;
                                });
                            } else {
                                argTypeMismatch = true;
                            }
                            break;
                        case '$varargs':
                            if (arg.type === TVType.Object) {
                                let maxLen = 0;
                                arg.indices.forEach((_, i) => {
                                    if (i > maxLen) {
                                        maxLen = i;
                                    }
                                });
                                const objs: ThValue[] = Array(maxLen).fill(TVNone.create());
                                arg.indices.forEach((v, i) => {
                                    objs[i] = v;
                                });
                                objs.forEach((o) => args.push(o));
                            } else {
                                argTypeMismatch = true;
                            }
                            break;
                        case '':
                            args.push(rawArg);
                            break;
                        default:
                            kwargs[key] = rawArg;
                            break;
                    }
                });

                mp.args = args;
                mp.$kwargs = kwargs;
                metaParams = mp;
            }
            break;
        case LibCallType.getAttr:
            {
                const mp: Partial<EvalLCBase.EGetAttrParams> = {};
                paramKeys.forEach((key, i) => {
                    const arg = params[i];
                    switch (key) {
                        case 'name':
                            if (arg.type === TVType.String) {
                                mp.name = arg.value;
                            } else {
                                argTypeMismatch = true;
                            }
                            break;
                        case 'self':
                            mp.self = arg;
                            if (arg.type === TVType.Object) {
                                let addr: TVAddr | undefined = paramRaw[i] as TVAddr;
                                while (addr && addr.addr && newHeap.getVal(addr)?.type !== TVType.Object) {
                                    addr = newHeap.getVal(addr) as TVAddr;
                                }
                                if (addr) {
                                    mp.selfAddr = addr;
                                } else {
                                    argTypeMismatch = true;
                                }
                            }
                            break;
                        case 'baseClass':
                            if (arg.type === TVType.Object) {
                                mp.baseClass = arg;
                            } else {
                                argTypeMismatch = true;
                            }
                            break;
                        case 'bind':
                            if (arg.type === TVType.Bool) {
                                mp.bind = arg.value;
                            } else {
                                argTypeMismatch = true;
                            }
                            break;
                        default:
                            argTypeMismatch = true;
                            break;
                    }
                });
                metaParams = mp;
            }
            break;
        default:
            // TODO: resolve explicit.
            break;
    }

    if (argTypeMismatch) {
        return [TVError.create(`PyTea Interpreter: LibCall ${expr.type} argument type mismatch`, expr.source), newHeap];
    }

    const impl = EvalLCBase.libCallImpls[expr.type];
    const ctx: EvalLCContext = { env, heap: newHeap, params };

    if (impl) {
        const result = impl(ctx, metaParams as EvalLCParamType, expr.source);
        return [result.result, result.heap];
    } else {
        return [TVError.create(`PyTea Interpreter: Cannot find LibCall ${expr.type}`, expr.source), newHeap];
    }
}
