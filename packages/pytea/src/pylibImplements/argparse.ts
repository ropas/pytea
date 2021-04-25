import { fetchAddr, sanitizeAddr } from '../backend/backUtils';
import { Context, ContextSet } from '../backend/context';
import {
    CodeSource,
    ShValue,
    SVAddr,
    SVBool,
    SVFloat,
    SVFunc,
    SVInt,
    SVNone,
    SVString,
    SVType,
} from '../backend/sharpValues';
import { ExpBool, ExpNum, ExpString } from '../backend/symExpressions';
import { TorchBackend } from '../backend/torchBackend';
import { PyteaService } from '../service/pyteaService';
import { LCImpl } from '.';
import { LCBase } from './libcall';

export namespace BuiltinsLCImpl {
    export function inject_argument(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx
                .failWithMsg(
                    `from 'LibCall.argparse.inject_argument': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }

        const heap = ctx.heap;
        const [namespaceAddr, argsAddr, kwargsAddr] = params;

        const namespace = sanitizeAddr(namespaceAddr, heap);
        const nsObj = fetchAddr(namespace, heap);
        const args = fetchAddr(argsAddr, heap);
        const kwargs = fetchAddr(kwargsAddr, heap);

        if (
            namespace?.type !== SVType.Addr ||
            nsObj?.type !== SVType.Object ||
            args?.type !== SVType.Object ||
            kwargs?.type !== SVType.Object
        ) {
            return ctx.warnWithMsg(`from 'LibCall.argparse.inject_argument': got non-object values.`, source).toSet();
        }

        let argStr: string;
        const dest = fetchAddr(kwargs.getKeyVal('dest'), heap);
        if (dest?.type === SVType.String && typeof dest.value === 'string') {
            argStr = dest.value;
        } else {
            const arg0 = fetchAddr(args.getIndice(0), heap);
            if (arg0?.type !== SVType.String || typeof arg0.value !== 'string') {
                return ctx
                    .failWithMsg(
                        `from 'LibCall.argparse.inject_argument': invalid arg type ${arg0?.toString()}.`,
                        source
                    )
                    .toSet();
            }
            argStr = arg0.value;
        }

        // ignore short argument

        if (argStr.startsWith('-') && !argStr.startsWith('--')) {
            const arg1 = fetchAddr(args.getIndice(1), heap);
            if (arg1?.type !== SVType.String || typeof arg1.value !== 'string') {
                return ctx
                    .failWithMsg(
                        `from 'LibCall.argparse.inject_argument': invalid arg type ${arg1?.toString()}.`,
                        source
                    )
                    .toSet();
            }
            argStr = arg1.value;
        }

        // TODO: regex for argument validation
        const argNameR = /^-*([-\w]*$)/.exec(argStr);
        if (!argNameR) {
            return ctx
                .failWithMsg(`from 'LibCall.argparse.inject_argument': invalid arg name ${argStr}.`, source)
                .toSet();
        }

        const cmdArgs = PyteaService.getCmdArgs();
        const argName = argNameR[1];
        const argPyName = argName.replace(/-/g, '_');

        const argTypeAddr = sanitizeAddr(kwargs.getKeyVal('type'), heap);
        let argType: SVType = SVType.String;
        if (argTypeAddr?.type === SVType.Addr) {
            // in builtins.ts, each primitive type object is an instance of _Primitive class.
            // cast to argType
            const argTypeVal = fetchAddr(argTypeAddr, heap);
            if (argTypeVal?.type === SVType.Object) {
                if ((sanitizeAddr(ctx.env.getId('int'), heap) as SVAddr).addr === argTypeAddr.addr) {
                    argType = SVType.Int;
                } else if ((sanitizeAddr(ctx.env.getId('float'), heap) as SVAddr).addr === argTypeAddr.addr) {
                    argType = SVType.Float;
                } else if ((sanitizeAddr(ctx.env.getId('bool'), heap) as SVAddr).addr === argTypeAddr.addr) {
                    argType = SVType.Bool;
                }
            }
            // TODO: what if not? (or argTypeVal is a callable function?)
        } else if (argTypeAddr?.type === SVType.Func) {
            argType = SVType.Func;
        }

        const argDefault = fetchAddr(kwargs.getKeyVal('default'), heap);
        const argAction = fetchAddr(kwargs.getKeyVal('action'), heap);
        const argConst = sanitizeAddr(kwargs.getKeyVal('const'), heap);
        const required = fetchAddr(kwargs.getKeyVal('required'), heap);
        const argValue = cmdArgs[argName];

        let action = '';
        if (argAction?.type === SVType.String) {
            action = argAction.value as string;
            switch (action) {
                case 'store_true':
                    return ctx
                        .setRetVal(SVNone.create())
                        .setHeap(
                            heap.setVal(
                                namespace,
                                nsObj.setAttr(argPyName, SVBool.create(argValue !== undefined, source))
                            )
                        )
                        .toSet();
                case 'store_false':
                    return ctx
                        .setRetVal(SVNone.create())
                        .setHeap(
                            heap.setVal(
                                namespace,
                                nsObj.setAttr(argPyName, SVBool.create(argValue === undefined, source))
                            )
                        )
                        .toSet();
                case 'store_const':
                    if (argConst)
                        return ctx
                            .setRetVal(SVNone.create())
                            .setHeap(heap.setVal(namespace, nsObj.setAttr(argPyName, argConst)))
                            .toSet();
                // eslint-disable-next-line no-fallthrough
                default:
                    // TODO: implement other actions like append
                    return ctx.warnWithMsg(`unimplemented action: ${action}`, source).toSet();
            }
        }

        // TODO: should notify typo on argument name?
        if (argValue !== undefined) {
            let setVal: ShValue;
            switch (argType) {
                case SVType.Func:
                    return TorchBackend.functionCall(
                        ctx,
                        argTypeAddr as SVFunc,
                        [SVString.create(`${argValue}`, source)],
                        source
                    );
                case SVType.Int:
                    setVal = SVInt.create(Number.parseInt(`${argValue}`), source);
                    break;
                case SVType.Float:
                    setVal = SVFloat.create(Number.parseFloat(`${argValue}`), source);
                    break;
                case SVType.Bool:
                    // TODO:  really?
                    setVal = SVBool.create(`${argValue}` === 'true', source);
                    break;
                default:
                    // fall back to string
                    setVal = SVString.create(`${argValue}`, source);
            }
            return ctx
                .setRetVal(SVNone.create())
                .setHeap(heap.setVal(namespace, nsObj.setAttr(argPyName, setVal)))
                .toSet();
        } else if (argDefault) {
            // found default value.
            return ctx
                .setRetVal(SVNone.create())
                .setHeap(heap.setVal(namespace, nsObj.setAttr(argPyName, argDefault)))
                .toSet();
        } else if (required?.type === SVType.Bool && required.value === false) {
            // set non-required input to default value
            let setVal: ShValue;
            switch (argType) {
                case SVType.Func:
                    return TorchBackend.functionCall(ctx, argTypeAddr as SVFunc, [SVString.create('', source)], source);
                case SVType.Int:
                    setVal = SVInt.create(0, source);
                    break;
                case SVType.Float:
                    setVal = SVFloat.create(0, source);
                    break;
                case SVType.Bool:
                    setVal = SVBool.create(false, source);
                    break;
                default:
                    // fall back to string
                    setVal = SVString.create('', source);
            }
            return ctx
                .setRetVal(SVNone.create())
                .setHeap(heap.setVal(namespace, nsObj.setAttr(argPyName, setVal)))
                .toSet();
        } else {
            // not found value. log and use symbolic value.
            let setVal: ShValue;
            switch (argType) {
                case SVType.Func:
                    return TorchBackend.functionCall(
                        ctx,
                        argTypeAddr as SVFunc,
                        [SVString.create(`${argValue}`, source)],
                        source
                    );
                case SVType.Int:
                    setVal = SVInt.create(ExpNum.fromSymbol(ctx.genSymInt(`arg_${argPyName}`, source)), source);
                    break;
                case SVType.Float:
                    setVal = SVFloat.create(ExpNum.fromSymbol(ctx.genSymFloat(`arg_${argPyName}`, source)), source);
                    break;
                case SVType.Bool:
                    // TODO:  really?
                    setVal = SVBool.create(ExpBool.fromSymbol(ctx.genSymBool(`arg_${argPyName}`, source)), source);
                    break;
                default:
                    // fall back to string
                    setVal = SVString.create(
                        ExpString.fromSymbol(ctx.genSymString(`arg_${argPyName}`, source)),
                        source
                    );
            }
            return ctx
                .addLog(
                    `from 'LibCall.argparse.inject_argument': value not found from default or pythonCmdArgs. use symbolic values for arg ${argPyName}`,
                    source
                )
                .setRetVal(SVNone.create())
                .setHeap(heap.setVal(namespace, nsObj.setAttr(argPyName, setVal)))
                .toSet();
        }
    }

    // return current subcommand
    export function set_subcommand(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx
                .failWithMsg(
                    `from 'LibCall.argparse.set_subcommand': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }

        const heap = ctx.heap;
        const [selfAddrx, kwargsAddr] = params;

        const selfAddr = sanitizeAddr(selfAddrx, heap);
        const self = fetchAddr(selfAddr, heap);
        const kwargs = fetchAddr(kwargsAddr, heap);

        if (selfAddr?.type !== SVType.Addr || self?.type !== SVType.Object || kwargs?.type !== SVType.Object) {
            return ctx.warnWithMsg(`from 'LibCall.argparse.set_subcommand': got non-object values.`, source).toSet();
        }

        const subcmd = PyteaService.getSubcommand();

        if (!subcmd) {
            return ctx.toSetWith(SVNone.create(source));
        }

        const dest = fetchAddr(kwargs.getKeyVal('dest'), heap);

        let newCtx = ctx;
        const subcmdVal = SVString.create(subcmd, source);

        if (dest && dest.type === SVType.String && typeof dest.value === 'string') {
            const namespace = sanitizeAddr(self.getAttr('parsed'), heap);
            const nsObj = fetchAddr(namespace, heap);

            if (nsObj && nsObj.type === SVType.Object) {
                const newNS = nsObj.setAttr(dest.value, subcmdVal);

                if (namespace?.type === SVType.Addr) {
                    const newHeap = ctx.heap.setVal(namespace, newNS);
                    newCtx = ctx.setHeap(newHeap);
                } else {
                    const newSelf = self.setAttr('parsed', newNS);
                    const newHeap = ctx.heap.setVal(selfAddr, newSelf);
                    newCtx = ctx.setHeap(newHeap);
                }
            }
        }

        return newCtx.toSetWith(subcmdVal);
    }

    export const libCallImpls: { [key: string]: LCImpl } = {
        inject_argument,
        set_subcommand,
    };
}

export const libCallMap: Map<string, LCImpl> = new Map([...Object.entries(BuiltinsLCImpl.libCallImpls)]);
