import { LCImpl } from '.';
import { fetchAddr, sanitizeAddr } from '../../backend/backUtils';
import { Context, ContextSet } from '../../backend/context';
import {
    ShValue,
    SVAddr,
    SVBool,
    SVFloat,
    SVFunc,
    SVInt,
    SVNone,
    SVNotImpl,
    SVString,
    SVType,
} from '../../backend/sharpValues';
import { ExpBool, ExpNum, ExpString } from '../../backend/symExpressions';
import { TorchBackend } from '../../backend/torchBackend';
import { ParseNode } from '../../parser/parseNodes';
import { PytService } from '../../pyt/pytService';
import { LCBase } from './libcall';

export namespace BuiltinsLCImpl {
    export function inject_argument(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
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
            return ctx.warnWithMsg(`from 'LibCall.argparse.inject_argument': got non-object values.`).toSet();
        }

        const argStr = fetchAddr(args.getIndice(0), heap);
        if (argStr?.type !== SVType.String) {
            return ctx
                .failWithMsg(`from 'LibCall.argparse.inject_argument': invalid arg type ${argStr?.type}.`)
                .toSet();
        }

        // TODO: regex for argument validation
        const argNameR = /^-*([-\w]*$)/.exec(argStr.value as string);
        if (!argNameR) {
            return ctx
                .failWithMsg(`from 'LibCall.argparse.inject_argument': invalid arg name ${argStr.value}.`)
                .toSet();
        }

        const options = PytService.getOptions();
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
        const argValue = options.pythonCmdArgs[argName];

        let action: string = '';
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
                default:
                    // TODO: implement other actions like append
                    return ctx.warnWithMsg(`unimplemented action: ${action}`).toSet();
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
    export const libCallImpls: { [key: string]: LCImpl } = {
        inject_argument,
    };
}

export const libCallMap: Map<string, LCImpl> = new Map([...Object.entries(BuiltinsLCImpl.libCallImpls)]);
