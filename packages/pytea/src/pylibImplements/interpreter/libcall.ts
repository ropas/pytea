import { List, Map } from 'immutable';

import { fetchAddr, functionCall } from '../../frontend/evalUtils';
import { ThEnv, ThHeap } from '../../frontend/torchEnvironments';
import { LibCallType, TEConst, TELibCall, TEName, TEObject, TSReturn } from '../../frontend/torchStatements';
import { ThValue, TVAddr, TVError, TVFunc, TVNone, TVObject, TVString, TVType } from '../../frontend/torchValues';
import { ParseNode } from '../../parser/parseNodes';
import { EvalLCContext, EvalLCImpl, EvalLCResult } from '.';

export namespace EvalLCBase {
    export type EBaseParamType =
        | EImportParams
        | EImportQualifiedParams
        | EExportGlobalParams
        | ESuperParams
        | ESetDefaultParams
        | ECallKVParams
        | EDebugParams
        | EGetAttrParams
        | EExplicitParams;

    export interface EExplicitParams {
        params: ThValue[];
    }
    export function explicit(ctx: EvalLCContext, params: EExplicitParams, source?: ParseNode): EvalLCResult {
        // placeholder. explicit call is evaluated in evalLibCall.
        const { heap } = ctx;
        return { heap, result: TVNone.create() };
    }

    export function objectClass(ctx: EvalLCContext, params: EExplicitParams, source?: ParseNode): EvalLCResult {
        const { heap } = ctx;

        let objectClass = TVObject.create();
        const objInit = TVFunc.create('__init__', List(['self']), TSReturn.create(TEConst.genNone()), new ThEnv());
        const objNew = TVFunc.create('__new__', List(['cls']), TSReturn.create(TEObject.create()), new ThEnv());
        objectClass = objectClass.setAttr('__init__', objInit);
        objectClass = objectClass.setAttr('__new__', objNew);

        const [addr, newHeap] = heap.allocNew(objectClass);

        return { heap: newHeap, result: addr };
    }

    // export set $module[globalVar] = address of globalVar
    export interface EExportGlobalParams {
        $module: TVAddr; // points SVObject
        globalVar: string;
    }
    export function exportGlobal(ctx: EvalLCContext, params: EExportGlobalParams, source?: ParseNode): EvalLCResult {
        const addr = params.$module;
        const obj = ctx.heap.getVal(addr) as TVObject;
        const global = ctx.env.getId(params.globalVar) as TVAddr;
        return { heap: ctx.heap.setVal(addr, obj.setAttr(params.globalVar, global)), result: global };
    }

    // return module object.
    export interface EImportParams {
        qualPath: string;
    }
    // import is JS reserved keyword.
    export function thImport(ctx: EvalLCContext, params: EImportParams, source?: ParseNode): EvalLCResult {
        const { heap } = ctx;
        // TODO: implement it.
        return { heap, result: TVNone.create() };
    }

    // return module object and inject it to env.
    export interface EImportQualifiedParams {
        qualPath: string;
    }
    export function importQualified(
        ctx: EvalLCContext,
        params: EImportQualifiedParams,
        source?: ParseNode
    ): EvalLCResult {
        const { heap } = ctx;
        // TODO: implement it.
        return { heap, result: TVNone.create() };
    }

    export interface ESuperParams {
        self: TVObject | TVNone;
        baseClass: TVObject;
        selfAddr?: TVAddr;
    }
    export function thSuper(ctx: EvalLCContext, params: ESuperParams, source?: ParseNode): EvalLCResult {
        const { heap } = ctx;
        const { baseClass, self, selfAddr } = params;

        const mro = baseClass.getAttr('__mro__');

        if (!mro) {
            return { heap, result: TVNone.create() };
        }
        const superClass = fetchAddr(mro, heap);
        if (!superClass || superClass.type !== TVType.Object) {
            return { heap, result: TVNone.create() };
        }
        if (!superClass.getIndice(1)) {
            return { heap, result: TVNone.create() };
        }

        const superObj = superClass.getIndice(1);

        if (self.type === TVType.Object && selfAddr && superObj?.type === TVType.Addr) {
            const baseEnv = new ThEnv().setId('self', selfAddr).setId('baseClass', superObj as TVAddr);
            const superProxy = TVFunc.create(
                'super$__getattr__',
                List(['name']),
                TSReturn.create(
                    TELibCall.create(LibCallType.getAttr, [
                        ['name', TEName.create('name')],
                        ['self', TEName.create('self')],
                        ['baseClass', TEName.create('baseClass')],
                        ['bind', TEConst.genBool(true)],
                    ])
                ),
                baseEnv
            );
            const proxyObj = TVObject.create().setAttr('__getattr__', superProxy);
            const [proxyAddr, newHeap] = heap.allocNew(proxyObj);
            return { heap: newHeap, result: proxyAddr };
        }

        return { heap, result: superObj! };
    }

    export function genList(ctx: EvalLCContext, params: EExplicitParams, source?: ParseNode): EvalLCResult {
        const { heap } = ctx;
        const [listLoc, heap1] = heap.malloc();

        let obj = TVObject.create(source);
        params.params.forEach((v, i) => {
            obj = obj.setIndice(i, v);
        });

        // TODO: set list methods.

        const heap2 = heap1.setVal(listLoc, obj);

        return { heap: heap2, result: obj };
    }

    export interface ESetDefaultParams {
        $func: TVFunc;
        defaults: { [paramName: string]: ThValue };
        $varargsName?: string;
        $kwargsName?: string;
    }
    export function setDefault(ctx: EvalLCContext, params: ESetDefaultParams, source?: ParseNode): EvalLCResult {
        const { heap } = ctx;
        const { $func, defaults, $varargsName, $kwargsName } = params;
        // TODO: implement it.

        let newFunc = $func;

        newFunc = newFunc.setDefaults(Map(defaults));
        newFunc = newFunc.setVKParam($varargsName, $kwargsName);

        return { heap, result: newFunc };
    }

    export interface ECallKVParams {
        $func: TVFunc;
        args: ThValue[];
        $kwargs: { [paramName: string]: ThValue };
    }
    export function callKV(ctx: EvalLCContext, params: ECallKVParams, source?: ParseNode): EvalLCResult {
        const { env, heap } = ctx;
        const { $func, args, $kwargs } = params;

        const retVal = functionCall(env, heap, $func, args, source, $kwargs);

        return { heap: retVal[1], result: retVal[0] };
    }

    export interface EDebugParams {
        params: [ThValue];
    }
    export function DEBUG(ctx: EvalLCContext, params: EDebugParams, source?: ParseNode): EvalLCResult {
        const { heap } = ctx;
        let value: ThValue | undefined = params.params[0];
        while (value?.type === TVType.Addr) {
            value = heap.getVal(value);
        }

        if (value) {
            console.log(value.toString());
        } else {
            console.log(params.params[0].toString());
        }

        return { heap, result: TVNone.create() };
    }

    export interface EGetAttrParams {
        name: string;
        self: ThValue;
        baseClass: TVObject;
        bind: boolean;
        selfAddr?: TVAddr;
    }
    export function getAttr(ctx: EvalLCContext, params: EGetAttrParams, source?: ParseNode): EvalLCResult {
        const { env, heap } = ctx;
        const { bind, selfAddr, name, self, baseClass } = params;
        let newHeap = heap;

        const selfAttr = self.attrs.get(name);
        if (selfAttr) {
            return { heap, result: selfAttr };
        }

        const [superAttr, attrHeap] = _evalAttr(env, newHeap, baseClass, name, source);
        newHeap = attrHeap;

        if (superAttr) {
            // TODO: recognize callable object.
            if (bind && superAttr.type === TVType.Func) {
                let addr = selfAddr;
                if (!addr) {
                    [addr, newHeap] = newHeap.malloc();
                }
                let boundFunc: ThValue | undefined = superAttr.bound(addr);
                if (!boundFunc) {
                    boundFunc = TVError.create(
                        `PyTea Interpreter: Function ${superAttr.name} is not bindable.`,
                        source
                    );
                }

                const selfObj = self as TVObject;
                //newHeap = newHeap.setVal(addr, selfObj.setAttr(name, boundFunc));

                return { heap: newHeap, result: boundFunc };
            } else {
                return { heap, result: superAttr };
            }
            // TODO: args, kwargs
        } else {
            return {
                heap: newHeap,
                result: TVError.create(`PyTea Interpreter: Attribute ${name} does not exist.`, source),
            };
        }
    }

    function _evalAttr(
        env: ThEnv,
        heap: ThHeap,
        object: TVObject,
        name: string,
        source?: ParseNode
    ): [ThValue, ThHeap] {
        const attr = object.getAttr(name);

        if (attr === undefined) {
            const getAttr = object.getAttr('__getattr__');
            if (getAttr && getAttr.type === TVType.Func) {
                return functionCall(env, heap, getAttr, [TVString.create(name, source)], source);
            } else {
                return [TVError.create(`PyTea Interpreter: Atribute ${name} not exist.`, source), heap];
            }
        }
        return [attr, heap];
    }

    export const libCallImpls: { [key in keyof typeof LibCallType]: EvalLCImpl } = {
        import: thImport,
        importQualified,
        super: thSuper,
        genList,
        DEBUG,
        setDefault,
        callKV,
        getAttr,
        explicit,
        exportGlobal,
        objectClass,
    };
}
