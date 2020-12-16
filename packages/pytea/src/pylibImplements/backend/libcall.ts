import { List, Map as IMMap } from 'immutable';

import * as BackUtils from '../../backend/backUtils';
import { Context, ContextSet } from '../../backend/context';
import { ShEnv } from '../../backend/sharpEnvironments';
import { ShValue, SVAddr, SVFunc, SVInt, SVNone, SVObject, SVString, SVType, SVUndef } from '../../backend/sharpValues';
import { TorchBackend } from '../../backend/torchBackend';
import { LibCallType, TEConst, TELibCall, TEName, TEObject, TSLet, TSReturn } from '../../frontend/torchStatements';
import { ParseNode } from '../../parser/parseNodes';
import { PytService } from '../../pyt/pytService';
import * as PytUtils from '../../pyt/pytUtils';
import { LCImpl } from '.';
import { SymExp } from '../../backend/symExpressions';

export namespace LCBase {
    export type BaseParamType =
        | ImportParams
        | SuperParams
        | SetDefaultParams
        | CallKVParams
        | GetAttrParams
        | ExportGlobalParams
        | RaiseParams
        | ExplicitParams;

    export interface ExplicitParams {
        params: ShValue[];
    }
    export function explicit(ctx: Context<ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
        // placeholder. explicit call is evaluated in evalLibCall.
        return ctx.warnWithMsg('unimplemented libcall', source).toSet();
    }

    // return new class object of `object`
    export function objectClass(ctx: Context<ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
        const { heap } = ctx;

        let [objectClass, objectAddr, newHeap] = SVObject.create(heap, source);

        const objInit = SVFunc.create('__init__', List(['self']), TSReturn.create(TEConst.genNone()), new ShEnv());
        const objNew = SVFunc.create('__new__', List(['cls']), TSReturn.create(TEObject.create()), new ShEnv());
        objectClass = objectClass.setAttr('__init__', objInit);
        objectClass = objectClass.setAttr('__new__', objNew);

        return ctx.setHeap(newHeap.setVal(objectAddr, objectClass)).toSetWith(objectAddr);
    }

    // return module object.
    export interface ImportParams {
        qualPath: string;
        assignTo?: string;
    }
    // import is JS reserved keyword.
    export function thImport(ctx: Context<ImportParams>, source?: ParseNode): ContextSet<ShValue> {
        const service = PytService.getGlobalService();
        if (!service) {
            return ctx.failWithMsg('PyTea service uninitialized.', source).toSet();
        }
        const assignTo = ctx.retVal.assignTo;
        const qualPath = ctx.retVal.qualPath;
        const currPath = ctx.relPath;

        const paths = PytUtils.scanQualPath(qualPath, currPath);

        let baseEnv = ctx.env;
        const basePath = ctx.relPath;

        const lastPath = paths[paths.length - 1];
        const hasWild = lastPath.endsWith('*');
        const wildId = paths.length - 1;

        let ctxSet: ContextSet<any> = ctx.toSet();
        paths.forEach((libRelPath, index) => {
            // import wildcard
            if (hasWild && libRelPath.endsWith('*')) {
                return;
            }

            // do not import twice
            ctxSet = ctxSet.flatMap((ctx) => {
                const currEnv = ctx.env;
                let imported = ctx.imported;

                if (imported.addrMap.has(libRelPath)) {
                    return ctx.toSet();
                }

                const [stmt, isInit] = service.getImportModuleStmt(libRelPath);
                if (!stmt) return ctx.toSet();

                const newPath = isInit ? `${libRelPath}.__init__` : libRelPath;

                const [moduleAddr, moduleHeap0] = ctx.heap.allocNew(SVUndef.create(source), source);
                const [nameAddr, moduleHeap] = moduleHeap0.allocNew(SVString.create(libRelPath), source);
                imported = imported.setId(libRelPath, moduleAddr);

                const defaultEnv = currEnv.set(
                    'addrMap',
                    currEnv.addrMap.filter((v) => v.addr < 0).set('__name__', nameAddr)
                );
                const newCtx = ctx.setEnv(defaultEnv).setHeap(moduleHeap).setImported(imported).setRelPath(newPath);
                const ctxSet = TorchBackend.runModule(stmt as TSLet, newPath, newCtx);

                return ctxSet.map((ctx) => {
                    const env = ctx.env;
                    let heap = ctx.heap;
                    let imported = ctx.imported;
                    env.addrMap.forEach((addr, id) => {
                        if (addr.addr >= 0) {
                            imported = imported.setId(`${libRelPath}.${id}`, addr);
                        }
                    });
                    heap = heap.setVal(moduleAddr, ctx.retVal);

                    if (hasWild && index === wildId - 1) {
                        baseEnv = baseEnv.mergeAddr(env);
                        const moduleBase = baseEnv.getId('$module');
                        const moduleAddr = BackUtils.sanitizeAddr(moduleBase, heap);
                        // assign imported variables to $module.
                        // (wildcard import does not controlled by frontend)
                        if (moduleAddr && moduleAddr.type === SVType.Addr) {
                            const moduleObj = heap.getVal(moduleAddr);
                            if (moduleObj && moduleObj.type === SVType.Object) {
                                let obj: SVObject = moduleObj;
                                env.addrMap.forEach((addr, id) => {
                                    if (addr.addr >= 0) {
                                        obj = obj.setAttr(id, addr);
                                    }
                                });
                                heap = heap.setVal(moduleAddr, obj);
                            }
                        }
                    }

                    return ctx.setImported(imported).setEnv(currEnv).setHeap(heap);
                });
            });
        });

        // TODO: Garbage collection

        return ctxSet.map((ctx) => {
            let newCtx = ctx.setEnv(baseEnv).setRelPath(basePath);
            const retVal = ctx.imported.getId(lastPath);

            if (assignTo) {
                const to = baseEnv.getId(assignTo);
                const val = BackUtils.sanitizeAddr(retVal, newCtx.heap);
                if (to) {
                    if (val && val.type !== SVType.Undef) {
                        newCtx = newCtx.setHeap(newCtx.heap.setVal(to, val));
                    } else if (retVal) {
                        newCtx = newCtx.setHeap(newCtx.heap.setVal(to, retVal));
                    }
                }
            }

            if (retVal) {
                return newCtx.setRetVal(retVal);
            } else if (hasWild) {
                return newCtx.setRetVal(SVNone.create());
            } else {
                return newCtx.warnWithMsg(`import ${qualPath}(${lastPath}) failed `, source) as Context<ShValue>;
            }
        });
    }

    // return module object and inject it to env.
    export function importQualified(ctx: Context<ImportParams>, source?: ParseNode): ContextSet<ShValue> {
        // TODO: qualified.
        const service = PytService.getGlobalService();
        if (!service) {
            return ctx.failWithMsg('PyTea service uninitialized.', source).toSet();
        }
        const qualPath = ctx.retVal.qualPath;
        const currPath = ctx.relPath;

        const paths = PytUtils.scanQualPath(qualPath, currPath);

        let baseEnv = ctx.env;
        const basePath = ctx.relPath;

        const lastPath = paths[paths.length - 1];
        const hasWild = lastPath.endsWith('*');
        const wildId = paths.length - 1;

        let ctxSet: ContextSet<any> = ctx.toSet();
        paths.forEach((libRelPath, index) => {
            // import wildcard
            if (hasWild && libRelPath.endsWith('*')) {
                return;
            }

            // do not import twice
            ctxSet = ctxSet.flatMap((ctx) => {
                const currEnv = ctx.env;
                const imported = ctx.imported;
                if (imported.addrMap.has(libRelPath)) return ctx.toSet();

                const [stmt, isInit] = service.getImportModuleStmt(libRelPath);
                if (!stmt) return ctx.toSet();

                const newPath = `${libRelPath}.__init__`;

                const newCtx = ctx.setEnv(new ShEnv()).setRelPath(newPath);
                const ctxSet = TorchBackend.runModule(stmt as TSLet, newPath, newCtx);
                return ctxSet.map((ctx) => {
                    const env = ctx.env;
                    let imported = ctx.imported;
                    env.addrMap.forEach((addr, id) => {
                        imported = imported.setId(`${libRelPath}.${id}`, addr);
                    });
                    imported = imported.setId(libRelPath, ctx.retVal);

                    if (hasWild && index === wildId - 1) {
                        let heap = ctx.heap;
                        baseEnv = baseEnv.mergeAddr(env);
                        const moduleBase = baseEnv.getId('$module');
                        const moduleAddr = BackUtils.sanitizeAddr(moduleBase, heap);
                        // assign imported variables to $module.
                        // (wildcard import does not controlled by frontend)
                        if (moduleAddr && moduleAddr.type === SVType.Addr) {
                            const moduleObj = heap.getVal(moduleAddr);
                            if (moduleObj && moduleObj.type === SVType.Object) {
                                let obj: SVObject = moduleObj;
                                env.addrMap.forEach((addr, id) => {
                                    obj = obj.setAttr(id, addr);
                                });
                                heap = heap.setVal(moduleAddr, obj);
                            }
                        }
                        return ctx.setImported(imported).setEnv(currEnv).setHeap(heap);
                    }

                    return ctx.setImported(imported).setEnv(currEnv);
                });
            });
        });

        // TODO: Garbage collection

        return ctxSet.map((ctx) => {
            const newCtx = ctx.setEnv(baseEnv).setRelPath(basePath);
            const retVal = ctx.imported.getId(lastPath);
            if (retVal) {
                return newCtx.setRetVal(retVal);
            } else {
                return newCtx.warnWithMsg(`import ${qualPath}(${lastPath}) failed `, source) as Context<ShValue>;
            }
        });
    }

    export interface SuperParams {
        self: SVObject | SVNone;
        baseClass: SVObject;
        selfAddr?: SVAddr;
    }
    export function thSuper(ctx: Context<SuperParams>, source?: ParseNode): ContextSet<ShValue> {
        const { heap } = ctx;
        const { baseClass, self, selfAddr } = ctx.retVal;

        const mro = baseClass.getAttr('__mro__');

        if (!mro) {
            return ctx.toSetWith(SVNone.create());
        }
        const superClass = BackUtils.fetchAddr(mro, heap);
        if (!superClass || superClass.type !== SVType.Object) {
            return ctx.toSetWith(SVNone.create());
        }

        const superObj = superClass.getIndice(1);
        if (!superObj) {
            return ctx.toSetWith(SVNone.create());
        }

        if (self.type === SVType.Object && selfAddr && superObj.type === SVType.Addr) {
            const baseEnv = new ShEnv().setId('self', selfAddr).setId('baseClass', superObj as SVAddr);
            const superProxy = SVFunc.create(
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
            let [proxyObj, proxyAddr, newHeap] = SVObject.create(heap, source);
            proxyObj = proxyObj.setAttr('__getattr__', superProxy);

            return ctx.setHeap(newHeap.setVal(proxyAddr, proxyObj)).toSetWith(proxyAddr);
        }

        return ctx.toSetWith(superObj);
    }

    export function genList(ctx: Context<ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
        const { heap } = ctx;
        const params = ctx.retVal.params;

        // TODO: flattening varargs
        let [obj, objAddr, newHeap] = SVObject.create(heap, source);
        params.forEach((v, i) => {
            obj = obj.setIndice(i, v);
        });
        obj = obj.setAttr('$length', SVInt.create(params.length, source));

        let listType = BackUtils.fetchAddr(ctx.env.getId('list'), ctx.heap);
        if (listType?.type === SVType.Object) {
            // class _Primitives defines self.mro = (self, object)
            listType = BackUtils.fetchAddr(listType.getAttr('__mro__'), ctx.heap);
        }

        if (listType?.type === SVType.Object) {
            obj = obj.setAttr('__mro__', listType);
        }

        return ctx.setHeap(newHeap.setVal(objAddr, obj)).toSetWith(objAddr);
    }

    export function genDict(ctx: Context<ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
        // TODO: genDict
        const { heap } = ctx;
        const params = ctx.retVal.params;
        const [listLoc, heap1] = heap.malloc();

        let [obj, objAddr, newHeap] = SVObject.create(heap, source);

        for (let i in params) {
            const kvTuple = BackUtils.fetchAddr(params[i], heap);
            if (kvTuple?.type !== SVType.Object) {
                return ctx.warnWithMsg(`from 'LibCall.genDict: parameter must be key-value tuple`, source).toSet();
            }
            const key = BackUtils.fetchAddr(kvTuple.getIndice(0), heap);
            const value = kvTuple.getIndice(1);
            if (!key || !value) {
                return ctx.warnWithMsg(`from 'LibCall.genDict: parameter must be key-value tuple`, source).toSet();
            }

            // TODO: hash object + symbolic something
            if (key.type === SVType.Int) {
                const keyRng = ctx.getCachedRange(key.value);
                if (keyRng?.toIntRange()?.isConst()) obj = obj.setIndice(keyRng.start, value);
            } else if (key.type === SVType.String && typeof key.value === 'string') {
                obj = obj.setKeyVal(key.value, value);
            }
        }
        obj = obj.setAttr('$length', SVInt.create(params.length, source));

        return ctx.setHeap(newHeap.setVal(objAddr, obj)).toSetWith(obj);
    }

    export interface SetDefaultParams {
        $func: SVFunc;
        defaults: { [paramName: string]: ShValue };
        $varargsName?: string;
        $kwargsName?: string;
        $keyOnlyNum?: number;
    }
    export function setDefault(ctx: Context<SetDefaultParams>, source?: ParseNode): ContextSet<ShValue> {
        const { $func, defaults, $varargsName, $kwargsName, $keyOnlyNum } = ctx.retVal;

        let newFunc = $func;

        newFunc = newFunc.setDefaults(IMMap(defaults));
        newFunc = newFunc.setVKParam($varargsName, $kwargsName, $keyOnlyNum);

        return ctx.toSetWith(newFunc);
    }

    export interface CallKVParams {
        $func: SVFunc;
        args: ShValue[];
        $kwargs: { [paramName: string]: ShValue };
    }
    export function callKV(ctx: Context<CallKVParams>, source?: ParseNode): ContextSet<ShValue> {
        const { $func, args, $kwargs } = ctx.retVal;

        return TorchBackend.functionCall(ctx, $func, args, source, $kwargs);
    }

    export function DEBUG(ctx: Context<ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
        const heap = ctx.heap;
        const params = ctx.retVal.params;

        const value: ShValue | undefined = BackUtils.fetchAddr(params[0], heap);
        let newCtx: Context<ExplicitParams>;

        PytService.log(value?.toString());

        if (value) {
            newCtx = ctx.addLogValue(value);
        } else {
            newCtx = ctx.addLogValue(params[0] ? params[0] : SVNone.create());
        }

        return newCtx.toSetWith(SVNone.create());
    }

    // export set $module[globalVar] = address of globalVar
    export interface ExportGlobalParams {
        $module: SVAddr; // points SVObject
        globalVar: string;
    }
    export function exportGlobal(ctx: Context<ExportGlobalParams>, source?: ParseNode): ContextSet<ShValue> {
        const addr = ctx.retVal.$module;
        const obj = ctx.heap.getVal(addr) as SVObject;
        const global = ctx.env.getId(ctx.retVal.globalVar) as SVAddr;
        return ctx.setHeap(ctx.heap.setVal(addr, obj.setAttr(ctx.retVal.globalVar, global))).toSetWith(global);
    }

    export interface GetAttrParams {
        name: string;
        self: ShValue;
        baseClass: SVObject;
        bind: boolean;
        selfAddr?: SVAddr;
    }
    export function getAttr(ctx: Context<GetAttrParams>, source?: ParseNode): ContextSet<ShValue> {
        const { bind, selfAddr, name, self, baseClass } = ctx.retVal;

        const selfAttr = self.attrs.get(name);
        if (selfAttr) {
            return ctx.setRetVal(selfAttr).toSet();
        }

        return TorchBackend.getAttrDeep(ctx, baseClass, name, source).map((ctx) => {
            let newHeap = ctx.heap;
            const superAttr = ctx.retVal;

            if (bind && superAttr.type === SVType.Func) {
                if (self.type !== SVType.Object) {
                    return ctx.warnWithMsg(`value is not an object. (not bindable)`, source);
                }

                let addr = selfAddr;
                if (!addr) {
                    [addr, newHeap] = newHeap.malloc();
                }

                const boundFunc = superAttr.bound(addr);
                if (!boundFunc) {
                    return ctx.warnWithMsg(`Function ${superAttr.name} is not bindable. (no parameters)`, source);
                }

                return ctx.setRetVal(boundFunc);
            } else {
                return ctx.setRetVal(superAttr);
            }
        });
    }
    export interface RaiseParams {
        value: ShValue;
    }
    export function raise(ctx: Context<RaiseParams>, source?: ParseNode): ContextSet<ShValue> {
        const raisedVal = BackUtils.fetchAddr(ctx.retVal.value, ctx.heap);

        if (raisedVal?.type === SVType.Object) {
            return ctx.getAttrDeep(raisedVal, '__name__', source).flatMap((ctx) => {
                if (ctx.retVal?.type === SVType.String) {
                    const errType = ctx.retVal.value;

                    return ctx.getAttrDeep(raisedVal, 'args', source).flatMap((ctx) => {
                        let errMsg = '';
                        const args = BackUtils.fetchAddr(ctx.retVal, ctx.heap);
                        if (args?.type === SVType.Object) {
                            const msg = BackUtils.fetchAddr(args.getIndice(0), ctx.heap);
                            if (msg?.type === SVType.String) {
                                errMsg = `: ${SymExp.toString(msg.value)}`;
                            }
                        }
                        return ctx.failWithMsg(`${errType}${errMsg}`, source).toSet();
                    });
                }
                return ctx.failWithMsg('TypeError: exceptions must derive from BaseException', source).toSet();
            });
        }

        return ctx.failWithMsg('TypeError: exceptions must derive from BaseException', source).toSet();
    }

    export const libCallImpls: { [key in keyof typeof LibCallType]: LCImpl } = {
        import: thImport,
        importQualified,
        super: thSuper,
        genList,
        genDict,
        DEBUG,
        setDefault,
        callKV,
        getAttr,
        explicit,
        exportGlobal,
        raise,
        objectClass,
    };
}

export const libCallMap = new Map([...Object.entries(LCBase.libCallImpls)]);
