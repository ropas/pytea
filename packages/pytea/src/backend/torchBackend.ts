/*
 * torchBackend.ts
 * Copyright (c) Seoul National University.
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo (mersshs@gmail.com)
 *
 * Symbolic Executor & Constraint Generator for PyTea Internal Representation
 */
import { List } from 'immutable';

import { ExpressionNode, ParseNode } from 'pyright-internal/parser/parseNodes';

import {
    LibCallType,
    TEAttr,
    TEBinOp,
    TEBopType,
    TECall,
    TEConst,
    TEConstType,
    TELibCall,
    TEName,
    TEObject,
    TESubscr,
    TETuple,
    TEType,
    TEUnaryOp,
    TEUopType,
    ThExpr,
    ThStmt,
    TSAssign,
    TSBreak,
    TSContinue,
    TSExpr,
    TSForIn,
    TSFunDef,
    TSIf,
    TSLet,
    TSPass,
    TSReturn,
    TSSeq,
    TSType,
} from '../frontend/torchStatements';
import { BuiltinsLCImpl } from '../pylibImplements/builtins';
import { evalLibCall } from '../pylibImplements/evaluator';
import { sanitizeAddrSet, SymOpUtils } from './backUtils';
import * as BackUtils from './backUtils';
import { Context, ContextSet, CtxExpr, CtxStmt } from './context';
import { ShEnv } from './sharpEnvironments';
import {
    ShContFlag,
    ShValue,
    SVAddr,
    SVBool,
    SVError,
    SVFloat,
    SVFunc,
    SVInt,
    SVNone,
    SVNotImpl,
    SVObject,
    SVString,
    SVType,
    svTypeToString,
    SVUndef,
} from './sharpValues';
import { ExpNum, ExpString, NumBopType, SymExp } from './symExpressions';

export namespace TorchBackend {
    export function runEmpty(stmt: ThStmt): ContextSet<ShValue | ShContFlag> {
        const emptyCtx: Context<undefined> = new Context({ retVal: undefined });
        return run(emptyCtx.toSet(), stmt);
    }

    // module torch script must be start with `$module: OBJECT in ...`
    // and ends with `$module.global_var_1 = global_var_1; ... $module.global_var_n = global_var_n`
    // return environment of context would be attributes of $module
    // retVal will be address of $module object itself.
    export function runModule(stmt: TSLet, relPath: string, ctxDefault?: Context<unknown>): ContextSet<SVAddr> {
        const id = stmt.name;
        const exp = TEObject.create(); //stmt.expr;
        const scope = stmt.scope;

        const ctx = ctxDefault ? ctxDefault : new Context({ retVal: undefined, relPath });
        const env = ctx.env;
        const nextSet = evaluate(ctx.toSet(), exp);

        return nextSet.flatMap((ctx) => {
            const value = ctx.retVal;
            const heap = ctx.heap;

            const [addr, newHeap] = heap.allocNew(value, stmt.source);
            const newCtx = ctx.setEnv(env.setId(id, addr)).setHeap(newHeap);
            const newSet = run(newCtx.toSet(), scope);

            return newSet.map((ctx) => {
                // module object. attributes would be new environment
                const module = BackUtils.fetchAddr(addr, ctx.heap) as SVObject;
                let newHeap = ctx.heap;

                let moduleEnv = new ShEnv();
                module.attrs.forEach((v, k) => {
                    moduleEnv = moduleEnv.setId(k, v as SVAddr);
                });

                const notImplAddr = moduleEnv.getId('NotImplemented');
                if (notImplAddr) {
                    newHeap = newHeap.setVal(notImplAddr, SVNotImpl.create('implicit NotImplemented'));
                }

                // run GC
                return ctx.setEnv(moduleEnv).setHeap(newHeap).setRetVal(addr);
            });
        });
    }

    // run builtin library scripts and subtract memory offset
    export function runBuiltin(stmt: ThStmt, relPath: string): ContextSet<SVAddr> {
        return runModule(stmt as TSLet, relPath).map((ctx) => ctx.asDefault());
    }

    export function run<T>(ctxSet: ContextSet<T>, stmt: ThStmt): ContextSet<ShValue | ShContFlag> {
        switch (stmt.stype) {
            case TSType.Pass:
                return _runPass(ctxSet, stmt);
            case TSType.Expr:
                return _runExpr(ctxSet, stmt);
            case TSType.Seq:
                return _runSeq(ctxSet, stmt);
            case TSType.Assign:
                return _runAssign(ctxSet, stmt);
            case TSType.If:
                return _runIf(ctxSet, stmt);
            case TSType.ForIn:
                return _runForIn(ctxSet, stmt);
            case TSType.Return:
                return _runReturn(ctxSet, stmt);
            case TSType.Continue:
                return _runContinue(ctxSet, stmt);
            case TSType.Break:
                return _runBreak(ctxSet, stmt);
            case TSType.Let:
                return _runLet(ctxSet, stmt);
            case TSType.FunDef:
                return _runFunDef(ctxSet, stmt);
        }
    }

    export function evaluate<T>(ctxSet: ContextSet<T>, expr: ThExpr): ContextSet<ShValue> {
        switch (expr.etype) {
            case TEType.Const:
                return sanitizeAddrSet(_evalConst(ctxSet, expr));
            case TEType.Object:
                return sanitizeAddrSet(_evalObject(ctxSet, expr));
            case TEType.Tuple:
                return sanitizeAddrSet(_evalTuple(ctxSet, expr));
            case TEType.Call:
                return sanitizeAddrSet(_evalCall(ctxSet, expr));
            case TEType.LibCall:
                return sanitizeAddrSet(_evalLibCall(ctxSet, expr));
            case TEType.BinOp:
                return sanitizeAddrSet(_evalBinOp(ctxSet, expr));
            case TEType.UnaryOp:
                return sanitizeAddrSet(_evalUnaryOp(ctxSet, expr));
            case TEType.Name:
                return sanitizeAddrSet(_evalName(ctxSet, expr));
            case TEType.Attr:
                return sanitizeAddrSet(_evalAttr(ctxSet, expr));
            case TEType.Subscr:
                return sanitizeAddrSet(_evalSubscr(ctxSet, expr));
        }
    }

    export function evalAll<T>(ctxSet: ContextSet<T>, exprs: ThExpr[]): ContextSet<ShValue[]> {
        let newSet: ContextSet<ShValue[]> = ctxSet.return([]);

        for (const expr of exprs) {
            newSet = newSet.flatMap((ctx) => {
                return evaluate(ctx.toSet(), expr).map((ctxNext) => ctxNext.setRetVal([...ctx.retVal, ctxNext.retVal]));
            });
        }

        return newSet;
    }

    export function classInit<T>(
        ctx: Context<T>,
        classObj: SVObject,
        args: ShValue[],
        source?: ParseNode,
        kwargs?: { [paramName: string]: ShValue }
    ): ContextSet<ShValue> {
        const init = BackUtils.fetchAddr(classObj.getAttr('__call__'), ctx.heap);
        if (!init || init.type !== SVType.Func) {
            return ctx.warnWithMsg(`object is not a class`, source).toSet();
        }

        return functionCall(ctx, init, args, source, kwargs);
    }

    export function libClassInit<T>(
        ctx: Context<T>,
        qualPath: string,
        args: ShValue[],
        source?: ParseNode,
        kwargs?: { [paramName: string]: ShValue }
    ): ContextSet<ShValue> {
        const addr = ctx.imported.getId(qualPath);
        if (!addr) {
            return ctx.warnWithMsg(`${qualPath} is not imported`, source).toSet();
        }

        const classVal = BackUtils.fetchAddr(addr, ctx.heap);
        if (classVal?.type !== SVType.Object) {
            return ctx.warnWithMsg(`${qualPath} is not an object`, source).toSet();
        }

        return classInit(ctx, classVal, args, source, kwargs);
    }

    // process Call(f, args)
    export function functionCall<T>(
        ctx: Context<T>,
        f: SVFunc | undefined,
        args: ShValue[],
        source?: ParseNode,
        kwargs?: { [paramName: string]: ShValue }
    ): ContextSet<ShValue> {
        const { env, heap } = ctx;
        if (f === undefined) {
            return ctx.warnWithMsg('function not found', source).toSet();
        }

        if (f.funcEnv === undefined) {
            return ctx.failWithMsg(`env of a function ${f} is not defined`, source).toSet();
        }

        let newHeap = heap;

        // process sigma_f[fname->l][params->l']
        // process H[l->f][l'->args]
        const argAddrs: SVAddr[] = [];

        const [fAddr, nextHeap1] = newHeap.allocNew(f, source);
        let fEnv = f.funcEnv.setId(f.name, fAddr);
        newHeap = nextHeap1;
        argAddrs.push(fAddr);

        const paramAddrs: Map<string, SVAddr> = new Map();
        f.params.forEach((param) => {
            const [addr, h] = newHeap.malloc();
            newHeap = h;
            fEnv = fEnv.setId(param, addr);
            paramAddrs.set(param, addr);
        });

        // assigning defaults
        f.defaults.forEach((v, k) => {
            if (paramAddrs.has(k)) {
                newHeap = newHeap.setVal(paramAddrs.get(k)!, v);
            }
        });

        // assigning pos args and varargs
        // TODO: set varargTuple as tuple class
        const posargLen =
            f.params.count() - (f.kwargsParam ? 1 : 0) - (f.varargsParam ? 1 : 0) - (f.keyOnlyNum ? f.keyOnlyNum : 0);

        let varargTuple: SVObject | undefined;
        if (f.varargsParam) {
            const tuple = SVObject.createWithAddr(paramAddrs.get(f.varargsParam!)!);
            varargTuple = tuple;
        }

        let varargLen = 0;
        args.forEach((v, i) => {
            if (i < posargLen) {
                const paramName = f.params.get(i)!;
                newHeap = newHeap.setVal(paramAddrs.get(paramName)!, v);
            } else if (varargTuple) {
                varargTuple = varargTuple.setIndice(i - posargLen, v);
                varargLen++;
            }
        });

        if (varargTuple) {
            varargTuple = varargTuple.setAttr('$length', SVInt.create(varargLen, source));
            newHeap = newHeap.setVal(paramAddrs.get(f.varargsParam!)!, varargTuple);
        }

        // assigning kwargs
        // TODO: set kwargDict as dict class
        let kwargDict: SVObject | undefined;
        if (f.kwargsParam) {
            const dictAddr = paramAddrs.get(f.kwargsParam!)!;
            kwargDict = SVObject.createWithAddr(dictAddr);
        }

        if (kwargs) {
            Object.entries(kwargs).forEach(([k, v]) => {
                if (paramAddrs.has(k)) {
                    newHeap = newHeap.setVal(paramAddrs.get(k)!, v);
                } else if (kwargDict) {
                    kwargDict = kwargDict.setKeyVal(k, v);
                }
            });
        }
        if (kwargDict) {
            newHeap = newHeap.setVal(paramAddrs.get(f.kwargsParam!)!, kwargDict);
        }

        const newCtx = ctx.setEnv(fEnv).setHeap(newHeap);

        // run the function body
        return run(newCtx.pushCallStack([f, source]).toSet(), f.funcBody).map((ctx) => {
            const value = ctx.retVal;

            // free temp addresses for argument if body has no closure
            let newCtx = ctx.popCallStack().setEnv(env);
            if (!f.hasClosure) {
                let heap = newCtx.heap;
                argAddrs.forEach((addr) => {
                    heap = heap.free(addr);
                });
                newCtx = newCtx.setHeap(heap);
            }

            if (value === ShContFlag.Brk || value === ShContFlag.Cnt || value === ShContFlag.Run) {
                // this is right. a function without any return method actually returns None.
                return newCtx.setRetVal(SVNone.create());
            } else {
                return newCtx.setRetVal(value);
            }
        });
    }

    export function functionCallSet<T>(
        ctxSet: ContextSet<T>,
        f: SVFunc,
        args: ShValue[],
        source?: ParseNode,
        kwargs?: { [paramName: string]: ShValue }
    ): ContextSet<ShValue> {
        return ctxSet.flatMap((ctx) => functionCall(ctx, f, args, source, kwargs));
    }

    function _runPass<T>(ctxSet: ContextSet<T>, stmt: TSPass): ContextSet<ShValue | ShContFlag> {
        return ctxSet.return(ShContFlag.Run);
    }

    function _runExpr<T>(ctxSet: ContextSet<T>, stmt: TSExpr): ContextSet<ShValue | ShContFlag> {
        const exp = stmt.expr;
        const newSet = evaluate(ctxSet, exp);

        return newSet.return(ShContFlag.Run);
    }

    function _runSeq<T>(ctxSet: ContextSet<T>, stmt: TSSeq): ContextSet<ShValue | ShContFlag> {
        const stmt1 = stmt.left;
        const stmt2 = stmt.right;

        const newSet = run(ctxSet, stmt1);
        return newSet.flatMap((ctx) => {
            const nextSet = ctx.toSet();

            if (ctx.retVal === ShContFlag.Run) {
                return run(nextSet, stmt2);
            } else {
                return nextSet;
            }
        });
    }

    // TODO: change all getAttr to this.
    export function getAttrDeep<T>(
        ctx: Context<T>,
        object: ShValue,
        name: string,
        source?: ParseNode
    ): ContextSet<ShValue> {
        const objVal = BackUtils.fetchAddr(object, ctx.heap);

        if (!objVal) {
            return ctx.warnWithMsg(`getAttrDeep(${name}): invalid address of object`, source).toSet();
        }

        // propagate warning
        if (objVal.type === SVType.Error) {
            return ctx.setRetVal(objVal).toSet();
        }

        // if name == '__dict__', return dictionary of attrs
        if (name === '__dict__') {
            let [dictObj, dictAddr, newHeap] = SVObject.create(ctx.heap, source);
            for (let [attrName, attrVal] of objVal.attrs.entries()) {
                dictObj = dictObj.setKeyVal(attrName, attrVal);
            }
            dictObj = dictObj.setAttr('$length', SVInt.create(dictObj.keyValues.size, source));
            let dictType = BackUtils.fetchAddr(ctx.env.getId('dict'), newHeap);
            if (dictType?.type === SVType.Object) {
                // class _Primitives defines self.mro = (self, object)
                dictType = BackUtils.fetchAddr(dictType.getAttr('__mro__'), newHeap);
            }
            if (dictType?.type === SVType.Object) {
                dictObj = dictObj.setAttr('__mro__', dictType);
            }
            return ctx.setHeap(newHeap.setVal(dictAddr, dictObj)).toSetWith(dictObj);
        }

        const attr = objVal.attrs.get(name);

        if (attr === undefined) {
            const getAttr = BackUtils.fetchAddr(objVal.attrs.get('__getattr__'), ctx.heap);
            if (getAttr && getAttr.type === SVType.Func) {
                return functionCall(ctx, getAttr, [SVString.create(name, source)], source);
            } else {
                // first, make a list of superclasses
                const mro = BackUtils.trackMro(objVal, ctx.heap, ctx.env);
                const classes = [];

                // iterate superclasses and find matching attr.
                for (const superAddr of mro) {
                    if (superAddr === undefined) continue;
                    const superClass = BackUtils.fetchAddr(SVAddr.create(superAddr), ctx.heap);

                    if (!superClass) continue;
                    classes.push(superClass);

                    const attr = superClass.attrs.get(name);
                    if (attr) {
                        const mayMethod = BackUtils.fetchAddr(attr, ctx.heap);

                        // if found function is unbounded method, bind it.
                        if (mayMethod?.type === SVType.Func) {
                            let bounded: SVFunc | undefined;
                            if (objVal.type === SVType.Object) {
                                bounded = mayMethod.bound(objVal.addr);
                            }
                            if (bounded) return ctx.setRetVal(bounded).toSet();
                        }
                        return ctx.setRetVal(attr).toSet();
                    }
                }

                // if not found, track __getattr__
                for (const superClass of classes) {
                    let getAttr = BackUtils.fetchAddr(superClass.attrs.get('__getattr__'), ctx.heap);
                    if (getAttr && getAttr.type === SVType.Func) {
                        if (objVal.type === SVType.Object) {
                            getAttr = getAttr.bound(objVal.addr);
                        }
                        return functionCall(ctx, getAttr, [SVString.create(name, source)], source);
                    }
                }
            }

            return ctx.warnWithMsg(`getAttrDeep(${name}): attribute not found`, source).toSet();
        } else {
            return ctx.setRetVal(attr).toSet();
        }
    }

    // TODO: change all getIndices to this.
    export function getIndiceDeep<T>(
        ctx: Context<T>,
        object: ShValue,
        index: number | ExpNum,
        source?: ParseNode
    ): ContextSet<ShValue> {
        const objVal = BackUtils.fetchAddr(object, ctx.heap);

        if (objVal?.type !== SVType.Object) {
            return ctx.warnWithMsg(`getIndiceDeep ${index}: value is not an object`, source).toSet();
        }

        const item = typeof index === 'number' ? objVal.indices.get(index) : undefined;

        if (item === undefined) {
            return ctx.getAttrDeep(object, '__getitem__', source).flatMap((ctx) => {
                const getItem = ctx.retVal;
                if (getItem?.type === SVType.Func) {
                    return functionCall(ctx, getItem, [SVInt.create(index, source)], source);
                } else {
                    return ctx.warnWithMsg(`getIndiceDeep ${index}: index ${index} not exist.`, source).toSet();
                }
            });
        }

        return ctx.setRetVal(item).toSet();
    }

    // TODO: change all getKeyVal to this.
    export function getKeyValDeep<T>(
        ctx: Context<T>,
        object: ShValue,
        key: string,
        source?: ParseNode
    ): ContextSet<ShValue> {
        const objVal = BackUtils.fetchAddr(object, ctx.heap);

        if (objVal?.type !== SVType.Object) {
            return ctx.warnWithMsg(`getKeyValDeep ${key}: value is not an object`, source).toSet();
        }

        const item = objVal.keyValues.get(key);

        if (item === undefined) {
            return ctx.getAttrDeep(object, '__getitem__', source).flatMap((ctx) => {
                const getItem = ctx.retVal;
                if (getItem?.type === SVType.Func) {
                    return functionCall(ctx, getItem, [SVString.create(key, source)], source);
                } else {
                    return ctx.warnWithMsg(`getKeyValDeep ${key}: key ${key} not exist.`, source).toSet();
                }
            });
        }

        return ctx.setRetVal(item).toSet();
    }

    function _runAssign<T>(ctxSet: ContextSet<T>, stmt: TSAssign): ContextSet<ShValue | ShContFlag> {
        const lexpr = stmt.left;
        const rexpr = stmt.right;

        if (lexpr.etype === TEType.Name) {
            const id = lexpr.ident;

            return ctxSet.flatMap((ctx) => {
                const nextSet = ctx.toSet();
                const addr = ctx.env.getId(id);
                if (addr === undefined) {
                    return ctx.failWithMsg(`address not found at id ${id}`, stmt.source).toSet() as CtxStmt;
                }

                // move semantics of $TMP$ value.
                if (rexpr.etype === TEType.Name && rexpr.ident.endsWith('$TMP$')) {
                    let newCtx = ctx;
                    let heap = ctx.heap;
                    const tmpAddr = ctx.env.getId(rexpr.ident);
                    if (tmpAddr) {
                        const tmpVal = BackUtils.fetchAddr(tmpAddr, heap);
                        heap = heap.free(tmpAddr);
                        if (tmpVal) {
                            heap = heap.setVal(addr, tmpVal);
                        }
                        newCtx = ctx.setHeap(heap).setEnv(ctx.env.removeId(rexpr.ident));
                    }
                    return newCtx.setRetVal(ShContFlag.Run).toSet();
                }
                const nextSet2 = evaluate(nextSet, rexpr);
                const nextSet3 = nextSet2.map((ctx) => {
                    return ctx.setHeap(ctx.heap.setVal(addr, ctx.retVal));
                });
                return nextSet3.return(ShContFlag.Run);
            });
        } else if (lexpr.etype === TEType.Attr) {
            const exp1 = lexpr.left;
            const id = lexpr.right;
            const exp2 = rexpr;

            const nextSet = evaluate(ctxSet, exp1);
            return nextSet.flatMap((ctx) => {
                const addr = ctx.retVal;
                if (addr.type !== SVType.Addr) {
                    if (addr.type === SVType.Error) return ctx.toSetWith(addr);
                    return ctx.failWithMsg(`${ctx.retVal} is not an address`, stmt.source).toSet() as CtxStmt;
                }
                const nextSet2 = ctx.toSet();
                const nextSet3 = evaluate(nextSet2, exp2);
                const nextSet4 = nextSet3.map((ctx) => {
                    const value = ctx.retVal;
                    const obj = BackUtils.fetchAddr(addr, ctx.heap);
                    if (obj === undefined || obj.type !== SVType.Object) {
                        if (obj?.type === SVType.Error) return ctx.setRetVal(obj);
                        return ctx.failWithMsg(`object not found at address ${addr}`, stmt.source);
                    }
                    const newObj = obj.setAttr(id, value);
                    const newHeap = ctx.heap.setVal(addr, newObj);
                    return ctx.setHeap(newHeap);
                });
                return nextSet4.return(ShContFlag.Run);
            });
        } else if (lexpr.etype === TEType.Subscr) {
            const exp1 = lexpr.left;
            const exp2 = lexpr.right;
            const exp3 = rexpr;

            const nextSet = evaluate(ctxSet, exp1);
            return nextSet.flatMap((ctx) => {
                const obj = BackUtils.fetchAddr(ctx.retVal, ctx.heap);
                if (obj?.type !== SVType.Object) {
                    if (obj?.type === SVType.Error) return ctx.toSetWith(obj);
                    return ctx.failWithMsg(`${ctx.retVal} is not an address`, stmt.source).toSet() as CtxStmt;
                }

                return ctx.getAttrDeep(obj, '__setitem__').flatMap((ctx) => {
                    const func = BackUtils.fetchAddr(ctx.retVal, ctx.heap);
                    if (func === undefined) {
                        return evaluate(ctx.toSet(), exp2).flatMap((ctx) => {
                            const n = BackUtils.fetchAddr(ctx.retVal, ctx.heap);
                            if (n?.type !== SVType.Int) {
                                return ctx
                                    .warnWithMsg(`__setitem__ index ${n} is not a number`, stmt.source)
                                    .toSet() as CtxStmt;
                            }
                            const nextSet4 = ctx.toSet();
                            const nextSet5 = evaluate(nextSet4, exp3);
                            const nextSet6 = nextSet5.map((ctx) => {
                                const value = ctx.retVal;
                                if (typeof n.value === 'number') {
                                    const newObj = obj.setIndice(n.value, value);
                                    const newHeap = ctx.heap.setVal(newObj.addr, newObj);
                                    return ctx.setHeap(newHeap);
                                } else {
                                    // TODO: when index n is a symbolic number.
                                    return ctx.addLog('');
                                }
                            });
                            return nextSet6.return(ShContFlag.Run);
                        });
                    } else {
                        if (func.type !== SVType.Func) {
                            return ctx.warnWithMsg(`__setitem__ ${func} is not a function`, stmt.source).toSet();
                        }
                        return evaluate(ctx.toSet(), exp2).flatMap((ctx) => {
                            const indexValue = ctx.retVal;
                            return evaluate(ctx.toSet(), exp3)
                                .flatMap((ctx) => {
                                    const setValue = ctx.retVal;
                                    return functionCall(ctx, func, [indexValue, setValue], stmt.source);
                                })
                                .return(ShContFlag.Run);
                        });
                    }
                });
            });
        } else {
            return ctxSet.fail('cannot reach here');
        }
    }

    function _runIf<T>(ctxSet: ContextSet<T>, stmt: TSIf): ContextSet<ShValue | ShContFlag> {
        const exp = stmt.cond;
        const stmt_t = stmt.thenStmt;
        const stmt_f = stmt.elseStmt;

        return evaluate(ctxSet, exp).flatMap((ctx) => {
            const value = ctx.retVal;
            const isTruthy = BackUtils.isTruthy(ctx, value, exp.source);

            if (isTruthy === true) {
                return run(ctx.toSet(), stmt_t);
            } else if (isTruthy === false) {
                return run(ctx.toSet(), stmt_f);
            } else {
                const [truePath, falsePath] = ctx.toSet().ifThenElse(isTruthy);
                return run(truePath, stmt_t).join(run(falsePath, stmt_f));
            }
        });
    }

    function _runForIn<T>(ctxSet: ContextSet<T>, stmt: TSForIn): ContextSet<ShValue | ShContFlag> {
        const ident = stmt.ident;
        const exp = stmt.loopVal;

        return ctxSet.flatMap((ctx) => {
            const envOrigin = ctx.env;
            const nextSet = evaluate(ctx.toSet(), exp);

            return nextSet.flatMap((ctx) => {
                const objAddr = ctx.retVal;
                const obj = BackUtils.fetchAddr(objAddr, ctx.heap);

                // TODO: for string
                if (obj?.type !== SVType.Object) {
                    return ctx.warnWithMsg(`loop value is not an object ${obj}`, exp.source).toSet();
                }

                return BuiltinsLCImpl.len(ctx.setRetVal({ params: [objAddr] }), exp.source).flatMap((ctx) => {
                    // TODO: in case of symbolic number len
                    const len = BackUtils.fetchAddr(ctx.retVal, ctx.heap);
                    let length: number | ExpNum;

                    if (len === undefined || len.type === SVType.Error || len.type === SVType.NotImpl) {
                        const ctx2 = ctx
                            .genIntGte('for$len', 0, stmt.loopVal.source)
                            .addLogValue(
                                SVError.create(
                                    `WARNING: loop length is not an int type: got ${
                                        len ? svTypeToString(len.type) : undefined
                                    }. use symbolic length`,
                                    stmt.loopVal.source
                                )
                            );
                        length = ctx2.retVal;
                        ctx = ctx2.setRetVal(ctx.retVal);
                    } else if (len.type !== SVType.Int) {
                        return ctx
                            .failWithMsg(`length is not an int type: got ${svTypeToString(len.type)}`, exp.source)
                            .toSet() as CtxStmt;
                    } else {
                        length = len.value;
                        const lenRng = ctx.ctrSet.getCachedRange(length)?.toIntRange();
                        if (lenRng && lenRng.isConst() && lenRng.start >= 0) {
                            length = lenRng.start;
                        } else {
                            length = len.value;
                        }
                    }

                    const [identAddr, newHeap] = ctx.heap.malloc();
                    const newEnv = ctx.env.setId(ident, identAddr);

                    const newCtx = ctx.setEnv(newEnv).setHeap(newHeap);
                    const resultSet =
                        typeof length === 'number'
                            ? _runConstFor(newCtx, identAddr, obj, length, stmt)
                            : _runSymbolicFor(newCtx, identAddr, obj, length, stmt);

                    return resultSet.map((ctx) => ctx.setEnv(envOrigin));
                });
            });
        });
    }

    function _runConstFor(
        ctx: Context<ShValue | ShContFlag>,
        identAddr: SVAddr,
        obj: ShValue,
        loopCnt: number,
        stmt: TSForIn
    ): ContextSet<ShValue | ShContFlag> {
        let brkRtnSet: ContextSet<ShValue | ShContFlag> = Context.getEmptySet();
        let nextSetIter: ContextSet<ShValue | ShContFlag> = ctx.toSet();

        for (let i = 0; i < loopCnt; i++) {
            nextSetIter = nextSetIter.flatMap((ctx) => {
                return ctx
                    .getIndiceDeep(obj, i, stmt.loopVal.source)
                    .map((ctx) => ctx.setHeap(ctx.heap.setVal(identAddr, ctx.retVal)));
            });

            nextSetIter = run(nextSetIter, stmt.loopBody);
            const runCnt = nextSetIter.filter((ctx) => ctx.retVal === ShContFlag.Run || ctx.retVal === ShContFlag.Cnt);
            const brkRtn = nextSetIter.filter((ctx) => ctx.retVal !== ShContFlag.Run && ctx.retVal !== ShContFlag.Cnt);

            brkRtnSet = brkRtnSet.join(brkRtn);
            nextSetIter = runCnt;
        }

        const resultSet = nextSetIter.join(brkRtnSet);
        return resultSet.map((ctx) =>
            ctx.retVal === ShContFlag.Brk || ctx.retVal === ShContFlag.Cnt || ctx.retVal === ShContFlag.Run
                ? ctx.setRetVal(ShContFlag.Run)
                : ctx
        );
    }

    function _runSymbolicFor(
        ctx: Context<ShValue | ShContFlag>,
        identAddr: SVAddr,
        obj: ShValue,
        loopCnt: ExpNum,
        stmt: TSForIn
    ): ContextSet<ShValue | ShContFlag> {
        // TODO: before implement SMT on a symbolic value, we require that loopCnt is bigger than 0
        return ctx
            .require(ctx.genLte(0, loopCnt), 'length of iterator is less than 0', stmt.loopVal.source)
            .flatMap((ctx) => {
                const identCtx = ctx.genIntGte('for$ident', 0, stmt.loopVal.source);
                const ident = identCtx.retVal;
                let nextSetIter: ContextSet<ShValue | ShContFlag> = identCtx
                    .guarantee(identCtx.genLte(ident, ExpNum.bop(NumBopType.Sub, loopCnt, 1, stmt.loopVal.source)))
                    .toSetWith(ShContFlag.Run);

                nextSetIter = nextSetIter.flatMap((ctx) => {
                    return ctx
                        .getIndiceDeep(obj, ident, stmt.loopVal.source)
                        .map((ctx) => ctx.setHeap(ctx.heap.setVal(identAddr, ctx.retVal)));
                });
                nextSetIter = run(nextSetIter, stmt.loopBody);
                return nextSetIter.map((ctx) =>
                    ctx.retVal === ShContFlag.Brk || ctx.retVal === ShContFlag.Cnt || ctx.retVal === ShContFlag.Run
                        ? ctx.setRetVal(ShContFlag.Run)
                        : ctx
                );
            })
            .return(ShContFlag.Run);
    }
    function _runReturn<T>(ctxSet: ContextSet<T>, stmt: TSReturn): ContextSet<ShValue> {
        return evaluate(ctxSet, stmt.expr);
    }

    function _runContinue<T>(ctxSet: ContextSet<T>, stmt: TSContinue): ContextSet<ShValue | ShContFlag> {
        return ctxSet.return(ShContFlag.Cnt);
    }

    function _runBreak<T>(ctxSet: ContextSet<T>, stmt: TSBreak): ContextSet<ShValue | ShContFlag> {
        return ctxSet.return(ShContFlag.Brk);
    }

    function _runLet<T>(ctxSet: ContextSet<T>, stmt: TSLet): ContextSet<ShValue | ShContFlag> {
        const id = stmt.name;
        const exp = stmt.expr;
        const stmt_scope = stmt.scope;

        if (exp === undefined) {
            return ctxSet.flatMap((ctx) => {
                const env = ctx.env;
                const heap = ctx.heap;

                const [addr, newHeap] = heap.allocNew(SVUndef.create(stmt.source), stmt.source);
                const newCtx = ctx.setEnv(env.setId(id, addr)).setHeap(newHeap);
                const newSet = run(newCtx.toSet(), stmt_scope);
                return newSet.map((ctx) => ctx.setEnv(env));
            });
        } else {
            return ctxSet.flatMap((ctx) => {
                const env = ctx.env;
                const nextSet = evaluate(ctx.toSet(), exp);
                return nextSet.flatMap((ctx) => {
                    const value = ctx.retVal;
                    const heap = ctx.heap;

                    const [addr, newHeap] = heap.allocNew(value, stmt.source);
                    const newCtx = ctx.setEnv(env.setId(id, addr)).setHeap(newHeap);
                    const newSet = run(newCtx.toSet(), stmt_scope);
                    return newSet.map((ctx) => ctx.setEnv(env));
                });
            });
        }
    }

    function _runFunDef<T>(ctxSet: ContextSet<T>, stmt: TSFunDef): ContextSet<ShValue | ShContFlag> {
        const funName = stmt.name;
        const params = stmt.params;
        const stmt_body = stmt.body;
        const stmt_scope = stmt.scope;

        return ctxSet.flatMap((ctx) => {
            const env = ctx.env;
            const heap = ctx.heap;
            const func = SVFunc.create(funName, List(params), stmt_body, env);

            const [addr, newHeap] = heap.allocNew(func, stmt.source);
            const newCtx = ctx.setEnv(env.setId(funName, addr)).setHeap(newHeap);
            const newSet = run(newCtx.toSet(), stmt_scope);
            return newSet.map((ctx) => ctx.setEnv(env));
        });
    }

    function _evalConst<T>(ctxSet: ContextSet<T>, expr: TEConst): ContextSet<ShValue> {
        return ctxSet.map((ctx) => {
            let retVal: ShValue;
            switch (expr.constType) {
                case TEConstType.Int:
                    retVal = SVInt.create(expr.value as number, expr.source);
                    break;
                case TEConstType.Float:
                    retVal = SVFloat.create(expr.value as number, expr.source);
                    break;
                case TEConstType.String:
                    retVal = SVString.create(expr.value as string, expr.source);
                    break;
                case TEConstType.Bool:
                    retVal = SVBool.create(expr.value as boolean, expr.source);
                    break;
                case TEConstType.None:
                    retVal = SVNone.create(expr.source);
                    break;
            }

            return ctx.setRetVal(retVal);
        });
    }

    function _evalObject<T>(ctxSet: ContextSet<T>, expr: TEObject): ContextSet<ShValue> {
        return ctxSet.map((ctx) => {
            const heap = ctx.heap;
            const [, addr, newHeap] = SVObject.create(heap, expr.source);
            return ctx.setHeap(newHeap).setRetVal(addr);
        });
    }

    function _evalTuple<T>(ctxSet: ContextSet<T>, expr: TETuple): ContextSet<ShValue> {
        return ctxSet.flatMap((ctx) => {
            const newObj = SVObject.create(ctx.heap, expr.source);
            const [tempTuple, tupleAddr, newHeap] = newObj;
            const tuple = tempTuple.setAttr('$length', SVInt.create(expr.values.length, expr.source));

            let evalSet: ContextSet<ShValue[]> = ctx.setHeap(newHeap).toSetWith([]);
            expr.values.forEach((expVal) => {
                evalSet = evalSet.flatMap((ctx) => {
                    const valueList = ctx.retVal;
                    const valueSet = evaluate(evalSet, expVal);
                    return valueSet.map((ctx) => ctx.setRetVal([...valueList, ctx.retVal]));
                });
            });

            return evalSet.map((ctx) => {
                let tupleType = BackUtils.fetchAddr(ctx.env.getId('tuple'), ctx.heap);
                if (tupleType?.type === SVType.Object) {
                    // class _Primitives defines self.mro = (self, object)
                    tupleType = BackUtils.fetchAddr(tupleType.getAttr('__mro__'), ctx.heap);
                }

                let ctxTuple = tuple;
                ctx.retVal.forEach((v, i) => {
                    ctxTuple = ctxTuple.setIndice(i, v);
                });

                // found tuple.
                if (tupleType?.type === SVType.Object) {
                    ctxTuple = ctxTuple.setAttr('__mro__', tupleType);
                }

                return ctx.setHeap(ctx.heap.setVal(tupleAddr, ctxTuple)).setRetVal(tupleAddr);
            });
        });
    }

    function _evalCall<T>(ctxSet: ContextSet<T>, expr: TECall): ContextSet<ShValue> {
        // written by Woosung Song
        return evaluate(ctxSet, expr.func).flatMap((ctx) => {
            const func = ctx.retVal;
            let argCtx: ContextSet<ShValue[]> = ctx.toSetWith([]);
            expr.params.forEach((param) => {
                argCtx = argCtx.flatMap((ctx) => {
                    const argList = ctx.retVal;
                    return evaluate(ctx.toSet(), param).map((ctx) => ctx.setRetVal([...argList, ctx.retVal]));
                });
            });

            return argCtx.flatMap((ctx) => {
                // propagate error
                if (func.type === SVType.Error) {
                    return ctx.toSetWith(func) as CtxExpr;
                }

                const funcVal = BackUtils.fetchAddr(func, ctx.heap);

                if (!funcVal) {
                    return ctx.warnWithMsg('call to invalid address', expr.source).toSet();
                }

                if (funcVal.type === SVType.Func) {
                    // debuggin original
                    return functionCall(ctx.setRetVal(funcVal), funcVal, ctx.retVal, expr.source);
                    //
                    //let newSet = functionCall(ctx.setRetVal(funcVal), funcVal, ctx.retVal, expr.source);
                    //newSet = newSet.map((ctx) => ctx.addLog(`%%%%evalCall :: retVal: ${ctx.retVal} `, expr.source));
                    //return newSet;
                } else if (funcVal.type === SVType.Object) {
                    const call = funcVal.getAttr('__call__');
                    if (call) {
                        const callVal = BackUtils.fetchAddr(call, ctx.heap);
                        if (callVal?.type === SVType.Func) {
                            return functionCall(ctx.setRetVal(callVal), callVal, ctx.retVal, expr.source);
                        }
                    }
                }
                return ctx.warnWithMsg('object is not callable', expr.source).toSet() as CtxExpr;
            });
        });
    }

    function _evalLibCall<T>(ctxSet: ContextSet<T>, expr: TELibCall): ContextSet<ShValue> {
        let libCallName: string = expr.type;
        if (libCallName === LibCallType.explicit) {
            const explicitName = expr.params[0];
            if (explicitName && explicitName[1].etype === TEType.Const) {
                libCallName = explicitName[1].value as string;
            }
        }
        return evalLibCall(
            ctxSet.map((ctx) => ctx.pushCallStack([libCallName, expr.source])),
            expr
        ).map((ctx) => ctx.popCallStack());
    }

    function _evalBinOp<T>(ctxSet: ContextSet<T>, expr: TEBinOp): ContextSet<ShValue> {
        return evaluate(ctxSet, expr.left).flatMap((ctx) => {
            const leftAddr = ctx.retVal;

            // short-circuit evaluation.
            if (expr.bopType === TEBopType.And || expr.bopType === TEBopType.Or) {
                const isAnd = expr.bopType === TEBopType.And;
                const truthy = BackUtils.isTruthy(ctx, leftAddr, expr.source as ExpressionNode);

                if (truthy === true) {
                    return isAnd ? evaluate(ctx.toSet(), expr.right) : ctx.toSet();
                } else if (truthy === false) {
                    return isAnd ? ctx.toSet() : evaluate(ctx.toSet(), expr.right);
                } else {
                    const [truePath, falsePath] = ctx.toSet().ifThenElse(truthy);
                    if (isAnd) {
                        return evaluate(truePath, expr.right).join(falsePath);
                    } else {
                        return evaluate(falsePath, expr.right).join(truePath);
                    }
                }
            }

            return evaluate(ctx.toSet(), expr.right).flatMap((ctx) => {
                const rightAddr = ctx.retVal;
                let leftVal = BackUtils.fetchAddr(leftAddr, ctx.heap);
                let rightVal = BackUtils.fetchAddr(ctx.retVal, ctx.heap);
                let ctrSet = ctx.ctrSet;

                if (!leftVal || !rightVal) {
                    return ctx.warnWithMsg('errornous binary operation', expr.source).toSet();
                }

                // string bop
                let result: ShValue | undefined;
                let hasString = false;
                if (leftVal.type === SVType.String && rightVal.type === SVType.String) {
                    result = SymOpUtils.binOpStr(
                        ctx.ctrSet,
                        leftVal,
                        rightVal,
                        expr.bopType,
                        expr.source as ExpressionNode
                    );
                    hasString = true;
                } else if (leftVal.type === SVType.String && SymOpUtils.isNumeric(rightVal)) {
                    result = SymOpUtils.binOpStrNum(
                        ctrSet,
                        leftVal,
                        rightVal,
                        expr.bopType,
                        expr.source as ExpressionNode
                    );
                    hasString = true;
                } else if (rightVal.type === SVType.String && SymOpUtils.isNumeric(leftVal)) {
                    result = SymOpUtils.binOpStrNum(
                        ctrSet,
                        rightVal,
                        leftVal,
                        expr.bopType,
                        expr.source as ExpressionNode
                    );
                    hasString = true;
                }

                if (hasString) {
                    if (result === undefined) {
                        return ctx.failWithMsg(`invalid operation ${expr.bopType} in string`, expr.source).toSet();
                    }
                    return ctx.toSetWith(result);
                }

                // numeric bop
                if (SymOpUtils.isNumeric(leftVal) && SymOpUtils.isNumeric(rightVal)) {
                    if (leftVal.type === SVType.Bool) {
                        const lvSet = ctrSet.castBoolToInt(leftVal.value, expr.source as ExpressionNode);
                        leftVal = SVInt.create(lvSet[0]);
                        ctrSet = lvSet[1];
                    }

                    if (rightVal.type === SVType.Bool) {
                        const rvSet = ctrSet.castBoolToInt(rightVal.value, expr.source as ExpressionNode);
                        rightVal = SVInt.create(rvSet[0]);
                        ctrSet = rvSet[1];
                    }

                    const retVal = SymOpUtils.binOpNum(leftVal, rightVal, expr.bopType, expr.source as ExpressionNode);
                    return ctx.setCtrSet(ctrSet).setRetVal(retVal).toSet();
                }

                // object addr check bop or check nonetype
                // TODO: == None / != None
                if (expr.bopType === TEBopType.Is) {
                    if (leftAddr.type === SVType.Addr && rightAddr.type === SVType.Addr) {
                        return ctx.toSetWith(SVBool.create(leftAddr.addr === rightAddr.addr, expr.source));
                    } else if (leftVal.type === SVType.None) {
                        return ctx.toSetWith(SVBool.create(rightVal.type === SVType.None, expr.source));
                    } else if (rightVal.type === SVType.None) {
                        return ctx.toSetWith(SVBool.create(false, expr.source));
                    }
                    return ctx.toSetWith(SVBool.create(false));
                } else if (expr.bopType === TEBopType.IsNot) {
                    if (leftAddr.type === SVType.Addr && rightAddr.type === SVType.Addr) {
                        return ctx.toSetWith(SVBool.create(leftAddr.addr !== rightAddr.addr, expr.source));
                    } else if (leftVal.type === SVType.None) {
                        return ctx.toSetWith(SVBool.create(rightVal.type !== SVType.None, expr.source));
                    } else if (rightVal.type === SVType.None) {
                        return ctx.toSetWith(SVBool.create(true, expr.source));
                    }
                    return ctx.toSetWith(SVBool.create(true));
                }

                // object dunder operator
                const defaultRetVal: ShValue = SVNotImpl.create(
                    `invalid bop ${TEBinOp.toStringBop(expr.bopType)}`,
                    expr.source
                );
                const [lop, rop] = SymOpUtils.operatorMap[expr.bopType];

                if (leftVal.type === SVType.Object) {
                    const lv = leftVal;
                    return ctx
                        .toSet()
                        .flatMap((ctx) => getAttrDeep(ctx, lv, lop, expr.source))
                        .flatMap((ctx) => {
                            const lopFun = BackUtils.fetchAddr(ctx.retVal, ctx.heap);
                            if (lopFun?.type === SVType.Func) {
                                return functionCall(ctx, lopFun, [rightAddr], expr.source).flatMap((ctx) => {
                                    const calcVal = ctx.retVal;
                                    if (calcVal.type === SVType.NotImpl && rightVal?.type === SVType.Object) {
                                        return getAttrDeep(ctx, rightVal, rop, expr.source).flatMap((ctx) => {
                                            const ropFun = BackUtils.fetchAddr(ctx.retVal, ctx.heap);
                                            if (ropFun?.type === SVType.Func) {
                                                return functionCall(ctx, ropFun, [leftAddr], expr.source);
                                            }
                                            return ctx.toSetWith(defaultRetVal);
                                        });
                                    }
                                    return ctx.toSet();
                                });
                            }

                            return ctx.toSetWith(defaultRetVal);
                        });
                }

                if (rightVal.type === SVType.Object) {
                    return getAttrDeep(ctx, rightVal, rop, expr.source).flatMap((ctx) => {
                        const ropFun = BackUtils.fetchAddr(ctx.retVal, ctx.heap);
                        if (ropFun?.type === SVType.Func) {
                            return functionCall(ctx, ropFun, [leftAddr], expr.source);
                        }
                        return ctx.toSetWith(defaultRetVal);
                    });
                }

                return ctx.toSetWith(defaultRetVal);
            });
        });
    }

    function _evalUnaryOp<T>(ctxSet: ContextSet<T>, expr: TEUnaryOp): ContextSet<ShValue> {
        // written by Woosung Song
        return evaluate(ctxSet, expr.base).flatMap((ctx) => {
            const value = ctx.retVal;

            if (SymOpUtils.isNumeric(value)) {
                let newValue = value; // type cast before run unary op.
                let newCtx = ctx;
                if (!SymOpUtils.isConstant(value)) {
                    switch (expr.uopType) {
                        case TEUopType.Neg:
                            if (value.type === SVType.Bool) {
                                const [castVal, ctrNew] = ctx.ctrSet.castBoolToInt(value.value);
                                newValue = SVInt.create(castVal, value.source);
                                newCtx = ctx.setCtrSet(ctrNew);
                            }
                            break;
                        case TEUopType.Not:
                            if (value.type === SVType.Int || value.type === SVType.Float) {
                                const castVal = ctx.ctrSet.castNumToBool(value.value);
                                if (typeof castVal === 'string') {
                                    return ctx.toSetWith(SVError.create(castVal, expr.source));
                                }
                                newValue = SVBool.create(castVal[0], value.source);
                                newCtx = ctx.setCtrSet(castVal[1]);
                            }
                            break;
                    }
                }
                return newCtx.toSetWith(SymOpUtils.unaryOp(newValue, expr.uopType, expr.source as ExpressionNode));
            }

            if (expr.uopType === TEUopType.Neg) {
                const obj = BackUtils.fetchAddr(value, ctx.heap);
                if (!obj || obj.type !== SVType.Object) {
                    return ctx.failWithMsg('bad operand type for unary -', expr.source).toSet();
                }

                const func = BackUtils.fetchAddr(obj.getAttr('__neg__'), ctx.heap);
                if (func !== undefined && func.type === SVType.Func) {
                    return functionCall(ctx, func, [], expr.source);
                }
            } else if (expr.uopType === TEUopType.Not) {
                return ctx.toSetWith(SVBool.create(value.type === SVType.None, expr.source));
            }

            return ctx.toSetWith(SVError.create('unimplemented unary syntax', expr.source));
        });
    }

    function _evalName<T>(ctxSet: ContextSet<T>, expr: TEName): ContextSet<ShValue> {
        return ctxSet.map((ctx) => {
            const env = ctx.env;
            const fetch = env.getId(expr.ident);
            if (fetch !== undefined) {
                return ctx.setRetVal(fetch);
            } else {
                const failed = `name '${expr.ident}' does not exist.`;
                return ctx.failWithMsg(failed, expr.source) as Context<ShValue>;
            }
        });
    }

    function _evalAttr<T>(ctxSet: ContextSet<T>, expr: TEAttr): ContextSet<ShValue> {
        return evaluate(ctxSet, expr.left).flatMap((ctx) => getAttrDeep(ctx, ctx.retVal, expr.right, expr.source));
    }

    function _evalSubscr<T>(ctxSet: ContextSet<T>, expr: TESubscr): ContextSet<ShValue> {
        return evaluate(ctxSet, expr.left).flatMap((ctx) => {
            const leftVal = ctx.retVal;
            return evaluate(ctx.toSet(), expr.right).flatMap((ctx) => {
                const rightVal = ctx.retVal;
                const heap = ctx.heap;

                // TODO: Compromise this - preprocssing.
                if (leftVal.type === SVType.Error) {
                    return ctx.toSet(); // propagate error
                }

                const object = BackUtils.fetchAddr(leftVal, heap);
                if (object === undefined || object.type !== SVType.Object) {
                    return ctx.warnWithMsg('object does not exist.', expr.source).toSet();
                }

                let value: ShValue | undefined;
                if (rightVal.type === SVType.Int) {
                    value = getItemByIndex(ctx, object, rightVal.value, rightVal.source);
                } else if (rightVal.type === SVType.String) {
                    value = getItemByKey(ctx, object, rightVal.value, rightVal.source);
                }

                if (value && value.type !== SVType.Error) {
                    return ctx.setRetVal(value).toSet();
                }

                return ctx.getAttrDeep(object, '__getitem__', expr.source).flatMap((ctx) => {
                    const func = ctx.retVal;
                    if (func !== undefined && func.type === SVType.Func) {
                        // cannot fetch value but has attribute '__getitem__'
                        return functionCall(ctx, func, [rightVal], expr.source);
                    } else if (value) {
                        return ctx.setRetVal(value).toSet();
                    } else {
                        return ctx.warnWithMsg('object is not subscriptable', expr.source).toSet();
                    }
                });
            });
        });
    }

    // return item. If idx or len is not a constant, return SVError. If out of bound, return undefined.
    export function getItemByIndex<T>(
        ctx: Context<T>,
        obj: SVObject,
        idx: number | ExpNum,
        source?: ParseNode
    ): ShValue | undefined {
        // TODO: compromise idx is not a constant
        // TODO: typecheck that obj is tensor.
        const idxRng = ctx.ctrSet.getCachedRange(idx);
        if (!idxRng || !idxRng.isConst()) {
            return SVError.create(`cannot infer index ${SymExp.toString(idx)} statically`, source);
        }

        const idNum = idxRng.start;
        let value: ShValue | undefined = obj.getIndice(idNum);

        if (!value && idNum < 0) {
            const objLen = obj.getAttr('$length');
            if (objLen && objLen.type === SVType.Int) {
                const lenRng = ctx.ctrSet.getCachedRange(objLen.value);
                if (!lenRng || !lenRng.isConst() || lenRng.start <= 0) {
                    return SVError.create(
                        `cannot infer object length ${SymExp.toString(objLen.value)} statically`,
                        source
                    );
                }
                value = obj.getIndice(BackUtils.absIndexByLen(lenRng.start, idNum));
            }
        }

        return value;
    }

    // return item. If key is not a constant, return SVError. undefined if key does not exist.
    export function getItemByKey<T>(
        ctx: Context<T>,
        obj: SVObject,
        key: string | ExpString,
        source?: ParseNode
    ): ShValue | undefined {
        // TODO: compromise idx is not a constant
        // TODO: typecheck that obj is tensor.
        const cachedKey = ctx.ctrSet.getCachedString(key);
        if (cachedKey === undefined) {
            return SVError.create(`cannot infer key ${SymExp.toString(key)} statically`, source);
        }

        return obj.getKeyVal(cachedKey);
    }
}
