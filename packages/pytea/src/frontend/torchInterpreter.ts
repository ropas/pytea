/*
 * torchInterpreter.ts
 * Copyright (c) Seoul National University.
 * Licensed under the MIT license.
 * Author: Woosung Song, Sehoon Kim, Ho Young Jhoo
 *
 * PyTea internal language interpreter.
 */

import { List } from 'immutable';

import { ParseNode } from 'pyright-internal/parser/parseNodes';

import { evalLibCall } from '../pylibImplements/interpreter/evaluator';
import { endPoint, functionCall, getItemByIndex, OperatorUtils } from './evalUtils';
import { defaultEnvHeap as builtinEnvHeap, ThEnv, ThHeap } from './torchEnvironments';
import {
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
    TSExpr,
    TSForIn,
    TSFunDef,
    TSIf,
    TSLet,
    TSReturn,
    TSSeq,
    TSType,
} from './torchStatements';
import {
    ThContFlag,
    ThValue,
    TVAddr,
    TVBool,
    TVError,
    TVFloat,
    TVFunc,
    TVInt,
    TVNone,
    TVNotImpl,
    TVObject,
    TVString,
    TVType,
    TVUndef,
} from './torchValues';

export namespace TorchInterpreter {
    let _envLog: ThEnv;
    let _globalFlag = true;

    export function test__printAll(heap: ThHeap) {
        heap = heap._runGC(_envLog);
        console.log('env: ' + _envLog.toString());
        console.log('heap: ' + heap.toString());
    }

    function _saveLog(): [ThEnv, boolean] {
        if (!_envLog) {
            _envLog = new ThEnv();
        }
        const retVal: [ThEnv, boolean] = [_envLog, _globalFlag];
        _envLog = new ThEnv();
        _globalFlag = true;
        return retVal;
    }

    function _loadLog(logs: [ThEnv, boolean]): void {
        _envLog = logs[0];
        _globalFlag = logs[1];
    }

    // return global env & result heap
    export function runEmpty(stmt: ThStmt): [ThEnv, ThHeap] {
        const logs = _saveLog();

        const [, finalHeap] = run(new ThEnv(), new ThHeap(), stmt);
        const retVal: [ThEnv, ThHeap] = [_envLog, finalHeap._runGC(_envLog)];

        _loadLog(logs);
        return retVal;
    }

    export function runDefault(stmt: ThStmt, remainEnv?: boolean): [ThValue | ThContFlag, ThHeap] {
        const logs = _saveLog();

        const [env, heap] = builtinEnvHeap();
        const retVal = run(env, heap, stmt);

        if (!remainEnv) {
            _loadLog(logs);
        }

        return retVal;
    }

    export function runImportModule(stmt: ThStmt): [TVAddr, ThHeap] {
        // expect that first statement is `$module := Object in ...`
        // '$module' will become result module object, and used as imported module.
        if (stmt.stype !== TSType.Let || stmt.name !== '$module') {
            throw 'Imported module statement does not starts with `let $module := Object in ...';
        }

        const logs = _saveLog();

        const [env, heap] = builtinEnvHeap();

        const id = stmt.name;
        const expr = stmt.expr;
        const stmtScope = stmt.scope;

        const [value, newHeap] = evaluate(env, heap, expr!);
        const [addr, newHeap2] = newHeap.allocNew(value, stmt.source);
        const [, newHeap3] = run(env.setId(id, addr), newHeap2, stmtScope);

        const finalHeap = newHeap3._runGC(_envLog);

        _loadLog(logs);
        return [addr, finalHeap];
    }

    export function run(env: ThEnv, heap: ThHeap, stmt: ThStmt): [ThValue | ThContFlag, ThHeap] {
        switch (stmt.stype) {
            case TSType.Pass:
                return _runPass(heap);
            case TSType.Expr:
                return _runExpr(env, heap, stmt);
            case TSType.Seq:
                return _runSeq(env, heap, stmt);
            case TSType.Assign:
                return _runAssign(env, heap, stmt);
            case TSType.If:
                return _runIf(env, heap, stmt);
            case TSType.ForIn:
                return _runForIn(env, heap, stmt);
            case TSType.Return:
                return _runReturn(env, heap, stmt);
            case TSType.Continue:
                return _runContinue(heap);
            case TSType.Break:
                return _runBreak(heap);
            case TSType.Let:
                return _runLet(env, heap, stmt);
            case TSType.FunDef:
                return _runFunDef(env, heap, stmt);
        }
    }

    export function evaluate(env: ThEnv, heap: ThHeap, expr: ThExpr): [ThValue, ThHeap] {
        switch (expr.etype) {
            case TEType.Const:
                return endPoint(_evalConst(heap, expr));
            case TEType.Object:
                return endPoint(_evalObject(heap, expr));
            case TEType.Tuple:
                return endPoint(_evalTuple(env, heap, expr));
            case TEType.Call:
                return endPoint(_evalCall(env, heap, expr));
            case TEType.LibCall:
                return endPoint(_evalLibCall(env, heap, expr));
            case TEType.BinOp:
                return endPoint(_evalBinOp(env, heap, expr));
            case TEType.UnaryOp:
                return endPoint(_evalUnaryOp(env, heap, expr));
            case TEType.Name:
                return endPoint(_evalName(env, heap, expr));
            case TEType.Attr:
                return endPoint(_evalAttr(env, heap, expr));
            case TEType.Subscr:
                return endPoint(_evalSubscr(env, heap, expr));
        }
    }

    function _runPass(heap: ThHeap): [ThValue | ThContFlag, ThHeap] {
        // written by Sehoon Kim
        return [ThContFlag.Run, heap];
    }

    function _runExpr(env: ThEnv, heap: ThHeap, stmt: TSExpr): [ThValue | ThContFlag, ThHeap] {
        // written by Sehoon Kim
        const exp = stmt.expr;
        const [, newHeap] = evaluate(env, heap, exp);

        return [ThContFlag.Run, newHeap];
    }

    function _runSeq(env: ThEnv, heap: ThHeap, stmt: TSSeq): [ThValue | ThContFlag, ThHeap] {
        // written by Sehoon Kim
        const stmt1 = stmt.left;
        const stmt2 = stmt.right;

        let [value, newHeap] = run(env, heap, stmt1);
        if (value === ThContFlag.Run) {
            [value, newHeap] = run(env, newHeap, stmt2);
        }

        return [value, newHeap];
    }

    function _runAssign(env: ThEnv, heap: ThHeap, stmt: TSAssign): [ThValue | ThContFlag, ThHeap] {
        // writtten by Sehoon Kim
        const lexpr = stmt.left;
        const rexpr = stmt.right;

        if (lexpr.etype === TEType.Name) {
            const id = lexpr.ident;
            const addr = env.getId(id);
            if (addr === undefined) {
                return [TVError.create('PyTea Interpreter: Cannot reach here', stmt.source), heap];
            }
            const [value, newHeap] = evaluate(env, heap, rexpr);
            return [ThContFlag.Run, newHeap.setVal(addr, value)];
        } else if (lexpr.etype === TEType.Attr) {
            const exp1 = lexpr.left;
            const id = lexpr.right;
            const exp2 = rexpr;
            let [addr, newHeap] = evaluate(env, heap, exp1);
            if (addr.type !== TVType.Addr) {
                return [TVError.create('PyTea Interpreter: Cannot reach here', stmt.source), heap];
            }
            let value;
            [value, newHeap] = evaluate(env, newHeap, exp2);
            const obj = newHeap.getVal(addr);
            if (obj === undefined || obj.type !== TVType.Object) {
                return [TVError.create('PyTea Interpreter: Cannot reach here', stmt.source), heap];
            }
            const newObj = obj.setAttr(id, value);
            return [ThContFlag.Run, newHeap.setVal(addr, newObj)];
        } else if (lexpr.etype === TEType.Subscr) {
            let newHeap: ThHeap;
            const exp1 = lexpr.left;
            const exp2 = lexpr.right;
            const exp3 = rexpr;
            const [addr, newHeap1] = evaluate(env, heap, exp1);
            newHeap = newHeap1;
            if (addr.type !== TVType.Addr) {
                return [TVError.create('PyTea Interpreter: Cannot reach here', stmt.source), heap];
            }
            let value_i: ThValue;
            let value: ThValue;
            [value_i, newHeap] = evaluate(env, newHeap, exp2);
            [value, newHeap] = evaluate(env, newHeap, exp3);
            const obj = newHeap.getVal(addr);
            if (obj === undefined || obj.type !== TVType.Object) {
                return [TVError.create('PyTea Interpreter: Cannot reach here', stmt.source), heap];
            }
            const func = obj.getAttr('__setitem__');
            if (func === undefined) {
                if (value_i.type !== TVType.Int) {
                    return [TVError.create('PyTea Interpreter: Cannot reach here', stmt.source), heap];
                }
                const idx = (value_i as TVInt).value;
                const newObj = obj.setIndice(idx, value);
                return [ThContFlag.Run, newHeap.setVal(addr, newObj)];
            } else {
                if (func.type !== TVType.Func) {
                    return [TVError.create('PyTea Interpreter: Cannot reach here', stmt.source), heap];
                }
                newHeap = _functionCallWrap(env, newHeap, func, [value_i, value], exp1.source)[1];
                return [ThContFlag.Run, newHeap];
            }
        } else {
            return [TVError.create('PyTea Interpreter: Cannot reach here', stmt.source), heap];
        }
    }

    function _runIf(env: ThEnv, heap: ThHeap, stmt: TSIf): [ThValue | ThContFlag, ThHeap] {
        // written by Sehoon Kim
        const exp = stmt.cond;
        const stmt_t = stmt.thenStmt;
        const stmt_f = stmt.elseStmt;

        const [value, newHeap] = evaluate(env, heap, exp);
        if (
            (value.type === TVType.Bool && value.value === false) ||
            (value.type === TVType.Int && value.value === 0) ||
            (value.type === TVType.Float && value.value === 0.0) ||
            (value.type === TVType.String && value.value === '') ||
            value.type === TVType.None
        ) {
            return run(env, newHeap, stmt_f);
        } else if (
            (value.type === TVType.Bool && value.value !== false) ||
            (value.type === TVType.Int && value.value !== 0) ||
            (value.type === TVType.Float && value.value !== 0.0) ||
            (value.type === TVType.String && value.value !== '')
        ) {
            return run(env, newHeap, stmt_t);
        } else {
            return [TVError.create('PyTea Interpreter: Cannot reach here', stmt.source), heap];
        }
    }

    function _runForIn(env: ThEnv, heap: ThHeap, stmt: TSForIn): [ThValue | ThContFlag, ThHeap] {
        // written by Sehoon Kim
        const ids = stmt.ident;
        const exp = stmt.loopVal;
        const stmt_body = stmt.loopBody;

        let [addr, newHeap] = evaluate(env, heap, exp);
        if (addr.type !== TVType.Addr) {
            return [TVError.create('PyTea Interpreter: Cannot reach here', stmt.source), heap];
        }
        const obj = newHeap.getVal(addr);
        if (obj === undefined || obj.type !== TVType.Object) {
            return [TVError.create('PyTea Interpreter: Cannot reach here', stmt.source), heap];
        }

        // attrs
        let len;
        if (obj.getAttr('__len__') !== undefined) {
            const func = obj.getAttr('__len__');
            let len_;
            [len_, newHeap] = _functionCallWrap(env, newHeap, func as TVFunc, []);
            len = (len_ as TVInt).value;
        }
        // TODO: keyValues
        // else if (o.getKeyVal('length') !== undefined) { }
        // indices
        else {
            len = obj.indices.size;
        }

        const addr_arg = [];
        let addr_temp;
        for (let i = 0; i < ids.length; i++) {
            [addr_temp, newHeap] = newHeap.malloc(stmt.source);
            addr_arg.push(addr_temp);
            env = env.setId(ids[i], addr_arg[i]);
        }

        let value;
        let attr;
        let addr_;
        let subObj;
        let arg;
        for (let i = 0; i < len; i++) {
            if (ids.length === 1) {
                [attr, newHeap] = getItemByIndex(env, newHeap, obj, i);
                if (attr === undefined) {
                    return [TVError.create('PyTea Interpreter: Cannot reach here', stmt.source), heap];
                }
                newHeap = newHeap.setVal(addr_arg[0], attr);
            } else {
                [attr, newHeap] = getItemByIndex(env, newHeap, obj, i);
                addr_ = attr as TVAddr;
                [addr_, newHeap] = endPoint([addr_, newHeap]);
                subObj = newHeap.getVal(addr_ as TVAddr);
                for (let j = 0; j < ids.length; j++) {
                    [arg, newHeap] = getItemByIndex(env, newHeap, subObj as TVObject, j);
                    newHeap = newHeap.setVal(addr_arg[j], arg);
                }
            }

            [value, newHeap] = run(env, newHeap, stmt_body);
            if (value === ThContFlag.Brk) {
                return [ThContFlag.Run, newHeap];
            } else if (value !== ThContFlag.Run && value !== ThContFlag.Cnt) {
                return [value, newHeap];
            }
        }

        return [ThContFlag.Run, newHeap];
    }

    function _runReturn(env: ThEnv, heap: ThHeap, stmt: TSReturn): [ThValue, ThHeap] {
        // written by Sehoon Kim
        const exp = stmt.expr;
        return evaluate(env, heap, exp);
    }

    function _runContinue(heap: ThHeap): [ThValue | ThContFlag, ThHeap] {
        // written by Sehoon Kim
        return [ThContFlag.Cnt, heap];
    }

    function _runBreak(heap: ThHeap): [ThValue | ThContFlag, ThHeap] {
        // written by Sehoon Kim
        return [ThContFlag.Brk, heap];
    }

    function _runLet(env: ThEnv, heap: ThHeap, stmt: TSLet): [ThValue | ThContFlag, ThHeap] {
        // written by Sehoon Kim
        const id = stmt.name;
        const exp = stmt.expr;
        const stmt_scope = stmt.scope;

        let addr, value, newHeap;
        if (exp === undefined) {
            [addr, newHeap] = heap.allocNew(TVUndef.create(stmt.source), stmt.source);
            [value, newHeap] = run(env.setId(id, addr), newHeap, stmt_scope);

            if (_globalFlag) {
                // for debug
                _envLog = _envLog.setId(id, addr);
            }

            return [value, newHeap];
        } else {
            let newValue;
            [value, newHeap] = evaluate(env, heap, exp);
            [addr, newHeap] = newHeap.allocNew(value, stmt.source);
            [newValue, newHeap] = run(env.setId(id, addr), newHeap, stmt_scope);

            if (_globalFlag) {
                // for debug
                _envLog = _envLog.setId(id, addr);
            }

            return [newValue, newHeap];
        }
    }

    function _runFunDef(env: ThEnv, heap: ThHeap, stmt: TSFunDef): [ThValue | ThContFlag, ThHeap] {
        // written by Sehoon Kim
        const funName = stmt.name;
        const params = stmt.params;
        const stmt_body = stmt.body;
        const stmt_scope = stmt.scope;

        const func = TVFunc.create(funName, List(params), stmt_body, env);

        let newHeap = heap;
        let addr = env.getId(funName);
        let newEnv = env;

        if (!addr) {
            [addr, newHeap] = newHeap.malloc();
            newEnv = env.setId(funName, addr);
        }

        newHeap = newHeap.setVal(addr, func);
        const [value, newHeap2] = run(newEnv, newHeap, stmt_scope);

        return [value, newHeap2];
    }

    function _evalConst(heap: ThHeap, expr: TEConst): [ThValue, ThHeap] {
        // written by Woosung Song
        switch (expr.constType) {
            case TEConstType.Int:
                return [TVInt.create(expr.value as number, expr.source), heap];
            case TEConstType.Float:
                return [TVFloat.create(expr.value as number, expr.source), heap];
            case TEConstType.String:
                return [TVString.create(expr.value as string, expr.source), heap];
            case TEConstType.Bool:
                return [TVBool.create(expr.value as boolean, expr.source), heap];
            case TEConstType.None:
                return [TVNone.create(expr.source), heap];
        }
    }

    function _evalObject(heap: ThHeap, expr: TEObject): [ThValue, ThHeap] {
        // written by Woosung Song
        return heap.allocNew(TVObject.create(expr.source), expr.source);
    }

    function _evalTuple(env: ThEnv, heap: ThHeap, expr: TETuple): [ThValue, ThHeap] {
        // written by Woo Sung Song
        let tuple = TVObject.create(expr.source);
        tuple = tuple.setAttr('length', TVInt.create(expr.values.length, expr.source));

        let newHeap = heap;
        tuple = expr.values.reduce((tupleObject, value, index) => {
            const [element, nextHeap] = evaluate(env, newHeap, value);
            newHeap = nextHeap;
            return tupleObject.setIndice(index, element);
        }, tuple);
        return newHeap.allocNew(tuple, expr.source);
    }

    function _evalCall(env: ThEnv, heap: ThHeap, expr: TECall): [ThValue, ThHeap] {
        // written by Woosung Song
        let newHeap = heap;

        // calculate E part
        const [func, nextHeap1] = evaluate(env, heap, expr.func);
        newHeap = nextHeap1;
        if (!func) {
            // actually an error.

            return [TVError.create('PyTea Interpreter: Try to call an unknown object.', expr.source), newHeap];
        }

        // calculate E1, E2, ..., En part
        const args = expr.params.map((value) => {
            const [calc, nextHeap] = evaluate(env, newHeap, value);
            newHeap = nextHeap;

            return calc;
        });

        // function and object's call are different
        let f: TVFunc | undefined = undefined;
        if (func.type === TVType.Func) {
            f = func as TVFunc;
        } else if (func.type === TVType.Addr) {
            // object call, for example, conv2d(..)
            f = (newHeap.getVal(func) as TVObject).getAttr('__call__') as TVFunc;
        }

        if (f === undefined) {
            // how do I handle this error?

            return [TVError.create('PyTea Interpreter: The given object is not callable.', expr.source), newHeap];
        }

        return _functionCallWrap(env, newHeap, f, args, expr.source);
    }

    function _evalLibCall(env: ThEnv, heap: ThHeap, expr: TELibCall): [ThValue, ThHeap] {
        const _oldFlag = _globalFlag;
        _globalFlag = false;

        const retVal = evalLibCall(env, heap, expr);
        _globalFlag = _oldFlag;

        return retVal;
    }

    function _evalBinOp(env: ThEnv, heap: ThHeap, expr: TEBinOp): [ThValue, ThHeap] {
        // written by Woosung Song
        let newHeap = heap;
        let [left, nextHeap1] = evaluate(env, newHeap, expr.left);
        let [right, nextHeap2] = evaluate(env, nextHeap1, expr.right);
        newHeap = nextHeap2;

        if (left.type === TVType.String && right.type === TVType.String) {
            return [OperatorUtils.evalStringTypeBinOp(left.value, right.value, expr.bopType, expr.source), newHeap];
        }

        // if left and right variables are swapped
        let swapped = false;

        // note that element type is one of Bool, Int and Float.
        if (OperatorUtils.isElementType(left.type)) {
            if (OperatorUtils.isElementType(right.type)) {
                return [OperatorUtils.evalElementTypeBinOp(left, right, expr.bopType, expr.source), newHeap];
            } else {
                // swap left and right
                [left, right] = [right, left];
                swapped = true;
            }
        }

        const leftAddr = left as TVAddr;
        const rightAddr = right as TVAddr;

        // exceptional operators
        // TODO: check function compare by id.
        if (expr.bopType === TEBopType.Is) {
            return [TVBool.create(leftAddr === rightAddr, expr.source), newHeap];
        } else if (expr.bopType === TEBopType.IsNot) {
            return [TVBool.create(leftAddr !== rightAddr, expr.source), newHeap];
        }

        const leftObject = newHeap.getVal(leftAddr);
        if (leftObject === undefined || leftObject.type !== TVType.Object) {
            // error
            return [TVError.create('PyTea Interpreter: Try to access non-existing object.', expr.source), newHeap];
        }

        const leftBopString = OperatorUtils.binOpString(!swapped, expr.bopType);
        if (leftBopString !== undefined) {
            const func = leftObject.getAttr(leftBopString);
            if (func !== undefined && func.type === TVType.Func) {
                const [value, nextHeap3] = _functionCallWrap(env, newHeap, func, [right], expr.source);
                newHeap = nextHeap3;

                if (value.type !== TVType.NotImpl) {
                    if (expr.bopType !== TEBopType.NotIn) {
                        return [value, newHeap];
                    } else if (value.type === TVType.Bool) {
                        // .NotIn is implemented by !(.In(.))
                        return [TVBool.create(!value.value, expr.source), newHeap];
                    }
                }
                // If value.type is TVType.NotImpl, then do __radd__
            }
        }

        // do radd
        const rightObject = newHeap.getVal(rightAddr);
        if (rightObject === undefined || rightObject.type !== TVType.Object) {
            // error
            return [TVError.create('PyTea Interpreter: Try to access non-existing object.', expr.source), newHeap];
        }

        const rightBopString = OperatorUtils.binOpString(swapped, expr.bopType);
        if (rightBopString !== undefined) {
            const func = rightObject.getAttr(rightBopString);
            if (func !== undefined && func.type === TVType.Func) {
                const [value, nextHeap3] = _functionCallWrap(env, newHeap, func, [left], expr.source);
                newHeap = nextHeap3;

                if (value.type !== TVType.NotImpl) {
                    if (expr.bopType !== TEBopType.NotIn) {
                        return [value, newHeap];
                    } else if (value.type === TVType.Bool) {
                        // .NotIn is implemented by !(.In(.))
                        return [TVBool.create(!value.value, expr.source), newHeap];
                    }
                }
            }
        }

        // both __add__ and __radd__ are not implemented.
        return [
            TVNotImpl.create('PyTea Interpreter: The given binary operator is not implemented.', expr.source),
            newHeap,
        ];
    }

    function _evalUnaryOp(env: ThEnv, heap: ThHeap, expr: TEUnaryOp): [ThValue, ThHeap] {
        // written by Woosung Song
        let [value, newHeap] = evaluate(env, heap, expr.base);

        if (OperatorUtils.isElementType(value.type)) {
            return [OperatorUtils.evalElementTypeUnaryOp(value, expr.uopType, expr.source), newHeap];
        }

        if (expr.uopType === TEUopType.Neg) {
            const valueAddr = value as TVAddr;

            const object = newHeap.getVal(valueAddr);
            if (object === undefined || object.type !== TVType.Object) {
                // error
                return [TVError.create('PyTea Interpreter: Try to access non-existing object.', expr.source), newHeap];
            }

            const func = object.getAttr('__neg__');
            if (func !== undefined && func.type === TVType.Func) {
                const [value, nextHeap1] = _functionCallWrap(env, newHeap, func, [], expr.source);
                newHeap = nextHeap1;

                if (value.type !== TVType.NotImpl) {
                    return [value, newHeap];
                }
            }
        } else if (expr.uopType === TEUopType.Not) {
            if (value.type === TVType.None) {
                return [TVBool.create(true, expr.source), newHeap];
            } else {
                // not (object) is True in python
                return [TVBool.create(false, expr.source), newHeap];
            }
        }

        return [
            TVError.create('PyTea Interpreter: Try to call an operator that is not supported.', expr.source),
            newHeap,
        ];
    }

    function _evalName(env: ThEnv, heap: ThHeap, expr: TEName): [ThValue, ThHeap] {
        // written by Woosung Song

        if (expr.ident === 'NotImplemented') {
            return [TVNotImpl.create('explicit call', expr.source), heap];
        }

        const fetch = env.getId(expr.ident);
        if (fetch !== undefined) {
            return [fetch, heap];
        } else {
            return [TVError.create('PyTea Interpreter: The variable name does not exist.', expr.source), heap];
        }
    }

    function _evalAttr(env: ThEnv, heap: ThHeap, expr: TEAttr): [ThValue, ThHeap] {
        // written by Woosung Song
        let newHeap = heap;
        const [left, nextHeap] = evaluate(env, heap, expr.left);
        newHeap = nextHeap;

        if (left.type === TVType.Addr) {
            // object
            const object = newHeap.getVal(left);
            if (object === undefined || object.type !== TVType.Object) {
                return [TVError.create('PyTea Interpreter: The object does not exist.', expr.source), newHeap];
            }
            const attr = object.getAttr(expr.right);

            if (attr === undefined) {
                const getAttr = object.getAttr('__getattr__');
                if (getAttr && getAttr.type === TVType.Func) {
                    return _functionCallWrap(
                        env,
                        newHeap,
                        getAttr,
                        [TVString.create(expr.right, expr.source)],
                        expr.source
                    );
                } else {
                    return [
                        TVError.create(`PyTea Interpreter: Atribute ${expr.right} not exist.`, expr.source),
                        newHeap,
                    ];
                }
            }
            return [attr, newHeap];
        } else {
            // elementType: Int, Bool, Float, String
            const attr = left.attrs.get(expr.right);

            if (attr === undefined) {
                return [
                    TVError.create(`PyTea Interpreter: Attribute ${expr.right} does not exist.`, expr.source),
                    newHeap,
                ];
            }
            return [attr, newHeap];
        }
    }

    function _evalSubscr(env: ThEnv, heap: ThHeap, expr: TESubscr): [ThValue, ThHeap] {
        // written by Woosung Song
        let newHeap = heap;
        const [left, nextHeap1] = evaluate(env, heap, expr.left);
        const [right, nextHeap2] = evaluate(env, nextHeap1, expr.right);
        newHeap = nextHeap2;

        if (left.type !== TVType.Addr) {
            return [TVError.create('PyTea Interpreter: Try to subscript of a non-object type.', expr.source), newHeap];
        }
        const object = newHeap.getVal(left);
        if (object === undefined || object.type !== TVType.Object) {
            return [TVError.create('PyTea Interpreter: The object does not exist.', expr.source), newHeap];
        }

        const func = object.getAttr('__getitem__');
        if (func !== undefined && func.type === TVType.Func) {
            // has attribute '__getitem__'
            return _functionCallWrap(env, newHeap, func, [right], expr.source);
        } else if (right.type === TVType.Int) {
            // no attribute. call getIndice
            const value = object.getIndice(right.value);
            if (value !== undefined) {
                return [value, newHeap];
            } else {
                return [TVError.create('PyTea Interpreter: Index out of the bound.', expr.source), newHeap];
            }
        } else if (right.type === TVType.String) {
            const value = object.getKeyVal(right.value);
            if (value !== undefined) {
                return [value, newHeap];
            } else {
                return [TVError.create('PyTea Interpreter: Subscription key does not exist', expr.source), newHeap];
            }
        }

        return [TVError.create('PyTea Interpreter: This object is not subscriptable.', expr.source), newHeap];
    }

    function _functionCallWrap(
        env: ThEnv,
        heap: ThHeap,
        f: TVFunc,
        args: ThValue[],
        source?: ParseNode,
        kwargs?: { [paramName: string]: ThValue }
    ): [ThValue, ThHeap] {
        const _oldFlag = _globalFlag;
        _globalFlag = false;

        const retVal = functionCall(env, heap, f, args, source, kwargs);
        _globalFlag = _oldFlag;

        return retVal;
    }
}
