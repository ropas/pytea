import { fetchAddr, fetchSize, sanitizeAddr, trackMro } from '../backend/backUtils';
import { Context, ContextSet } from '../backend/context';
import { isInstanceOf, simplifyNum, strLen } from '../backend/expUtils';
import * as ExpUtils from '../backend/expUtils';
import {
    CodeSource,
    PrimitiveType,
    ShValue,
    SVAddr,
    SVBool,
    SVErrorLevel,
    SVFloat,
    SVInt,
    SVNone,
    SVNotImpl,
    SVObject,
    SVString,
    SVType,
} from '../backend/sharpValues';
import { ExpNum, ExpNumSymbol, NumBopType, NumUopType, StringOpType, SymExp } from '../backend/symExpressions';
import { TorchBackend } from '../backend/torchBackend';
import { PyteaService } from '../service/pyteaService';
import { formatCodeSource } from '../service/pyteaUtils';
import { LCImpl } from '.';
import { LCBase } from './libcall';

export namespace BuiltinsLCImpl {
    export function superGetAttr(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.builtins.superGetAttr': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }

        const { env, heap } = ctx;
        const [selfClass, self, attrAddr] = params;

        const mro = trackMro(selfClass, heap, env);
        const attr = fetchAddr(attrAddr, heap);

        if (attr?.type !== SVType.String || typeof attr.value !== 'string') {
            return ctx.warnWithMsg(`from 'LibCall.builtins.superGetAttr': attr is not a string`, source).toSet();
        }

        if (mro.length <= 1 || mro[1] === undefined) {
            return ctx.warnWithMsg(`from 'LibCall.builtins.superGetAttr': has no superclass`, source).toSet();
        }

        const superClass = fetchAddr(SVAddr.create(mro[1], source), heap);
        if (superClass?.type !== SVType.Object) {
            return ctx.warnWithMsg(`from 'LibCall.builtins.superGetAttr': superclass is not an object`, source).toSet();
        }

        return TorchBackend.getAttrDeep(ctx, superClass, attr.value, source).map((ctx) => {
            const retVal = ctx.retVal;
            if (retVal.type === SVType.Func && self.type === SVType.Addr) {
                return ctx.setRetVal(retVal.bound(self));
            }
            return ctx;
        });
    }

    export function isinstance(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.builtins.isinstance': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }

        const { env, heap } = ctx;
        const [selfAddr, classAddr] = params;

        const result = isInstanceOf(selfAddr, classAddr, env, heap);
        if (result === undefined) {
            return ctx.warnWithMsg(`from 'LibCall.builtins.isinstance': got invalid address`, source).toSet();
        }

        return ctx.toSetWith(SVBool.create(result, source));
    }

    export function cast(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx
                .failWithMsg(
                    `from 'LibCall.builtins.cast': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }

        const { env, heap } = ctx;
        const value = fetchAddr(params[0], heap);
        const type = fetchAddr(params[1], heap);
        // const kwargs = fetchAddr(params[2], heap)

        if (!value || type?.type !== SVType.Int || typeof type.value !== 'number') {
            return ctx.failWithMsg(`from 'LibCall.builtins.cast': invalid value type`, source).toSet();
        }

        // create empty value
        if (value.type === SVType.None) {
            let typename: string;
            switch (type.value) {
                case PrimitiveType.Int:
                    return ctx.toSetWith(SVInt.create(0, source));
                case PrimitiveType.Float:
                    return ctx.toSetWith(SVFloat.create(0.0, source));
                case PrimitiveType.Str:
                    return ctx.toSetWith(SVString.create('None', source));
                case PrimitiveType.Tuple:
                    typename = 'tuple';
                    break;
                case PrimitiveType.List:
                    typename = 'list';
                    break;
                case PrimitiveType.Dict:
                    typename = 'dict';
                    break;
                case PrimitiveType.Set:
                    typename = 'set';
                    break;
                default:
                    return ctx.setRetVal(SVNotImpl.create('not implemented casting', source)).toSet();
            }

            const [obj, addr, newHeap] = SVObject.create(heap, source);
            const typeMro = (fetchAddr(heap.getVal(env.getId(typename)!)!, heap) as SVObject).getAttr('__mro__')!;
            return ctx
                .setHeap(
                    newHeap.setVal(addr, obj.setAttr('__mro__', typeMro).setAttr('$length', SVInt.create(0, source)))
                )
                .toSetWith(addr);
        }

        switch (type.value) {
            // TODO: floor
            case PrimitiveType.Int: {
                switch (value.type) {
                    case SVType.Int:
                        return ctx.toSetWith(value);
                    case SVType.Float:
                        if (typeof value.value === 'number') {
                            return ctx.toSetWith(SVInt.create(Math.floor(value.value), source));
                        } else {
                            return ctx.toSetWith(
                                SVInt.create(ExpNum.uop(NumUopType.Floor, value.value, source), source)
                            );
                        }
                    case SVType.String:
                        {
                            if (typeof value.value === 'string') {
                                const intVal = Number.parseInt(value.value);
                                if (intVal.toString() === value.value)
                                    return ctx.toSetWith(SVInt.create(intVal, source));
                            }
                        }
                        break;
                    case SVType.Object: {
                        const size = fetchSize(value, heap);
                        if (typeof size !== 'string') {
                            const ctrRank0 = ctx.genEq(size.rank(), 0, source);
                            const ctrRank1 = ctx.genAnd(
                                ctx.genEq(size.rank(), 1, source),
                                ctx.genEq(ExpNum.index(size.shape, 0, source), 1, source),
                                source
                            );
                            return ctx
                                .require(
                                    [ctx.genOr(ctrRank0, ctrRank1, source)],
                                    `from 'LibCall.builtins.cast': must be scalar or tensor with 1 elem`,
                                    source
                                )
                                .return(SVInt.create(ExpNum.fromSymbol(ctx.genSymInt('tensorElem', source)), source));
                        }
                        break;
                    }
                    default:
                        break;
                }
                return ctx
                    .addLog('int parsing of unknown value', source)
                    .toSetWith(SVInt.create(ExpNum.fromSymbol(ctx.genSymInt('parseInt', source)), source));
            }
            case PrimitiveType.Tuple: {
                const list = env.getId('list')!;
                const tuple = env.getId('tuple')!;
                if (value.type === SVType.Object && isInstanceOf(value, list, env, heap)) {
                    const mro = fetchAddr(value.getAttr('__mro__'), heap);
                    if (mro?.type === SVType.Object) {
                        // force casting
                        const tupleObj = fetchAddr(heap.getVal(tuple)!, heap)!;
                        const tupleMro = (tupleObj as SVObject).getAttr('__mro__')!;
                        const casted = value.setAttr('__mro__', tupleMro);

                        return ctx.setHeap(ctx.heap.setVal(casted.addr, casted)).toSetWith(casted.addr);
                    }
                } else if (isInstanceOf(value, tuple, env, heap)) {
                    return ctx.toSetWith(value);
                }
                break;
            }
            case PrimitiveType.List: {
                const list = env.getId('list')!;
                const tuple = env.getId('tuple')!;
                if (value.type === SVType.Object && isInstanceOf(value, tuple, env, heap)) {
                    const mro = fetchAddr(value.getAttr('__mro__'), heap);
                    if (mro?.type === SVType.Object) {
                        // force casting
                        const listObj = fetchAddr(heap.getVal(list)!, heap)!;
                        const listMro = (listObj as SVObject).getAttr('__mro__')!;
                        const casted = value.setAttr('__mro__', listMro);
                        return ctx.setHeap(ctx.heap.setVal(casted.addr, casted)).toSetWith(casted.addr);
                    }
                } else if (isInstanceOf(value, list, env, heap)) {
                    return ctx.toSetWith(value);
                }
                break;
            }
            case PrimitiveType.Float: {
                switch (value.type) {
                    case SVType.Int:
                        return ctx.toSetWith(SVFloat.create(value.value, source));
                    case SVType.Float:
                        return ctx.toSetWith(value);
                    case SVType.String:
                        {
                            if (typeof value.value === 'string') {
                                const floatVal = Number.parseFloat(value.value);
                                if (floatVal.toString() === value.value)
                                    return ctx.toSetWith(SVFloat.create(floatVal, source));
                            }
                        }
                        break;
                    case SVType.Object: {
                        const size = fetchSize(value, heap);
                        if (typeof size !== 'string') {
                            const ctrRank0 = ctx.genEq(size.rank(), 0, source);
                            const ctrRank1 = ctx.genAnd(
                                ctx.genEq(size.rank(), 1, source),
                                ctx.genEq(ExpNum.index(size.shape, 0, source), 1, source),
                                source
                            );
                            return ctx
                                .require(
                                    [ctx.genOr(ctrRank0, ctrRank1, source)],
                                    `from 'LibCall.builtins.cast': must be scalar or tensor with 1 elem`,
                                    source
                                )
                                .return(
                                    SVFloat.create(ExpNum.fromSymbol(ctx.genSymFloat('tensorElem', source)), source)
                                );
                        }
                        break;
                    }
                    default:
                        break;
                }
                return ctx
                    .addLog('float parsing of unknown value', source)
                    .toSetWith(SVFloat.create(ExpNum.fromSymbol(ctx.genSymFloat('parseFloat', source)), source));
            }
            case PrimitiveType.Str:
                // TODO: symoblic value into symbolic string
                switch (value.type) {
                    case SVType.Int:
                    case SVType.Float:
                        return ctx.toSetWith(SVString.create(value.value.toString(), source));
                    case SVType.String:
                        return ctx.toSetWith(value);
                    default:
                        // TODO: call __str__
                        return ctx.toSetWith(SVString.create(value.toString(), source));
                }
            case PrimitiveType.Bool:
            case PrimitiveType.Dict:
            case PrimitiveType.Set:
                break;
        }

        return ctx.setRetVal(SVNotImpl.create('not implemented casting', source)).toSet();
    }

    export function list_append(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx
                .failWithMsg(
                    `from 'LibCall.builtins.list_append': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }

        const { heap } = ctx;
        const list = fetchAddr(params[0], heap);
        const value = sanitizeAddr(params[1], heap);

        if (list?.type !== SVType.Object || !value) {
            return ctx.warnWithMsg(`from 'LibCall.builtins.list_append': invalid value type`, source).toSet();
        }

        return len(ctx.setRetVal({ params: [list] }), source).map((ctx) => {
            const listLen = fetchAddr(ctx.retVal, ctx.heap);
            if (listLen?.type !== SVType.Int) {
                return ctx.warnWithMsg(`from 'LibCall.builtins.list_append': cannot fetch length of list`, source);
            }

            const listLenExp = listLen.value;
            if (typeof listLenExp === 'number') {
                const listObj = list
                    .setAttr('$length', SVInt.create(listLenExp + 1, source))
                    .setIndice(listLenExp, value);
                return ctx.setHeap(ctx.heap.setVal(list.addr, listObj));
            }

            const lenRng = ctx.getCachedRange(listLen.value)?.toIntRange();
            if (lenRng) {
                if (lenRng.isConst()) {
                    const constLen = lenRng.start;
                    const listObj = list
                        .setAttr('$length', SVInt.create(constLen + 1, source))
                        .setIndice(constLen, value);
                    return ctx.setHeap(ctx.heap.setVal(list.addr, listObj));
                } else {
                    // list length is symbolic. failed to append
                    const listObj = list.setAttr(
                        '$length',
                        SVInt.create(ExpNum.bop(NumBopType.Add, listLenExp, 1, source), source)
                    );
                    return ctx
                        .addLog(
                            `from 'LibCall.builtins.list_append: symbolic lengthed list - ${ExpNum.toString(
                                listLenExp
                            )}. just increment length`,
                            source
                        )
                        .setHeap(ctx.heap.setVal(list.addr, listObj));
                }
            } else {
                return ctx.failWithMsg(
                    `from 'LibCall.builtins.list_append: invalid list length ${ExpNum.toString(listLenExp)}`,
                    source
                );
            }
        });
    }

    export function dict_items(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 1) {
            return ctx
                .failWithMsg(
                    `from 'LibCall.builtins.dict_items': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }

        const { heap } = ctx;
        const dict = fetchAddr(params[0], heap);

        if (dict?.type !== SVType.Object) {
            return ctx.warnWithMsg(`from 'LibCall.builtins.dict_items': invalid value type`, source).toSet();
        }

        let newCtx = ctx;
        const items: ShValue[] = [];

        dict.keyValues.forEach((value, key) => {
            const setTuple = newCtx.genTuple([SVString.create(key, source), value], source);
            items.push(setTuple[1]);
            newCtx = setTuple[2];
        });

        const [, itemList, finalCtx] = newCtx.genList(items, source);
        return finalCtx.toSetWith(itemList);
    }

    export function dict_keys(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 1) {
            return ctx
                .failWithMsg(
                    `from 'LibCall.builtins.dict_keys': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }

        const { heap } = ctx;
        const dict = fetchAddr(params[0], heap);

        if (dict?.type !== SVType.Object) {
            return ctx.warnWithMsg(`from 'LibCall.builtins.dict_keys': invalid value type`, source).toSet();
        }

        const items: ShValue[] = [];

        dict.keyValues.forEach((value, key) => {
            items.push(SVString.create(key, source));
        });

        const [, itemList, finalCtx] = ctx.genList(items, source);
        return finalCtx.toSetWith(itemList);
    }

    export function dict_values(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 1) {
            return ctx
                .failWithMsg(
                    `from 'LibCall.builtins.dict_values': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }

        const { heap } = ctx;
        const dict = fetchAddr(params[0], heap);

        if (dict?.type !== SVType.Object) {
            return ctx.warnWithMsg(`from 'LibCall.builtins.dict_values': invalid value type`, source).toSet();
        }

        const items: ShValue[] = [];
        dict.keyValues.forEach((value) => {
            items.push(value);
        });

        const [, itemList, finalCtx] = ctx.genList(items, source);
        return finalCtx.toSetWith(itemList);
    }

    // TODO: fix this to support non-string typed key
    export function dict_setitem(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx
                .failWithMsg(
                    `from 'LibCall.builtins.dict_setitem': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }

        const { heap } = ctx;
        let dict = fetchAddr(params[0], heap);
        const key = fetchAddr(params[1], heap);
        const value = sanitizeAddr(params[2], heap);
        let isNewKey = true;

        if (dict?.type !== SVType.Object || !key || !value) {
            return ctx.warnWithMsg(`from 'LibCall.builtins.dict_setitem': invalid value type`, source).toSet();
        }
        if (key.type === SVType.Int) {
            const keyRng = ctx.getCachedRange(key.value);
            if (keyRng?.toIntRange()?.isConst()) {
                if (dict.indices.has(keyRng.start)) isNewKey = false;
                dict = dict.setIndice(keyRng.start, value);
            } else {
                return ctx.failWithMsg(`from 'LibCall.builtins.dict_setitem': invalid key type`, source).toSet();
            }
        } else if (key.type === SVType.String && typeof key.value === 'string') {
            if (dict.keyValues.has(key.value)) isNewKey = false;
            dict = dict.setKeyVal(key.value, value);
        } else {
            return ctx
                .warnWithMsg(`from 'LibCall.builtins.dict_setitem': currently only supports int or string key`, source)
                .toSet();
        }

        if (isNewKey) {
            const len = dict.getAttr('$length');
            if (len?.type !== SVType.Int) {
                return ctx
                    .warnWithMsg(`from 'LibCall.builtins.dict_setitem': dict length is not an integer type`, source)
                    .toSet();
            }
            const newLen = simplifyNum(ctx.ctrSet, ExpNum.bop(NumBopType.Add, len.value, 1, source));
            dict = dict.setAttr('$length', SVInt.create(newLen, source));
        }

        return ctx.setHeap(heap.setVal(dict.addr, dict)).toSetWith(SVNone.create(source));
    }

    // TODO: fix this to support non-string typed key
    export function dict_getitem(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx
                .failWithMsg(
                    `from 'LibCall.builtins.dict_getitem': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }

        const { heap } = ctx;
        const dict = fetchAddr(params[0], heap);
        const key = fetchAddr(params[1], heap);
        let value;

        if (dict?.type !== SVType.Object || !key) {
            return ctx.warnWithMsg(`from 'LibCall.builtins.dict_getitem': invalid value type`, source).toSet();
        }
        if (key.type === SVType.Int) {
            const keyRng = ctx.getCachedRange(key.value);
            if (keyRng?.toIntRange()?.isConst()) {
                value = dict.getIndice(keyRng.start);
            } else {
                return ctx
                    .failWithMsg(`from 'LibCall.builtins.dict_getitem': invalid key type ${keyRng}`, source)
                    .toSet();
            }
        } else if (key.type === SVType.String && typeof key.value === 'string') {
            value = dict.getKeyVal(key.value);
        } else {
            return ctx
                .warnWithMsg(`from 'LibCall.builtins.dict_setitem': currently only supports int or string key`, source)
                .toSet();
        }

        if (value === undefined) {
            return ctx.failWithMsg(`LibCall.builtins.dict_getitem': invalid key value`, source).toSet();
        }

        return ctx.toSetWith(value);
    }

    // TODO: fix this to support non-string typed key
    export function dict_pop(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx
                .failWithMsg(
                    `from 'LibCall.builtins.dict_setitem': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }

        const { heap } = ctx;
        const dict = fetchAddr(params[0], heap);
        const key = fetchAddr(params[1], heap);
        const defaultVal = sanitizeAddr(params[2], heap);

        if (dict?.type !== SVType.Object || key?.type !== SVType.String || !defaultVal) {
            // currently, only supports string typed key
            return ctx.warnWithMsg(`from 'LibCall.builtins.dict_pop': invalid value type`, source).toSet();
        }
        if (typeof key.value !== 'string') {
            return ctx
                .warnWithMsg(`from 'LibCall.builtins.dict_pop': does not supports symbolic string`, source)
                .toSet();
        }

        let retVal = dict.getKeyVal(key.value);
        if (retVal === undefined) {
            retVal = defaultVal;
        }

        return ctx.toSetWith(retVal);
    }

    export function str_islower(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 1) {
            return ctx
                .failWithMsg(
                    `from 'LibCall.builtins.str_islower': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }

        const { heap } = ctx;
        const str = fetchAddr(params[0], heap);

        if (str?.type !== SVType.String) {
            return ctx.warnWithMsg(`from 'LibCall.builtins.str_islower': value is not a string`, source).toSet();
        }

        let value = str.value;
        if (typeof value !== 'string') {
            // TODO: calculate symbolic string
            const exp = ExpUtils.simplifyString(ctx.ctrSet, value);
            if (exp.opType !== StringOpType.Const) {
                return ctx
                    .warnWithMsg(`from 'LibCall.builtins.str_islower': value is not a constant string`, source)
                    .toSet();
            }
            value = exp.value;
        }

        return ctx.toSetWith(SVBool.create(value === value.toLowerCase(), source));
    }

    export function str_startswith(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx
                .failWithMsg(
                    `from 'LibCall.builtins.str_startswith': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }

        const { heap } = ctx;
        const str = fetchAddr(params[0], heap);
        const tester = fetchAddr(params[1], heap);

        if (str?.type !== SVType.String || tester?.type !== SVType.String) {
            return ctx.warnWithMsg(`from 'LibCall.builtins.str_startswith': value is not a string`, source).toSet();
        }

        let value = str.value;
        if (typeof value !== 'string') {
            // TODO: calculate symbolic string
            const exp = ExpUtils.simplifyString(ctx.ctrSet, value);
            if (exp.opType !== StringOpType.Const) {
                return ctx
                    .warnWithMsg(`from 'LibCall.builtins.str_startswith': value is not a constant string`, source)
                    .toSet();
            }
            value = exp.value;
        }

        let testerVal = tester.value;
        if (typeof testerVal !== 'string') {
            // TODO: calculate symbolic string
            const exp = ExpUtils.simplifyString(ctx.ctrSet, testerVal);
            if (exp.opType !== StringOpType.Const) {
                return ctx
                    .warnWithMsg(`from 'LibCall.builtins.str_startswith': prefix is not a constant string`, source)
                    .toSet();
            }
            testerVal = exp.value;
        }

        return ctx.toSetWith(SVBool.create(value.startsWith(testerVal), source));
    }

    export function str_endswith(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx
                .failWithMsg(
                    `from 'LibCall.builtins.str_startswith': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }

        const { heap } = ctx;
        const str = fetchAddr(params[0], heap);
        const tester = fetchAddr(params[1], heap);

        if (str?.type !== SVType.String || tester?.type !== SVType.String) {
            return ctx.warnWithMsg(`from 'LibCall.builtins.str_endswith': value is not a string`, source).toSet();
        }

        let value = str.value;
        if (typeof value !== 'string') {
            // TODO: calculate symbolic string
            const exp = ExpUtils.simplifyString(ctx.ctrSet, value);
            if (exp.opType !== StringOpType.Const) {
                return ctx
                    .warnWithMsg(`from 'LibCall.builtins.str_endswith': value is not a constant string`, source)
                    .toSet();
            }
            value = exp.value;
        }

        let testerVal = tester.value;
        if (typeof testerVal !== 'string') {
            // TODO: calculate symbolic string
            const exp = ExpUtils.simplifyString(ctx.ctrSet, testerVal);
            if (exp.opType !== StringOpType.Const) {
                return ctx
                    .warnWithMsg(`from 'LibCall.builtins.str_endswith': prefix is not a constant string`, source)
                    .toSet();
            }
            testerVal = exp.value;
        }

        return ctx.toSetWith(SVBool.create(value.endsWith(testerVal), source));
    }

    export function has_key(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx
                .failWithMsg(
                    `from 'LibCall.builtins.has_key': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }

        const { heap } = ctx;
        const value = fetchAddr(params[0], heap);
        const key = fetchAddr(params[1], heap);

        if (value?.type !== SVType.Object) {
            return ctx.warnWithMsg(`from 'LibCall.builtins.has_key': value is not an object`, source).toSet();
        }

        if (key?.type !== SVType.String || typeof key.value !== 'string') {
            return ctx.warnWithMsg(`from 'LibCall.builtins.has_key': key is not a constant string`, source).toSet();
        }

        return ctx.toSetWith(SVBool.create(value.keyValues.has(key.value), source));
    }

    export function namedtuple_pushField(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx
                .failWithMsg(
                    `from 'LibCall.builtins.namedtuple_pushField': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }

        const { heap } = ctx;
        const nameList = fetchAddr(params[0], heap);
        const field = fetchAddr(params[1], heap);

        if (nameList?.type !== SVType.Object) {
            return ctx
                .warnWithMsg(`from 'LibCall.builtins.namedtuple_pushField': 'names' is not a list`, source)
                .toSet();
        }

        if (field?.type !== SVType.String || typeof field.value !== 'string') {
            return ctx
                .warnWithMsg(`from 'LibCall.builtins.namedtuple_pushField': field is not a constant string`, source)
                .toSet();
        }

        const len = fetchAddr(nameList.getAttr('$length'), heap);
        if (len?.type !== SVType.Int || typeof len.value !== 'number') {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.builtins.namedtuple_pushField': 'names' does not have constant length`,
                    source
                )
                .toSet();
        }

        let newName = nameList;
        const lenVal = len.value;
        const fields = field.value.split(/[ ,]/).filter((v) => v !== '');
        for (let i = 0; i < fields.length; i++) {
            newName = newName.setIndice(lenVal + i, SVString.create(fields[i], source));
        }

        newName = newName.setAttr('$length', SVInt.create(fields.length, source));

        const newHeap = heap.setVal(newName.addr, newName);

        return ctx.setHeap(newHeap).toSetWith(SVNone.create(undefined));
    }

    export function len(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 1) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.builtins.len': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }

        const { heap } = ctx;
        const value = fetchAddr(params[0], heap);

        if (value?.type === SVType.String) {
            const len = strLen(ctx, value.value, source);
            if (len instanceof Context) {
                return len.setRetVal(SVInt.create(len.retVal, source)).toSet();
            } else {
                return ctx.setRetVal(SVInt.create(len, source)).toSet();
            }
        } else if (value?.type !== SVType.Object) {
            return ctx.warnWithMsg(`from 'LibCall.builtins.len': value is not iterable`, source).toSet();
        }

        if (value.attrs.has('$length')) {
            return ctx.toSetWith(value.getAttr('$length')!);
        }

        return ctx.getAttrDeep(value, '__len__', source).flatMap((ctx) => {
            const lenFun = fetchAddr(ctx.retVal, ctx.heap);
            if (lenFun?.type === SVType.Func) {
                return TorchBackend.functionCall(ctx, lenFun, [], source);
            }

            return ctx.warnWithMsg(`from 'LibCall.builtins.len': object has no '__len__'`, source).toSet();
        });
    }

    // inclusive randint (a <= retVal <= b)
    export function randInt(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.builtins.randInt': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }

        const heap = ctx.heap;
        const [a, b, prefixAddr] = params;

        const aVal = fetchAddr(a, heap);
        const bVal = fetchAddr(b, heap);
        const prefix = (fetchAddr(prefixAddr, heap)! as SVString).value as string;

        // inject explicit variable range
        const varRangeMap = PyteaService.getVariableRange();
        const varRng = PyteaService.getVariableSeedRng(prefix);

        if (prefix in varRangeMap) {
            const range = varRangeMap[prefix];
            let num: ExpNumSymbol | number | undefined;

            if (varRng) {
                if (range === null) {
                    num = varRng.nextInt();
                } else if (typeof range === 'number') {
                    num = range;
                } else {
                    num = varRng.nextInt(range[0], range[1]);
                }

                return ctx.toSetWith(SVInt.create(num, source));
            }

            if (range === null) {
                num = ExpNum.fromSymbol(ctx.genSymInt(prefix, source));
                return ctx.toSetWith(SVInt.create(num, source));
            }

            if (typeof range === 'number') {
                return ctx.toSetWith(SVInt.create(range, source));
            } else {
                let symCtx: Context<unknown> = ctx;
                if (typeof range[0] === 'number') {
                    const numCtx = ctx.genIntGte(prefix, range[0], source);
                    symCtx = numCtx;
                    num = numCtx.retVal;
                    if (typeof range[1] === 'number') {
                        symCtx = symCtx.guarantee(symCtx.genLte(num, range[1], source));
                    }
                } else if (typeof range[1] === 'number') {
                    num = ExpNum.fromSymbol(ctx.genSymInt(prefix, source));
                    symCtx = symCtx.guarantee(symCtx.genLte(num, range[1], source));
                } else {
                    num = ExpNum.fromSymbol(ctx.genSymInt(prefix, source));
                }

                return symCtx.toSetWith(SVInt.create(num, source));
            }
        }

        if (!(aVal?.type === SVType.Int || aVal?.type === SVType.Float)) {
            return ctx.warnWithMsg(`from 'LibCall.builtins.randInt: value a is non-numeric`, source).toSet();
        }
        if (!(bVal?.type === SVType.Int || bVal?.type === SVType.Float)) {
            return ctx.warnWithMsg(`from 'LibCall.builtins.randInt: value b is non-numeric`, source).toSet();
        }

        if (varRng && typeof aVal.value === 'number' && typeof bVal.value === 'number') {
            const num = varRng.nextInt(aVal.value, bVal.value);
            return ctx.toSetWith(SVInt.create(num, source));
        }

        if (aVal.value === bVal.value) {
            return ctx.toSetWith(SVInt.create(aVal.value, source));
        }

        return ctx
            .require(
                ctx.genLte(aVal.value, bVal.value, source),
                `from 'LibCall.builtins.randInt': min value is greater than max value.`,
                source
            )
            .flatMap((ctx) => {
                let symCtx = ctx.genIntGte(prefix, aVal.value, source);
                const num = symCtx.retVal;
                symCtx = symCtx.guarantee(symCtx.genLte(num, bVal.value, source));

                return symCtx.toSetWith(SVInt.create(num, source));
            });
    }

    // exclusive randfloat (a <= retVal < b)
    export function randFloat(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.builtins.randFloat': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }

        const heap = ctx.heap;
        const [a, b, prefixAddr] = params;

        const aVal = fetchAddr(a, heap);
        const bVal = fetchAddr(b, heap);
        const prefix = (fetchAddr(prefixAddr, heap)! as SVString).value as string;

        // inject explicit variable range (inclusive)
        const varRangeMap = PyteaService.getVariableRange();
        const varRng = PyteaService.getVariableSeedRng(prefix);

        if (prefix in varRangeMap) {
            const range = varRangeMap[prefix];
            let num: ExpNumSymbol | number | undefined;

            if (varRng) {
                if (range === null) {
                    num = varRng.nextFloat();
                } else if (typeof range === 'number') {
                    num = range;
                } else {
                    num = varRng.nextFloat(range[0], range[1]);
                }

                return ctx.toSetWith(SVFloat.create(num, source));
            }

            if (range === null) {
                num = ExpNum.fromSymbol(ctx.genSymFloat(prefix, source));
                return ctx.toSetWith(SVFloat.create(num, source));
            }

            if (typeof range === 'number') {
                return ctx.toSetWith(SVFloat.create(range, source));
            } else {
                let symCtx: Context<unknown> = ctx;
                if (typeof range[0] === 'number') {
                    const numCtx = ctx.genFloatGte(prefix, range[0], source);
                    symCtx = numCtx;
                    num = numCtx.retVal;
                    if (typeof range[1] === 'number') {
                        symCtx = symCtx.guarantee(symCtx.genLte(num, range[1], source));
                    }
                } else if (typeof range[1] === 'number') {
                    num = ExpNum.fromSymbol(ctx.genSymFloat(prefix, source));
                    symCtx = symCtx.guarantee(symCtx.genLte(num, range[1], source));
                } else {
                    num = ExpNum.fromSymbol(ctx.genSymFloat(prefix, source));
                }

                return symCtx.toSetWith(SVFloat.create(num, source));
            }
        }

        if (!(aVal?.type === SVType.Int || aVal?.type === SVType.Float)) {
            return ctx.warnWithMsg(`from 'LibCall.builtins.randFloat: value a is non-numeric`, source).toSet();
        }
        if (!(bVal?.type === SVType.Int || bVal?.type === SVType.Float)) {
            return ctx.warnWithMsg(`from 'LibCall.builtins.randFloat: value b is non-numeric`, source).toSet();
        }

        if (varRng && typeof aVal.value === 'number' && typeof bVal.value === 'number') {
            const num = varRng.nextFloat(aVal.value, bVal.value);
            return ctx.toSetWith(SVFloat.create(num, source));
        }

        if (aVal.value === bVal.value) {
            return ctx.toSetWith(SVFloat.create(aVal.value, source));
        }

        return ctx
            .require(
                ctx.genLte(aVal.value, bVal.value, source),
                `from 'LibCall.builtins.randFloat': min value is greater than max value.`,
                source
            )
            .flatMap((ctx) => {
                let symCtx = ctx.genFloatGte(prefix, aVal.value, source);
                const num = symCtx.retVal;
                symCtx = symCtx.guarantee(symCtx.genLt(num, bVal.value, source));

                return symCtx.toSetWith(SVFloat.create(num, source));
            });
    }

    export function exit(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        return ctx.failWithMsg('explicit process exit function call', source).toSet();
    }

    export function warn(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const warnObj = fetchAddr(ctx.retVal.params[0], ctx.heap);
        const warnMsg =
            warnObj?.type === SVType.String && typeof warnObj.value === 'string'
                ? warnObj.value
                : 'Explicit warn called';
        return ctx.warnWithMsg(warnMsg, source).toSet();
    }

    // explicit setIndice by value
    export function setIndice(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.builtins.setIndice': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }
        const heap = ctx.heap;
        const [objAddr, indiceAddr, value] = params;

        const obj = fetchAddr(objAddr, heap);
        const indice = fetchAddr(indiceAddr, heap);

        if (indice?.type !== SVType.Int || typeof indice.value !== 'number') {
            return ctx.warnWithMsg(`from 'LibCall.builtins.setIndice': index is not a constant`, source).toSet();
        }

        if (obj?.type !== SVType.Object) {
            return ctx.warnWithMsg(`from 'LibCall.builtins.setIndice': got non-object`, source).toSet();
        }

        return ctx.setHeap(heap.setVal(obj.addr, obj.setIndice(indice.value, value))).toSetWith(SVNone.create(source));
    }

    // wrapper of getItemByIndexExpNum
    export function getItemByIndex(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.builtins.getItemByIndex': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }
        const heap = ctx.heap;
        const [objAddr, indiceAddr] = params;

        const obj = fetchAddr(objAddr, heap);
        const indice = fetchAddr(indiceAddr, heap);

        if (indice?.type !== SVType.Int) {
            return ctx.warnWithMsg(`from 'LibCall.builtins.getItemByIndex': index is not an integer`, source).toSet();
        }

        if (obj?.type !== SVType.Object) {
            return ctx.warnWithMsg(`from 'LibCall.builtins.getItemByIndex': got non-object`, source).toSet();
        }

        const item = TorchBackend.getItemByIndex(ctx, obj, indice.value, source);
        if (!item) {
            const indiceStr = SymExp.toString(indice.value);
            let lengthStr = 'unknown';
            const objLen = obj.getAttr('$length');
            if (objLen && objLen.type === SVType.Int) {
                lengthStr = SymExp.toString(objLen.value);
            }
            return ctx.failWithMsg(`index out of range (indice: ${indiceStr}, length: ${lengthStr})`, source).toSet();
        } else if (item.type === SVType.Error) {
            switch (item.level) {
                case SVErrorLevel.Error:
                    return ctx.failWithMsg(item.reason, source).toSet();
                case SVErrorLevel.Warning:
                    return ctx.warnWithMsg(item.reason, source).toSet();
                case SVErrorLevel.Log:
                    return ctx.addLog(item.reason, source).toSetWith(item);
            }
        }

        return ctx.toSetWith(item);
    }

    // clone object (shallow-clone)
    export function clone(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 1) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.builtins.clone': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }
        const heap = ctx.heap;
        const [objAddr] = params;

        const obj = fetchAddr(objAddr, heap);

        if (obj?.type === SVType.Object) {
            const [newAddr, heap2] = heap.malloc(source);
            const newObj = obj.set('addr', newAddr);
            const heap3 = heap2.setVal(newAddr, newObj);
            return ctx.setHeap(heap3).toSetWith(newAddr);
        }

        return ctx.toSetWith(objAddr);
    }

    // explicit getAttr by value
    export function getAttr(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.builtins.getAttr': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }
        const heap = ctx.heap;
        const [objAddr, attrAddr] = params;

        const obj = fetchAddr(objAddr, heap);
        const attr = fetchAddr(attrAddr, heap);

        if (attr?.type !== SVType.String || typeof attr.value !== 'string') {
            return ctx.warnWithMsg(`from 'LibCall.builtins.getAttr': attribute name is not a constant`, source).toSet();
        }

        if (obj?.type !== SVType.Object) {
            return ctx.warnWithMsg(`from 'LibCall.builtins.getAttr': got non-object`, source).toSet();
        }

        const value = obj.getAttr(attr.value);
        if (!value) {
            return ctx
                .warnWithMsg(`from 'LibCall.builtins.getAttr': ${attr.value} is not in object. return warning.`, source)
                .toSet();
        }

        return ctx.toSetWith(value);
    }

    // explicit setAttr by value
    export function setAttr(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.builtins.setAttr': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }
        const heap = ctx.heap;
        const [objAddr, attrAddr, value] = params;

        const obj = fetchAddr(objAddr, heap);
        const attr = fetchAddr(attrAddr, heap);

        if (attr?.type !== SVType.String || typeof attr.value !== 'string') {
            return ctx.warnWithMsg(`from 'LibCall.builtins.setAttr': attribute name is not a constant`, source).toSet();
        }

        if (obj?.type !== SVType.Object) {
            return ctx.warnWithMsg(`from 'LibCall.builtins.setAttr': got non-object`, source).toSet();
        }

        return ctx.setHeap(heap.setVal(obj.addr, obj.setAttr(attr.value, value))).toSetWith(SVNone.create(source));
    }

    export function callable(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 1) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.builtins.callable': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }
        const heap = ctx.heap;
        const value = fetchAddr(params[0], heap);
        if (!value || value.type === SVType.Error) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.builtins.callable': checking callable of unknown value. propagates it.`,
                    source
                )
                .toSet();
        }

        if (value.type === SVType.Func) {
            return ctx.setRetVal(SVBool.create(true, source)).toSet();
        }

        if (value.type !== SVType.Object) {
            return ctx.setRetVal(SVBool.create(false, source)).toSet();
        }

        const call = fetchAddr(value.getAttr('__call__'), ctx.heap);
        if (call?.type === SVType.Error) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.builtins.callable': checking callable of unknown __call__ value. propagates it.`,
                    source
                )
                .toSet();
        }

        return ctx.setRetVal(SVBool.create(call?.type === SVType.Func, source)).toSet();
    }

    export function time(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        return ctx.toSetWith(SVFloat.create(Date.now() / 1000.0, source));
    }

    // IMPORTANT API:
    // boxed lengthed list is not unfoled by for loop
    // it prevents constant unfolding of torch DataLoader
    // to prevent boxing, set 'boxDataLoader' option to false in 'pyteaconfig.json'
    export function box(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 1) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.builtins.box': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }

        const valueAddr = params[0];

        if (!PyteaService.isBoxAllowed()) {
            return ctx.toSetWith(valueAddr);
        }

        const value = fetchAddr(valueAddr, ctx.heap);

        if (value?.type === SVType.Int) {
            const range = ctx.getCachedRange(value.value);
            if (range?.isConst()) {
                const boxed = SVInt.create(ExpNum.box(range.start, source), source);
                return ctx.toSetWith(boxed);
            }
        }

        return ctx.toSetWith(valueAddr);
    }

    // Debug probe for breakpoint in TS
    export function debug(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        const { env, heap, retVal, ctrSet, callStack, logs, relPath } = ctx;
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        const position = formatCodeSource(source);

        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        const params = ctx.retVal.params;
        const fetch = fetchAddr;
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        const nameof = (name: string) => fetch(env.getId(name), heap);
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        const addrof = (addr?: SVAddr) => fetch(addr, heap);
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        const vs = ShValue.toString;
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        const es = SymExp.toString;
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        const util = ExpUtils;

        // set breakpoint here.
        return ctx.toSetWith(SVNone.create(source));
    }

    export const libCallImpls: { [key: string]: LCImpl } = {
        superGetAttr,
        isinstance,
        cast,
        list_append,
        dict_items,
        dict_keys,
        dict_values,
        dict_getitem,
        dict_setitem,
        dict_pop,
        str_islower,
        str_startswith,
        str_endswith,
        namedtuple_pushField,
        has_key,
        len,
        randInt,
        randFloat,
        exit,
        warn,
        clone,
        getAttr,
        setAttr,
        setIndice,
        getItemByIndex,
        callable,
        box,
        time,
        debug,
    };
}

export const libCallMap: Map<string, LCImpl> = new Map([...Object.entries(BuiltinsLCImpl.libCallImpls)]);
