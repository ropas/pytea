import { fetchAddr, sanitizeAddr, trackMro } from '../backend/backUtils';
import { Context, ContextSet } from '../backend/context';
import { isInstanceOf, strLen } from '../backend/expUtils';
import {
    CodeSource,
    PrimitiveType,
    ShValue,
    SVAddr,
    SVBool,
    SVFloat,
    SVInt,
    SVNone,
    SVNotImpl,
    SVObject,
    SVSize,
    SVString,
    SVType,
} from '../backend/sharpValues';
import { ExpNum, ExpNumSymbol, NumBopType, NumUopType } from '../backend/symExpressions';
import { TorchBackend } from '../backend/torchBackend';
import { PyteaService } from '../service/pyteaService';
import { LCImpl } from '.';
import { LCBase } from './libcall';

export namespace BuiltinsLCImpl {
    export function superGetAttr(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
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

        const superClass = fetchAddr(SVAddr.create(mro[1]), heap);
        if (superClass?.type !== SVType.Object) {
            return ctx.warnWithMsg(`from 'LibCall.builtins.superGetAttr': superclass is not an object`, source).toSet();
        }

        return TorchBackend.getAttrDeep(ctx, superClass, attr.value).map((ctx) => {
            const retVal = ctx.retVal;
            if (retVal.type === SVType.Func && self.type === SVType.Addr) {
                return ctx.setRetVal(retVal.bound(self));
            }
            return ctx;
        });
    }

    export function isinstance(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
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

    export function cast(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
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
                    return ctx.setRetVal(SVNotImpl.create('not implemented', source)).toSet();
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
            case PrimitiveType.Float:
            case PrimitiveType.Str:
            case PrimitiveType.Bool:
            case PrimitiveType.Dict:
            case PrimitiveType.Set:
                break;
        }

        return ctx.setRetVal(SVNotImpl.create('not implemented', source)).toSet();
    }

    export function list_append(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
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
            if (list?.type !== SVType.Object) {
                console.log(params[0]?.type);
                console.log(list?.type);
                console.log(SVType.Object);
            }
            if (!value) console.log('two');
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

    export function dict_items(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
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

        let [pairList, pairListAddr, newHeap] = SVObject.create(heap, source);
        let pairListLen = 0;
        let pair, pairAddr;
        for (const [key, value] of dict.keyValues) {
            [pair, pairAddr, newHeap] = SVObject.create(newHeap, source);
            pair = pair.setAttr('$length', SVInt.create(2, source));
            pair = pair.setIndice(0, SVString.create(key, source));
            pair = pair.setIndice(1, value);
            pairList = pairList.setIndice(pairListLen, pair);
            pairListLen++;
        }
        pairList = pairList.setAttr('$length', SVInt.create(pairListLen, source));

        return ctx.setHeap(newHeap.setVal(pairListAddr, pairList)).toSetWith(pairListAddr);
    }

    // TODO: fix this to support non-string typed key
    export function dict_setitem(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
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
        const value = sanitizeAddr(params[2], heap);

        if (dict?.type !== SVType.Object || key?.type !== SVType.String || !value) {
            // currently, only supports string typed key
            return ctx.warnWithMsg(`from 'LibCall.builtins.dict_setitem': invalid value type`, source).toSet();
        }
        if (typeof key.value !== 'string') {
            return ctx
                .warnWithMsg(`from 'LibCall.builtins.dict_setitem': does not supports symbolic string`, source)
                .toSet();
        }

        const newDict = dict.setKeyVal(key.value, value);

        return ctx.setHeap(heap.setVal(dict.addr, newDict)).toSetWith(SVNone.create(source));
    }

    // TODO: fix this to support non-string typed key
    export function dict_getitem(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
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

        if (dict?.type !== SVType.Object || key?.type !== SVType.String) {
            // currently, only supports string typed key
            return ctx.warnWithMsg(`from 'LibCall.builtins.dict_getitem': invalid value type`, source).toSet();
        }
        if (typeof key.value !== 'string') {
            return ctx
                .warnWithMsg(`from 'LibCall.builtins.dict_getitem': does not supports symbolic string`, source)
                .toSet();
        }

        const value = dict.getKeyVal(key.value);
        if (value === undefined) {
            return ctx.failWithMsg(`LibCall.builtins.dict_getitem': invalid key value`, source).toSet();
        }

        return ctx.toSetWith(value);
    }

    // TODO: fix this to support non-string typed key
    export function dict_pop(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
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

    export function len(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
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
            const len = strLen(ctx, value.value);
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
    export function randInt(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
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
        if (prefix in varRangeMap) {
            const range = varRangeMap[prefix];
            let num: ExpNumSymbol | undefined;

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

        let symCtx = ctx.genIntGte(prefix, aVal.value, source);
        const num = symCtx.retVal;
        symCtx = symCtx.guarantee(symCtx.genLte(num, bVal.value, source));

        return symCtx.toSetWith(SVInt.create(num, source));
    }

    // exclusive randfloat (a <= retVal < b)
    export function randFloat(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
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
        if (prefix in varRangeMap) {
            const range = varRangeMap[prefix];
            let num: ExpNumSymbol | undefined;

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

        let symCtx = ctx.genFloatGte(prefix, aVal.value, source);
        const num = symCtx.retVal;
        symCtx = symCtx.guarantee(symCtx.genLt(num, bVal.value, source));

        return symCtx.toSetWith(SVFloat.create(num, source));
    }

    // get `(objectAddr, size)`, set object to SVSize with shape `size`
    export function setSize(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.builtins.setSize': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }
        const heap = ctx.heap;
        const [objAddr, sizeAddr] = params;
        const obj = fetchAddr(objAddr, heap);

        if (objAddr.type !== SVType.Addr || obj?.type !== SVType.Object) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.builtins.setSize': not an object type:\n\t${objAddr.toString()} -> ${obj?.toString()}`,
                    source
                )
                .toSet();
        }

        return ctx.parseSize(sizeAddr, source).map((ctx) => {
            const size = ctx.retVal;
            if (typeof size === 'string') {
                return ctx.warnWithMsg(size, source) as Context<ShValue>;
            }

            const sizeObj = SVSize.fromObject(ctx, obj, size);
            return ctx.setHeap(ctx.heap.setVal(objAddr, sizeObj)).setRetVal(sizeObj);
        });
    }

    export function exit(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
        return ctx.failWithMsg('explicit process exit function call', source).toSet();
    }

    export function warn(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
        const warnObj = fetchAddr(ctx.retVal.params[0], ctx.heap);
        const warnMsg =
            warnObj?.type === SVType.String && typeof warnObj.value === 'string'
                ? warnObj.value
                : 'Explicit warn called';
        return ctx.warnWithMsg(warnMsg, source).toSet();
    }

    // explicit setIndice by value
    export function setIndice(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
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
            return ctx.warnWithMsg(`from 'LibCall.builtins.setIndice': attribute is not a constant`, source).toSet();
        }

        if (obj?.type !== SVType.Object) {
            return ctx.warnWithMsg(`from 'LibCall.builtins.setIndice': got non-object`, source).toSet();
        }

        return ctx.setHeap(heap.setVal(obj.addr, obj.setIndice(indice.value, value))).toSetWith(SVNone.create(source));
    }

    // explicit setAttr by value
    export function setAttr(ctx: Context<LCBase.ExplicitParams>, source?: CodeSource): ContextSet<ShValue> {
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
            return ctx.warnWithMsg(`from 'LibCall.builtins.setAttr': attribute is not a constant`, source).toSet();
        }

        if (obj?.type !== SVType.Object) {
            return ctx.warnWithMsg(`from 'LibCall.builtins.setAttr': got non-object`, source).toSet();
        }

        return ctx.setHeap(heap.setVal(obj.addr, obj.setAttr(attr.value, value))).toSetWith(SVNone.create(source));
    }

    export const libCallImpls: { [key: string]: LCImpl } = {
        superGetAttr,
        isinstance,
        cast,
        list_append,
        dict_items,
        dict_getitem,
        dict_setitem,
        dict_pop,
        len,
        randInt,
        randFloat,
        setSize,
        exit,
        warn,
        setAttr,
        setIndice,
    };
}

export const libCallMap: Map<string, LCImpl> = new Map([...Object.entries(BuiltinsLCImpl.libCallImpls)]);
