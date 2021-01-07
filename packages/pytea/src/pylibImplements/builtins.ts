import { ParseNode } from 'pyright-internal/parser/parseNodes';

import { fetchAddr, sanitizeAddr, trackMro } from '../backend/backUtils';
import { Context, ContextSet } from '../backend/context';
import { isInstanceOf, simplifyString, strLen } from '../backend/expUtils';
import {
    PrimitiveType,
    ShValue,
    SVAddr,
    SVBool,
    SVFloat,
    SVInt,
    SVNotImpl,
    SVSize,
    SVType,
} from '../backend/sharpValues';
import { ExpNum, NumBopType, NumUopType } from '../backend/symExpressions';
import { TorchBackend } from '../backend/torchBackend';
import { LCImpl } from '.';
import { LCBase } from './libcall';

export namespace BuiltinsLCImpl {
    export function superGetAttr(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
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

    export function isinstance(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
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

    export function cast(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx
                .failWithMsg(
                    `from 'LibCall.builtins.cast': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }

        const { heap } = ctx;
        const value = fetchAddr(params[0], heap);
        const type = fetchAddr(params[1], heap);
        // const kwargs = fetchAddr(params[2], heap)

        if (!value || type?.type !== SVType.Int || typeof type.value !== 'number') {
            return ctx.failWithMsg(`from 'LibCall.builtins.cast': invalid value type`, source).toSet();
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
            case PrimitiveType.Tuple:
            case PrimitiveType.List:
            case PrimitiveType.Float:
            case PrimitiveType.Str:
            case PrimitiveType.Bool:
            case PrimitiveType.Dict:
            case PrimitiveType.Set:
                break;
        }

        return ctx.setRetVal(SVNotImpl.create('not implemented', source)).toSet();
    }

    export function list_append(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
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

    export function len(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
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
    export function randInt(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.builtins.randInt': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }

        const heap = ctx.heap;
        const [a, b] = params;

        const aVal = fetchAddr(a, heap);
        const bVal = fetchAddr(b, heap);

        if (!(aVal?.type === SVType.Int || aVal?.type === SVType.Float)) {
            return ctx.warnWithMsg(`from 'LibCall.builtins.randInt: value a is non-numeric`, source).toSet();
        }
        if (!(bVal?.type === SVType.Int || bVal?.type === SVType.Float)) {
            return ctx.warnWithMsg(`from 'LibCall.builtins.randInt: value b is non-numeric`, source).toSet();
        }

        let symCtx = ctx.genIntGte('randInt', aVal.value, source);
        const num = symCtx.retVal;
        symCtx = symCtx.guarantee(symCtx.genLte(num, bVal.value, source));

        return symCtx.toSetWith(SVInt.create(num, source));
    }

    // exclusive randfloat (a <= retVal < b)
    export function randFloat(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx
                .warnWithMsg(
                    `from 'LibCall.builtins.randFloat': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }

        const heap = ctx.heap;
        const [a, b] = params;

        const aVal = fetchAddr(a, heap);
        const bVal = fetchAddr(b, heap);

        if (!(aVal?.type === SVType.Int || aVal?.type === SVType.Float)) {
            return ctx.warnWithMsg(`from 'LibCall.builtins.randFloat: value a is non-numeric`, source).toSet();
        }
        if (!(bVal?.type === SVType.Int || bVal?.type === SVType.Float)) {
            return ctx.warnWithMsg(`from 'LibCall.builtins.randFloat: value b is non-numeric`, source).toSet();
        }

        let symCtx = ctx.genFloatGte('randFloat', aVal.value, source);
        const num = symCtx.retVal;
        symCtx = symCtx.guarantee(symCtx.genLt(num, bVal.value, source));

        return symCtx.toSetWith(SVFloat.create(num, source));
    }

    // get `(objectAddr, size)`, set object to SVSize with shape `size`
    export function setSize(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
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

    export function exit(ctx: Context<LCBase.ExplicitParams>, source?: ParseNode): ContextSet<ShValue> {
        return ctx.failWithMsg('explicit exit function call', source).toSet();
    }

    export const libCallImpls: { [key: string]: LCImpl } = {
        superGetAttr,
        isinstance,
        cast,
        list_append,
        len,
        randInt,
        randFloat,
        setSize,
        exit,
    };
}

export const libCallMap: Map<string, LCImpl> = new Map([...Object.entries(BuiltinsLCImpl.libCallImpls)]);
