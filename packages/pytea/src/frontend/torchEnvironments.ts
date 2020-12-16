/*
 * torchEnvironments.ts
 * Copyright (c) Seoul National University.
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo
 *
 * Environment and heaps for dynamic semantics of PyTea internal languages.
 */

import { Map, Record } from 'immutable';

import { ParseNode } from '../parser/parseNodes';
import { TorchInterpreter } from './torchInterpreter';
import { ThStmt } from './torchStatements';
import { ThValue, TVAddr, TVType } from './torchValues';

interface ThEnvProps {
    readonly addrMap: Map<string, TVAddr>; // negative address is builtin values
}

const thEnvDefaults: ThEnvProps = {
    addrMap: Map(),
};

export class ThEnv extends Record(thEnvDefaults) implements ThEnvProps {
    constructor(addrMap?: Map<string, TVAddr>) {
        addrMap ? super({ addrMap }) : super();
    }

    // return TVUndef if id is not in addrMap
    getId(id: string): TVAddr | undefined {
        return this.addrMap.get(id);
    }

    setId(id: string, addr: TVAddr): ThEnv {
        return this.set('addrMap', this.addrMap.set(id, addr));
    }

    removeId(id: string): ThEnv {
        return this.set('addrMap', this.addrMap.delete(id));
    }

    toString(): string {
        let revMap: Map<number, string> = Map();
        this.addrMap.forEach((v, k) => {
            revMap = revMap.set(v.addr, k);
        });
        const revArr = [...revMap.keys()];
        revArr.sort((a, b) => a - b);
        return `{\n${revArr.map((i) => `  ${revMap.get(i)} => ${i},`).join('\n')}\n}`;
    }

    // add offset to every address
    addOffset(offset: number): ThEnv {
        return new ThEnv(this.addrMap.mapEntries(([k, addr]) => [k, addr.addOffset(offset)]));
    }
}

interface ThHeapProps {
    readonly addrMax: number;
    readonly valMap: Map<number, ThValue>; // negative address is builtin values
}

const thHeapDefaults: ThHeapProps = {
    addrMax: 0,
    valMap: Map(),
};

export class ThHeap extends Record(thHeapDefaults) implements ThHeapProps {
    constructor() {
        super();
    }

    addNew(value: ThValue): ThHeap {
        const addr = this.addrMax;
        return this.set('addrMax', addr + 1).set('valMap', this.valMap.set(addr, value));
    }

    // return TVUndef if addr is not in valMap
    getVal(addr: number | TVAddr): ThValue | undefined {
        if (typeof addr !== 'number') {
            addr = addr.addr;
        }
        return this.valMap.get(addr);
    }

    setVal(addr: number | TVAddr, value: ThValue): ThHeap {
        if (typeof addr !== 'number') {
            addr = addr.addr;
        }
        let heap = this;
        if (heap.addrMax < addr) {
            heap = heap.set('addrMax', addr);
        }
        return heap.set('valMap', heap.valMap.set(addr, value));
    }

    // return new address and malloced heap
    malloc(source?: ParseNode): [TVAddr, ThHeap] {
        const addr = this.addrMax;
        return [TVAddr.create(addr + 1, source), this.set('addrMax', addr + 1)];
    }

    // malloc with value assignment
    allocNew(value: ThValue, source?: ParseNode): [TVAddr, ThHeap] {
        const [addr, heap] = this.malloc(source);
        return [addr, heap.setVal(addr, value)];
    }

    toString(): string {
        const keyArr = [...this.valMap.keys()];
        keyArr.sort((a, b) => a - b);
        return `{\n${keyArr.map((i) => `  ${i} => ${this.valMap.get(i)?.toString()},`).join('\n')}\n}`;
    }

    free(addr: number | TVAddr): ThHeap {
        if (typeof addr !== 'number') {
            addr = addr.addr;
        }
        return this.set('valMap', this.valMap.remove(addr));
    }

    // add offset to every address in values on the heap.
    addOffset(offset: number): ThHeap {
        return this.set('addrMax', this.addrMax + offset).set(
            'valMap',
            this.valMap.mapEntries(([k, v]) => [k >= 0 ? k + offset : k, ThValue.addOffset(v, offset)])
        );
    }

    // Garbage Collection (env as GC Root); use simple mark and sweep
    // CAVEAT: DO NOT run GC while running interpreter! If does, some local or global values will be lost!
    _runGC(env: ThEnv): ThHeap {
        // use JS in-place Set, not a that of immutable.js
        const marked: Set<number> = new Set();
        let heap: ThHeap = this;

        function markVal(value: ThValue): void {
            value.attrs.forEach((v) => markVal(v));

            switch (value.type) {
                case TVType.Addr:
                    mark(value);
                    break;
                case TVType.Object:
                    value.indices.forEach((v) => markVal(v));
                    value.keyValues.forEach((v) => markVal(v));
                    break;
                case TVType.Func:
                    value.defaults.forEach((v) => markVal(v));
                    value.funcEnv?.addrMap.forEach((addr) => mark(addr));
                    break;
                default:
                    break;
            }
        }

        function mark(addr: number | TVAddr): void {
            if (typeof addr !== 'number') {
                addr = addr.addr;
            }

            if (marked.has(addr)) {
                return;
            } else {
                marked.add(addr);
            }

            const val = heap.getVal(addr);
            if (!val) {
                return;
            }

            markVal(val);
        }

        env.addrMap.forEach((addr) => mark(addr));

        heap = heap.set(
            'valMap',
            heap.valMap.filter((_, k) => k < 0 || marked.has(k))
        );

        return heap;
    }
}

export function mergeEnvHeap(base: [ThEnv, ThHeap], merged: [ThEnv, ThHeap]): [ThEnv, ThHeap] {
    const [env1, heap1] = base;
    const [env2, heap2] = merged;

    const offset = heap1.addrMax;

    const newEnv = env1.set('addrMap', env1.addrMap.merge(env2.addOffset(offset).addrMap));
    const newHeap = heap1
        .set('addrMax', heap2.addrMax + offset)
        .set('valMap', heap1.valMap.merge(heap2.addOffset(offset).valMap));

    return [newEnv, newHeap];
}

let _defaultEnv: ThEnv;
let _defaultHeap: ThHeap;
export function registDefault(stmt: ThStmt): void {
    const builtinCtx = TorchInterpreter.runEmpty(stmt);
    _defaultEnv = builtinCtx[0];
    _defaultHeap = builtinCtx[1];

    const offset = -_defaultHeap.addrMax - 1;
    _defaultEnv = _defaultEnv.addOffset(offset);
    _defaultHeap = _defaultHeap.addOffset(offset);
}

// make assign builtin values and functions, and assign them to negative addresses
export function defaultEnvHeap(): [ThEnv, ThHeap] {
    return [_defaultEnv, _defaultHeap];
}
