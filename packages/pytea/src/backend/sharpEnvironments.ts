/*
 * sharpEnvironments.ts
 * Copyright (c) Seoul National University.
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo (mersshs@gmail.com)
 *
 * Environment and heaps for static semantics of PyTea Internal Representation
 */

import { Map, Record } from 'immutable';

import { CodeSource, ShValue, SVAddr, SVType } from './sharpValues';

interface ShEnvProps {
    readonly addrMap: Map<string, SVAddr>; // negative address is builtin values
}

const shEnvDefaults: ShEnvProps = {
    addrMap: Map(),
};

export class ShEnv extends Record(shEnvDefaults) implements ShEnvProps {
    constructor(addrMap?: Map<string, SVAddr>) {
        addrMap ? super({ addrMap }) : super();
    }

    hasId(id: string): boolean {
        return this.addrMap.has(id);
    }

    getId(id: string): SVAddr | undefined {
        return this.addrMap.get(id);
    }

    setId(id: string, addr: SVAddr): ShEnv {
        return this.set('addrMap', this.addrMap.set(id, addr));
    }

    removeId(id: string): ShEnv {
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
    addOffset(offset: number): ShEnv {
        return new ShEnv(this.addrMap.mapEntries(([k, addr]) => [k, addr.addOffset(offset)]));
    }

    mergeAddr(env: ShEnv): ShEnv {
        return new ShEnv(this.addrMap.merge(env.addrMap));
    }
}

interface ShHeapProps {
    readonly addrMax: number;
    readonly valMap: Map<number, ShValue>; // negative address is builtin values
}

const shHeapDefaults: ShHeapProps = {
    addrMax: 0,
    valMap: Map(),
};

export class ShHeap extends Record(shHeapDefaults) implements ShHeapProps {
    constructor() {
        super();
    }

    getVal(addr: number | SVAddr): ShValue | undefined {
        if (typeof addr !== 'number') {
            addr = addr.addr;
        }
        return this.valMap.get(addr);
    }

    getValRecur(addr: number | SVAddr): ShValue | undefined {
        let retVal = this.getVal(addr);
        while (retVal && retVal.type === SVType.Addr) {
            retVal = this.getVal(retVal);
        }

        return retVal;
    }

    setVal(addr: number | SVAddr, value: ShValue): ShHeap {
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
    malloc(source: CodeSource | undefined): [SVAddr, ShHeap] {
        const addr = this.addrMax;
        return [SVAddr.create(addr + 1, source), this.set('addrMax', addr + 1)];
    }

    // malloc with value assignment
    allocNew(value: ShValue, source: CodeSource | undefined): [SVAddr, ShHeap] {
        const [addr, heap] = this.malloc(source);
        return [addr, heap.setVal(addr, value)];
    }

    toString(): string {
        const keyArr = [...this.valMap.keys()];
        keyArr.sort((a, b) => a - b);
        return `{\n${keyArr.map((i) => `  ${i} => ${this.valMap.get(i)?.toString()},`).join('\n')}\n}`;
    }

    free(addr: number | SVAddr): ShHeap {
        if (typeof addr !== 'number') {
            addr = addr.addr;
        }

        let newHeap = this.set('valMap', this.valMap.remove(addr));

        if (addr === this.addrMax) {
            newHeap = this.set('addrMax', addr - 1);
        }

        return newHeap;
    }

    // add offset to every address in values on the heap.
    addOffset(offset: number): ShHeap {
        return this.set('addrMax', this.addrMax + offset).set(
            'valMap',
            this.valMap.mapEntries(([k, v]) => [k >= 0 ? k + offset : k, ShValue.addOffset(v, offset)])
        );
    }

    filter(predicate: (value: ShValue, key: number) => boolean) {
        return this.set('valMap', this.valMap.filter(predicate));
    }

    // Garbage Collection (env as GC Root); use simple mark and sweep
    // CAVEAT: DO NOT run GC while running interpreter! If does, some local or global values will be lost!
    _runGC(envList: ShEnv[]): ShHeap {
        // use JS in-place Set, not a that of immutable.js
        const marked: Set<number> = new Set();
        let heap: ShHeap = this;

        function markVal(value: ShValue): void {
            switch (value.type) {
                case SVType.Addr:
                    mark(value);
                    break;
                case SVType.Object:
                    value.attrs.forEach((v) => markVal(v));
                    value.indices.forEach((v) => markVal(v));
                    value.keyValues.forEach((v) => markVal(v));
                    break;
                case SVType.Func:
                    value.defaults.forEach((v) => markVal(v));
                    value.funcEnv?.addrMap.forEach((addr) => mark(addr));
                    break;
                default:
                    break;
            }
        }

        function mark(addr: number | SVAddr): void {
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

        envList.forEach((env) => {
            env.addrMap.forEach((addr) => mark(addr));
        });

        heap = heap.set(
            'valMap',
            heap.valMap.filter((_, k) => k < 0 || marked.has(k))
        );

        return heap;
    }
}
