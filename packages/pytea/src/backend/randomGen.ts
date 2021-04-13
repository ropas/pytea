/*
 * randomGen.ts
 * Copyright (c) Seoul National University.
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo (mersshs@gmail.com)
 *
 * Simple Seedable Random Number Generator
 * (based on fanatid/xorshift.js)
 */

export class RandomGen {
    private _rng: XorShift128Plus;

    // range is inclusive
    constructor(seed: number | null | undefined) {
        this._rng = new XorShift128Plus(seed);
    }

    nextInt(from?: number | null, to?: number | null): number {
        if (from === undefined || from === null) {
            if (to !== undefined && to !== null) {
                from = to - 10000;
            } else {
                from = 1;
            }
        }
        if (to === undefined || to === null) {
            if (from !== undefined && from !== null) {
                to = from + 10000;
            } else {
                to = 10000;
            }
        }

        let a = Math.floor(from);
        const b = Math.floor(to) + 1;
        if (a < from) {
            a = a + 1;
        }
        return Math.floor((b - a) * this._rng.random() + a);
    }

    nextFloat(from?: number | null, to?: number | null): number {
        if (from === undefined || from === null) {
            if (to !== undefined && to !== null) {
                from = to - 10000;
            } else {
                from = 1;
            }
        }
        if (to === undefined || to === null) {
            if (from !== undefined && from !== null) {
                to = from + 10000;
            } else {
                to = 10000;
            }
        }
        return (from - to) * this._rng.random() + to;
    }

    reset(): void {
        this._rng.reset();
    }
}

class XorShift128Plus {
    private _s0H: number;
    private _s0L: number;
    private _s1H: number;
    private _s1L: number;

    constructor(private _seed?: null | string | number | number[]) {
        this._s0H = 0;
        this._s0L = 0;
        this._s1H = 0;
        this._s1L = 0;
        this.reset();
    }

    random(): number {
        const x = this.randomInt64();
        return (x[0] * 0x00200000 + (x[1] >>> 11)) * Math.pow(2, -53);
    }

    randomInt64(): [number, number] {
        // uint64_t s1 = s[0];
        let s1H = this._s0H;
        let s1L = this._s0L;
        // const uint64_t s0 = s[1];
        const s0H = this._s1H;
        const s0L = this._s1L;
        // s[0] = s0;
        this._s0H = s0H;
        this._s0L = s0L;
        // s1 ^= s1 << 23;
        s1H ^= ((s1H & 0x000001ff) << 23) | (s1L >>> 9);
        s1L ^= (s1L & 0x000001ff) << 23;
        // s[1] = s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26);
        this._s1H ^= s1H ^ (s1H >>> 17) ^ (s0H >>> 26);
        this._s1H >>>= 0;
        this._s1L ^= s1L ^ (((s1H & 0x0001ffff) << 15) | (s1L >>> 17)) ^ (((s0H & 0x03ffffff) << 6) | (s0L >>> 26));
        this._s1L >>>= 0;
        // return s[1] + s0;
        const t = this._s1L + s0L;
        return [(((t / 0x0100000000) | 0) + this._s1H + s0H) >>> 0, t >>> 0];
    }

    reset(): void {
        let seed = this._seed;

        if (typeof seed === 'string') {
            seed = this._hex2seed(4, seed);
        } else if (typeof seed === 'number') {
            seed = [
                (seed & 0xff0000) ^ 0xcafebabe,
                (seed & 0xff00) ^ 0xbe11fa54,
                (seed & 0xff000000) ^ 0x5eed3e11,
                (seed & 0xff) ^ 0xf7739aab,
            ];
        } else if (seed === null || seed === undefined) {
            seed = [Math.random() * 0xffffffff, Date.now() / 65536, Math.random() * 0xffffffff, Date.now() % 65536];
        } else if (seed.length !== 4) {
            throw new TypeError('seed length should equal 4');
        }

        this._s0H = seed[0] >>> 0;
        this._s0L = seed[1] >>> 0;
        this._s1H = seed[2] >>> 0;
        this._s1L = seed[3] >>> 0;
    }

    private _hex2seed(size: number, hex: string): number[] {
        const arr = new Array(size);
        for (let i = 0; i < size; ++i) {
            arr[i] = parseInt(hex.slice(i * 8, (i + 1) * 8), 16) >>> 0;
        }
        return arr;
    }
}
