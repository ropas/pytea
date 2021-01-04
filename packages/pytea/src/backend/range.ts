/*
 * range.ts
 * Copyright (c) Seoul National University.
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo (mersshs@gmail.com)
 *
 * Range based simple SMT
 */
export class NumRange {
    // TODO: check NaN
    readonly start: number;
    readonly end: number;
    readonly hasStart: boolean; // is start inclusive?
    readonly hasEnd: boolean; // is end inclusive?

    constructor(start: number, end: number, hasStart: boolean, hasEnd: boolean) {
        this.start = start;
        this.end = end;
        this.hasStart = hasStart;
        this.hasEnd = hasEnd;
    }

    toString(): string {
        return `${this.hasStart ? '[' : '('}${this.start}, ${this.end}${this.hasEnd ? ']' : ')'}`;
    }

    // return closed range
    static fromConst(num: number): NumRange {
        return new NumRange(num, num, true, true);
    }
    static genClosed(start: number, end: number): NumRange {
        return new NumRange(start, end, true, true);
    }
    static genTop(): NumRange {
        return new NumRange(-Infinity, Infinity, false, false);
    }
    static genFalse(): NumRange {
        return new NumRange(1, -1, true, true);
    }
    static genLt(num: number): NumRange {
        return new NumRange(-Infinity, num, false, false);
    }
    static genLte(num: number): NumRange {
        return new NumRange(-Infinity, num, false, true);
    }
    static genGt(num: number): NumRange {
        return new NumRange(num, Infinity, false, false);
    }
    static genGte(num: number): NumRange {
        return new NumRange(num, Infinity, true, false);
    }

    valid(): boolean {
        if (this.start === this.end) return this.hasStart && this.hasEnd;
        return this.start < this.end;
    }

    isConst(): boolean {
        return this.start === this.end && this.hasStart && this.hasEnd;
    }
    // WARNING: this.isTruthy() == false is not means its falsy!!
    isTruthy(): boolean {
        return this.gt(0) === true || this.lt(0) === true;
    }
    // WARNING: this.isFalsy() == false is not means its truthy!!
    isFalsy(): boolean {
        return this.isConst() && this.start === 0;
    }

    toIntRange(): NumRange | undefined {
        if (!this.valid()) return;

        let start = this.start;
        let end = this.end;
        if (start !== -Infinity) {
            if (Number.isInteger(start)) {
                start = this.hasStart ? start : start + 1;
            } else {
                start = Math.floor(start) + 1;
            }
        }

        if (end !== Infinity) {
            if (Number.isInteger(end)) {
                end = this.hasEnd ? end : end - 1;
            } else {
                end = Math.floor(end);
            }
        }

        const newValue = new NumRange(start, end, start !== -Infinity, end !== Infinity);
        if (!newValue.valid()) return;

        return newValue;
    }

    lt(num: number): boolean | undefined {
        if (!this.valid()) return;
        if (this.isConst()) return this.start < num;

        if (this.end < num) return true;
        if (this.end === num) return this.hasEnd ? undefined : true;
        if (this.start >= num) return false;

        return;
    }
    lte(num: number): boolean | undefined {
        if (!this.valid()) return;
        if (this.isConst()) return this.start <= num;

        if (this.end <= num) return true;
        if (this.start > num) return false;
        if (this.start === num) return this.hasStart ? undefined : false;

        return;
    }
    gt(num: number): boolean | undefined {
        if (!this.valid()) return;
        if (this.isConst()) return this.start > num;

        if (this.end <= num) return false;
        if (this.start > num) return true;
        if (this.start === num) return this.hasStart ? undefined : true;

        return;
    }
    gte(num: number): boolean | undefined {
        if (!this.valid()) return;
        if (this.isConst()) return this.start >= num;

        if (this.end < num) return false;
        if (this.end === num) return this.hasEnd ? undefined : false;
        if (this.start >= num) return true;

        return;
    }
    eq(num: number): boolean | undefined {
        if (!this.isConst()) return;
        return this.start === num;
    }

    // TODO: return false if invalid??
    ltRange(that: NumRange): boolean | undefined {
        if (!this.valid() || !that.valid()) return;

        if (this.end < that.start) return true;
        if (this.end === that.start && !(this.hasEnd && that.hasStart)) return true;
        if (that.end <= this.start) return false;

        return;
    }
    lteRange(that: NumRange): boolean | undefined {
        if (!this.valid() || !that.valid()) return;

        if (this.end <= that.start) return true;
        if (that.end < this.start) return false;
        if (that.end === this.start && !(that.hasEnd && this.hasStart)) return false;

        return;
    }
    gtRange(that: NumRange): boolean | undefined {
        return that.lteRange(this);
    }
    gteRange(that: NumRange): boolean | undefined {
        return that.ltRange(this);
    }

    contains(num: number): boolean {
        if (!this.valid()) {
            return false;
        }

        if (this.start <= num && num <= this.end) {
            if (this.start === num) return this.hasStart;
            if (this.end === num) return this.hasEnd;
            return true;
        }

        return false;
    }

    intersect(that: NumRange): NumRange {
        if (!this.valid()) {
            return this;
        }
        if (!that.valid()) {
            return that;
        }

        let start: number;
        let end: number;
        let hasStart: boolean;
        let hasEnd: boolean;

        if (this.start === that.start) {
            start = this.start;
            hasStart = this.hasStart && that.hasStart;
        } else if (this.start < that.start) {
            start = that.start;
            hasStart = that.hasStart;
        } else {
            start = this.start;
            hasStart = this.hasStart;
        }

        if (this.end === that.end) {
            end = this.end;
            hasEnd = this.hasEnd && that.hasEnd;
        } else if (this.start < that.start) {
            end = this.end;
            hasEnd = this.hasEnd;
        } else {
            end = that.end;
            hasEnd = that.hasEnd;
        }

        return new NumRange(start, end, hasStart, hasEnd);
    }

    union(that: NumRange): NumRange {
        if (!this.valid()) {
            return this;
        }
        if (!that.valid()) {
            return that;
        }

        let start: number;
        let end: number;
        let hasStart: boolean;
        let hasEnd: boolean;

        if (this.start === that.start) {
            start = this.start;
            hasStart = this.hasStart || that.hasStart;
        } else if (this.start < that.start) {
            start = this.start;
            hasStart = this.hasStart;
        } else {
            start = that.start;
            hasStart = that.hasStart;
        }

        if (this.end === that.end) {
            end = this.end;
            hasEnd = this.hasEnd || that.hasEnd;
        } else if (this.start < that.start) {
            end = that.end;
            hasEnd = that.hasEnd;
        } else {
            end = this.end;
            hasEnd = this.hasEnd;
        }

        return new NumRange(start, end, hasStart, hasEnd);
    }

    neg(): NumRange {
        if (!this.valid()) return this;

        return new NumRange(-this.end, -this.start, this.hasEnd, this.hasStart);
    }

    abs(): NumRange {
        if (!this.valid()) return this;

        if (this.start > 0) return this;
        if (this.end <= 0) return this.neg();

        if (-this.start < this.end) {
            return new NumRange(0, this.end, true, this.hasEnd);
        } else if (-this.start === this.end) {
            return new NumRange(0, this.end, true, this.hasEnd || this.hasStart);
        } else {
            return new NumRange(0, -this.start, true, this.hasStart);
        }
    }

    ceil(): NumRange | undefined {
        if (!this.valid()) return;
        let start: number;
        let hasStart = true;
        let end: number;
        let hasEnd = true;

        if (this.start === -Infinity) {
            start = -Infinity;
            hasStart = false;
        } else if (Number.isInteger(this.start)) {
            start = this.hasStart ? this.start : this.start + 1;
        } else {
            start = Math.ceil(this.start);
        }

        if (this.end === Infinity) {
            end = Infinity;
            hasEnd = false;
        } else {
            end = Math.ceil(this.end);
        }

        return new NumRange(start, end, hasStart, hasEnd);
    }

    floor(): NumRange | undefined {
        if (!this.valid()) return;
        let start: number;
        let hasStart = true;
        let end: number;
        let hasEnd = true;

        if (this.start === -Infinity) {
            start = -Infinity;
            hasStart = false;
        } else {
            start = Math.floor(this.start);
        }

        if (this.end === Infinity) {
            end = Infinity;
            hasEnd = false;
        } else if (Number.isInteger(this.end)) {
            end = this.hasEnd ? this.end : this.end - 1;
        } else {
            end = Math.floor(this.end);
        }

        return new NumRange(start, end, hasStart, hasEnd);
    }

    add(that: NumRange): NumRange {
        if (!this.valid()) {
            return this;
        }
        if (!that.valid()) {
            return that;
        }
        return new NumRange(
            this.start + that.start,
            this.end + that.end,
            this.hasStart && that.hasStart,
            this.hasEnd && that.hasEnd
        );
    }

    sub(that: NumRange): NumRange {
        return this.add(that.neg());
    }

    mul(that: NumRange): NumRange {
        if (!this.valid()) {
            return this;
        }
        if (!that.valid()) {
            return that;
        }

        if (this.isConst()) {
            const start = this.start;
            if (start === 0) return this;
            else if (start > 0) return new NumRange(start * that.start, start * that.end, that.hasStart, that.hasEnd);
            else return new NumRange(start * that.end, start * that.start, that.hasEnd, that.hasStart);
        }

        if (that.isConst()) {
            const start = that.start;
            if (start === 0) return that;
            else if (start > 0) return new NumRange(start * this.start, start * this.end, this.hasStart, this.hasEnd);
            else return new NumRange(start * this.end, start * this.start, this.hasEnd, this.hasStart);
        }

        let a = this.start * that.start;
        let b = this.start * that.end;
        let c = this.end * that.start;
        let d = this.end * that.end;

        // -0, NaN / Infinity resolve
        if (this.start === 0) {
            a = 0;
            b = 0;
        }
        if (that.start === 0) {
            a = 0;
            c = 0;
        }
        if (this.end === 0) {
            c = 0;
            d = 0;
        }
        if (that.end === 0) {
            b = 0;
            d = 0;
        }

        const [min, minpos] = min4(a, b, c, d);
        const [max, maxpos] = max4(a, b, c, d);

        return new NumRange(min, max, flagByPos(minpos, this, that), flagByPos(maxpos, this, that));
    }

    floordiv(that: NumRange): NumRange {
        if (!this.valid()) {
            return this;
        }
        if (!that.valid()) {
            return that;
        }

        if (this.isConst() && that.isConst()) {
            if (that.start === 0) {
                return NumRange.genTop();
            }
            return NumRange.fromConst(Math.floor(this.start / that.start));
        }

        if (that.contains(0)) {
            return NumRange.genTop();
        }

        // TODO: can we do like this??
        let a = Math.floor(this.start / that.start);
        let b = Math.floor(this.start / that.end);
        let c = Math.floor(this.end / that.start);
        let d = Math.floor(this.end / that.end);

        // -0, NaN / Infinity resolve
        if (that.start === 0) {
            a = this.start >= 0 ? Infinity : -Infinity;
            c = this.end >= 0 ? Infinity : -Infinity;
        }
        if (that.end === 0) {
            b = this.start >= 0 ? -Infinity : Infinity;
            d = this.end >= 0 ? -Infinity : Infinity;
        }

        const [min, minpos] = min4(a, b, c, d);
        const [max, maxpos] = max4(a, b, c, d);

        return new NumRange(
            min,
            max,
            min === -Infinity ? false : flagByPos(minpos, this, that),
            max === Infinity ? flagByPos(maxpos, this, that) : false
        );
    }

    truediv(that: NumRange): NumRange {
        if (!this.valid()) {
            return this;
        }
        if (!that.valid()) {
            return that;
        }

        if (this.isConst() && that.isConst()) {
            if (that.start === 0) {
                return NumRange.genTop();
            }
            return NumRange.fromConst(this.start / that.start);
        }

        if (that.contains(0)) {
            return NumRange.genTop();
        }

        // TODO: can we do like this??
        let a = this.start / that.start;
        let b = this.start / that.end;
        let c = this.end / that.start;
        let d = this.end / that.end;

        // -0, NaN / Infinity resolve
        if (that.start === 0) {
            a = this.start >= 0 ? Infinity : -Infinity;
            c = this.end >= 0 ? Infinity : -Infinity;
        }
        if (that.end === 0) {
            b = this.start >= 0 ? -Infinity : Infinity;
            d = this.end >= 0 ? -Infinity : Infinity;
        }

        const [min, minpos] = min4(a, b, c, d);
        const [max, maxpos] = max4(a, b, c, d);

        return new NumRange(
            min,
            max,
            min === -Infinity ? false : flagByPos(minpos, this, that),
            max === Infinity ? flagByPos(maxpos, this, that) : false
        );
    }

    mod(that: NumRange): NumRange {
        if (!this.valid()) {
            return this;
        }
        if (!that.valid()) {
            return that;
        }

        if (this.isConst()) {
            if (that.isConst()) {
                return NumRange.fromConst(this.start % that.start);
            } else if (that.gt(Math.abs(this.start))) {
                return this;
            }
        }

        // TODO: more precisely
        return new NumRange(0, Math.max(Math.abs(that.start), Math.abs(that.end)), true, false);
    }

    pow(that: NumRange): NumRange {
        if (!this.valid()) {
            return this;
        }
        if (!that.valid()) {
            return that;
        }

        if (this.isConst()) {
            const start = this.start;
            if (start === 0) return this;
            else if (start > 0) return new NumRange(start * that.start, start * that.end, that.hasStart, that.hasEnd);
            else return new NumRange(start * that.end, start * that.start, that.hasEnd, that.hasStart);
        }

        if (that.isConst()) {
            const start = that.start;
            if (start === 0) return that;
            else if (start > 0) return new NumRange(start * this.start, start * this.end, this.hasStart, this.hasEnd);
            else return new NumRange(start * this.end, start * this.start, this.hasEnd, this.hasStart);
        }

        // TODO: can we do like this??
        const a = Math.floor(this.start / that.start);
        const b = Math.floor(this.start / that.end);
        const c = Math.floor(this.end / that.start);
        const d = Math.floor(this.end / that.end);
        const [min, minpos] = min4(a, b, c, d);
        const [max, maxpos] = max4(a, b, c, d);

        return new NumRange(
            min,
            max,
            min === -Infinity ? false : flagByPos(minpos, this, that),
            max === Infinity ? flagByPos(maxpos, this, that) : false
        );
    }

    max(that: NumRange): NumRange {
        if (!this.valid()) {
            return this;
        }
        if (!that.valid()) {
            return that;
        }

        let start: number;
        let end: number;
        let hasStart: boolean;
        let hasEnd: boolean;

        if (this.start < that.start) {
            start = that.start;
            hasStart = that.hasStart;
        } else if (this.start === that.start) {
            start = this.start;
            hasStart = this.hasStart && that.hasStart;
        } else {
            start = this.start;
            hasStart = this.hasStart;
        }

        if (this.end < that.end) {
            end = that.end;
            hasEnd = that.hasEnd;
        } else if (this.end === that.end) {
            end = this.end;
            hasEnd = this.hasEnd || that.hasEnd;
        } else {
            end = this.end;
            hasEnd = this.hasEnd;
        }

        return new NumRange(start, end, hasStart, hasEnd);
    }

    min(that: NumRange): NumRange {
        if (!this.valid()) {
            return this;
        }
        if (!that.valid()) {
            return that;
        }

        let start: number;
        let end: number;
        let hasStart: boolean;
        let hasEnd: boolean;

        if (this.start < that.start) {
            start = this.start;
            hasStart = this.hasStart;
        } else if (this.start === that.start) {
            start = this.start;
            hasStart = this.hasStart || that.hasStart;
        } else {
            start = that.start;
            hasStart = that.hasStart;
        }

        if (this.end < that.end) {
            end = this.end;
            hasEnd = this.hasEnd;
        } else if (this.end === that.end) {
            end = this.end;
            hasEnd = this.hasEnd && that.hasEnd;
        } else {
            end = that.end;
            hasEnd = that.hasEnd;
        }

        return new NumRange(start, end, hasStart, hasEnd);
    }
}

function min4(a: number, b: number, c: number, d: number): [number, number] {
    if (Number.isNaN(a)) a = Infinity;
    if (Number.isNaN(b)) b = Infinity;
    if (Number.isNaN(c)) c = Infinity;
    if (Number.isNaN(d)) d = Infinity;

    const ab: [number, number] = a < b ? [a, 0] : [b, 1];
    const cd: [number, number] = c < d ? [c, 2] : [d, 3];
    return ab[0] < cd[0] ? ab : cd;
}

function max4(a: number, b: number, c: number, d: number): [number, number] {
    if (Number.isNaN(a)) a = -Infinity;
    if (Number.isNaN(b)) b = -Infinity;
    if (Number.isNaN(c)) c = -Infinity;
    if (Number.isNaN(d)) d = -Infinity;

    const ab: [number, number] = a > b ? [a, 0] : [b, 1];
    const cd: [number, number] = c > d ? [c, 2] : [d, 3];
    return ab[0] > cd[0] ? ab : cd;
}

// pos: 0    -> (this.start, that.start)
// pos: 1    -> (this.start, that.end)
// pos: 2    -> (this.end, that.start)
// pos: o.w. -> (this.end, that.end)
function flagByPos(pos: number, left: NumRange, right: NumRange): boolean {
    switch (pos) {
        case 0:
            return left.hasStart && right.hasStart;
        case 1:
            return left.hasStart && right.hasEnd;
        case 2:
            return left.hasEnd && right.hasStart;
        default:
            return left.hasEnd && right.hasEnd;
    }
}
