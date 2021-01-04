/*
 * fraction.ts
 * Copyright (c) Seoul National University.
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo (mersshs@gmail.com)
 *
 * implement fraction. guaranteed that denominator is non-zero integer
 */

export function gcd(a: number, b: number): number {
    if (a === 0 || b === 0) return 1;
    if (!(Number.isInteger(a) && Number.isInteger(b))) return 1;
    a = Math.abs(a);
    b = Math.abs(b);
    let tmp = 0;

    while (a !== 0 && b !== 0) {
        a = a % b;
        tmp = a;
        a = b;
        b = tmp;
    }

    return a === 0 ? b : a;
}

export class Fraction {
    up: number; // numerator
    down: number; // denominator

    constructor(up: number, down: number) {
        this.up = up;
        this.down = down;
    }

    add(that: Fraction): Fraction {
        return new Fraction(this.up * that.down + this.down * that.up, this.down * that.down);
    }

    addN(n: number): Fraction {
        return new Fraction(this.up + n * this.down, this.down);
    }

    sub(that: Fraction): Fraction {
        return new Fraction(this.up * that.down - this.down * that.up, this.down * that.down);
    }

    subN(n: number): Fraction {
        return new Fraction(this.up - n * this.down, this.down);
    }

    mul(that: Fraction): Fraction {
        return new Fraction(this.up * that.up, this.down * that.down);
    }

    mulN(n: number): Fraction {
        return new Fraction(this.up * n, this.down);
    }

    div(that: Fraction): Fraction {
        if (that.up === 0) {
            return new Fraction(this.up * that.down * this.down >= 0 ? Infinity : -Infinity, 1);
        }
        return new Fraction(this.up * that.down, this.down * that.up);
    }

    divN(n: number): Fraction {
        if (n === 0) {
            return new Fraction(this.up * n * this.down >= 0 ? Infinity : -Infinity, 1);
        }
        if (Number.isInteger(n)) {
            return new Fraction(this.up, n * this.down);
        } else {
            return new Fraction(this.up / n, this.down);
        }
    }

    floor(): Fraction {
        return new Fraction(Math.floor(this.up / this.down) * this.down, this.down);
    }

    neg(): Fraction {
        return new Fraction(-this.up, this.down);
    }

    toNum(): number {
        return this.up / this.down;
    }

    // normalize by gcd
    norm(): Fraction {
        const g = gcd(this.up, this.down);
        if (this.down < 0) {
            this.down *= -1;
            this.up *= -1;
        }

        this.up /= g;
        this.down /= g;

        return this;
    }

    toString(): string {
        return `${this.up}/${this.down}`;
    }
}
