/*
 * pylibImplement/backend/index.ts
 * Copyright (c) Seoul National University.
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo
 *
 * Base types and collections of every LibCall.
 */
import { ParseNode } from 'pyright-internal/parser/parseNodes';

import { Context, ContextSet } from '../../backend/context';
import { ShValue } from '../../backend/sharpValues';
import { libCallMap as argparseMap } from './argparse';
import { libCallMap as builtinsMap } from './builtins';
import { LCBase, libCallMap as baseMap } from './libcall';
import { libCallMap as shapeMap } from './shape';
import { libCallMap as torchMap } from './torch';
import { libCallMap as torchvisionMap } from './torchvision';

export type LCParamType = LCBase.BaseParamType;

// TODO: how to not use any?
export type LCImpl = (ctx: Context<any>, source?: ParseNode) => ContextSet<ShValue>;

export function registLibCall(impls: Map<string, LCImpl>, relPath: string): Map<string, LCImpl> {
    const map: Map<string, LCImpl> = new Map();
    impls.forEach((v, k) => {
        map.set(`${relPath}.${k}`, v);
    });
    return map;
}

export const libCallMap: Map<string, LCImpl> = new Map([
    ...baseMap,
    ...registLibCall(torchMap, 'torch'),
    ...registLibCall(torchvisionMap, 'torchvision'),
    ...registLibCall(shapeMap, 'shape'),
    ...registLibCall(builtinsMap, 'builtins'),
    ...registLibCall(argparseMap, 'argparse'),
]);
