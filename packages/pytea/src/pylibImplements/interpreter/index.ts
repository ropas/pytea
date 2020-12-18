import { ParseNode } from 'pyright-internal/parser/parseNodes';

import { ThEnv, ThHeap } from '../../frontend/torchEnvironments';
import { ThValue } from '../../frontend/torchValues';
import { EvalLCBase } from './libcall';

export type EvalLCParamType = EvalLCBase.EBaseParamType;

export interface EvalLCContext {
    env: ThEnv;
    heap: ThHeap;
    params: ThValue[];
}

export interface EvalLCResult {
    heap: ThHeap;
    result: ThValue;
}

// TODO: how to remove any?
export type EvalLCImpl = (ctx: EvalLCContext, params: any, source?: ParseNode) => EvalLCResult;

export const libCallImpls: Map<string, EvalLCImpl> = new Map([...Object.entries(EvalLCBase.libCallImpls)]);
