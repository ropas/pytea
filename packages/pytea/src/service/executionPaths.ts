/*
 * executionPaths.ts
 * Copyright (c) Seoul National University
 * Licensed under the MIT license.
 *
 * Format results (Execution Paths) to communicate between language server and client
 */

import { Constraint } from '../backend/constraintType';
import { Context } from '../backend/context';
import { ShContFlag, ShValue, SVError, SVType } from '../backend/sharpValues';

export class FilePathStore {
    private _pathList: string[];
    private _idMap: Map<string, number>;

    constructor() {
        this._pathList = [];
        this._idMap = new Map();
    }

    addPath(path: string): number {
        if (this._idMap.has(path)) return this._idMap.get(path)!;

        const id = this._pathList.length;
        this._idMap.set(path, id);
        this._pathList.push(path);

        return id;
    }

    addPathList(paths: string[]): void {
        paths.forEach((p) => this.addPath(p));
    }

    getPath(id: number): string | undefined {
        if (this._pathList.length < id) return;
        return this._pathList[id];
    }

    getId(path: string): number | undefined {
        return this._idMap.get(path);
    }

    getPathList(): string[] {
        return [...this._pathList];
    }

    length(): number {
        return this._pathList.length;
    }
}

export enum ExecutionPathStatus {
    Success,
    Stopped,
    Failed,
}

// this will be used in communication between language server and client
// for more information of this path, the client should make another request to server
export interface ExecutionPathProps {
    status: ExecutionPathStatus;

    // Map each variables from env to correlated values in heap.
    // To remove deep value, all SVObjects will be replaced to its address (SVAddr)
    variables: Map<string, ShValue>;

    // ctrPool of ConstraintSet
    ctrPool: Constraint[];

    // indices in ctrPool
    hardCtr: number[];
    softCtr: number[];
    pathCtr: number[];

    // Logs, Warnings, or Errors from context
    logs: SVError[];

    // env and heap size of original context
    envSize: number;
    heapSize: number;
}

// concrete and specific path of analysis result.
// this will be managed from server.
export class ExecutionPath {
    readonly props: ExecutionPathProps;

    private _ctx: Context<ShValue | ShContFlag>;
    private _status: ExecutionPathStatus;
    private _pathStore: FilePathStore;

    constructor(result: Context<ShValue | ShContFlag>, status: ExecutionPathStatus, pathStore: FilePathStore) {
        this._ctx = result;
        this._status = status;
        this._pathStore = pathStore;

        this.props = this._initialize();
    }

    private _initialize(): ExecutionPathProps {
        const status = this._status;
        const variables = new Map<string, ShValue>();
        const ctrPool: Constraint[] = [];
        const logs: SVError[] = [];

        const ctx = this._ctx;
        const { env, heap, ctrSet } = ctx;

        if (status === ExecutionPathStatus.Success) {
            // TODO: currently assume that address 1 is main module object
            const module = heap.getVal(1);
            if (module?.type === SVType.Object) {
                // TODO
            }
        } else {
            // TODO
        }

        ctx.logs.forEach((value) => {
            if (value.type === SVType.Error) {
                const cleanValue = value.set('source', undefined);
                logs.push(cleanValue);
            } else {
                // TODO: clean normal value
            }
        });

        return {
            status,
            variables,
            ctrPool,
            hardCtr: ctrSet.hardCtr.toArray(),
            softCtr: ctrSet.softCtr.toArray(),
            pathCtr: ctrSet.pathCtr.toArray(),
            logs,
            envSize: env.addrMap.size,
            heapSize: heap.valMap.size,
        };
    }
}
