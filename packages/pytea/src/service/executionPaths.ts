/*
 * executionPaths.ts
 * Copyright (c) Seoul National University
 * Licensed under the MIT license.
 *
 * Format results (Execution Paths) to communicate between language server and client
 */

import { getFileInfo } from 'pyright-internal/analyzer/analyzerNodeInfo';
import { convertOffsetToPosition } from 'pyright-internal/common/positionUtils';
import { ParseNodeType } from 'pyright-internal/parser/parseNodes';

import { ctrToStr } from '../backend/constraintType';
import { Context } from '../backend/context';
import { simplifyConstraint } from '../backend/expUtils';
import { CodeRange, CodeSource, ShContFlag, ShValue, SVError, SVErrorProps, SVType } from '../backend/sharpValues';

// maps file path to unique id (number) and vice versa
export class FilePathStore {
    private _pathList: string[];
    private _idMap: Map<string, number>;

    constructor(pathList?: string[]) {
        this._pathList = pathList ?? [];
        this._idMap = new Map();

        this.addPathList(this._pathList);
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

    toCodeRange(source: CodeSource | undefined): CodeRange | undefined {
        if (!source) return;

        if (!('fileId' in source)) {
            let moduleNode = source;
            while (moduleNode.nodeType !== ParseNodeType.Module) {
                moduleNode = moduleNode.parent!;
            }

            const fileInfo = getFileInfo(moduleNode)!;
            const filePath = fileInfo.filePath;

            const lines = fileInfo.lines;
            const start = convertOffsetToPosition(source.start, lines);
            const end = convertOffsetToPosition(source.start + source.length, lines);
            const fileId = this.addPath(filePath);

            return {
                fileId,
                range: { start, end },
            };
        }

        return source;
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
    pathId: number;
    projectRoot: string;

    status: ExecutionPathStatus;

    // Map each variables from env to correlated (stringified) values in heap.
    // To remove deep value, all SVObjects will be replaced to its address (SVAddr)
    variables: { [varName: string]: string };

    // stringified ctrPool of ConstraintSet with source
    ctrPool: [string, CodeRange?][];

    callStack: [string, CodeRange?][];

    // indices in ctrPool
    hardCtr: number[];
    softCtr: number[];
    pathCtr: number[];

    // Logs, Warnings, or Errors from context
    logs: SVErrorProps[];

    // env and heap size of original context
    envSize: number;
    heapSize: number;
}

// concrete and specific path of analysis result.
// this will be managed from server.
export class ExecutionPath {
    readonly props: ExecutionPathProps;
    readonly ctx: Context<ShValue | ShContFlag>;

    private _pathId: number;
    private _projectRoot: string;
    private _pathStore: FilePathStore;

    private _status: ExecutionPathStatus;

    constructor(
        result: Context<ShValue | ShContFlag>,
        pathId: number,
        projectRoot: string,
        status: ExecutionPathStatus,
        pathStore: FilePathStore
    ) {
        this.ctx = result;

        this._pathId = pathId;
        this._projectRoot = projectRoot;
        this._status = status;
        this._pathStore = pathStore;

        this.props = this._initialize();
    }

    private _initialize(): ExecutionPathProps {
        const status = this._status;
        const variables = {};
        const ctrPool: [string, CodeRange?][] = [];
        const logs: SVError[] = [];
        const callStack: [string, CodeRange?][] = [];

        const ctx = this.ctx;
        const { env, heap, ctrSet } = ctx;

        if (status === ExecutionPathStatus.Success) {
            // TODO: currently assume that address 1 is main module object
            const module = heap.getVal(1);
            if (module?.type === SVType.Object) {
                // TODO
            }
        } else {
            // TODO: add call stack
            ctx.callStack
                .filter(([f, _]) => {
                    // filter callKV libcall
                    if (typeof f === 'string') {
                        return f !== 'callKV';
                    } else {
                        return f.name !== 'callKV';
                    }
                })
                .forEach(([func, node]) => {
                    callStack.push([
                        `${typeof func === 'string' ? func : func.name}`,
                        this._pathStore.toCodeRange(node),
                    ]);
                });
            callStack.reverse();
        }

        ctx.logs.forEach((value) => {
            if (value.type === SVType.Error) {
                const cleanValue = value.set('source', this._pathStore.toCodeRange(value.source));
                logs.push(cleanValue);
            } else {
                // TODO: clean normal value
            }
        });

        ctx.ctrSet.ctrPool.forEach((ctr) => {
            const ctrStr = ctrToStr(simplifyConstraint(ctx.ctrSet, ctr));
            const source = this._pathStore.toCodeRange(ctr.source);

            ctrPool.push([ctrStr, source]);
        });

        return {
            pathId: this._pathId,
            projectRoot: this._projectRoot,
            status,
            variables,
            ctrPool,
            callStack,
            hardCtr: ctrSet.hardCtr.toArray(),
            softCtr: ctrSet.softCtr.toArray(),
            pathCtr: ctrSet.pathCtr.toArray(),
            logs,
            envSize: env.addrMap.size,
            heapSize: heap.valMap.size,
        };
    }
}
