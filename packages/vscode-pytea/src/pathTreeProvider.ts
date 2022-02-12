/*
 * pathTreeProvider.ts
 * Copyright (c) Seoul National University
 * Licensed under the MIT license.
 *
 * Manage datas of sidebar
 */

import { ExecutionPathProps, ExecutionPathStatus } from 'pytea/service/executionPaths';
import {
    commands,
    Event,
    EventEmitter,
    ExtensionContext,
    ProviderResult,
    TreeDataProvider,
    TreeItem,
    TreeView,
    Uri,
    window,
} from 'vscode';

import { Commands } from './commands';

interface PyteaTreeProvider {
    refresh(props?: ExecutionPathProps): void;
}

// TODO: use ES6 Symbol?
type NumId = { id: number };
const idCache: { [id: number]: NumId } = {};
function getId(id: number): NumId {
    if (id in idCache) return idCache[id];
    const newId = { id };
    idCache[id] = newId;
    return newId;
}

export class PathSelectionProvider implements TreeDataProvider<NumId> {
    private _onDidChangeTreeData: EventEmitter<any> = new EventEmitter<any>();
    readonly onDidChangeTreeData: Event<any> = this._onDidChangeTreeData.event;

    constructor(private pathProps: ExecutionPathProps[]) {}

    public refresh(pathProps: ExecutionPathProps[]): void {
        this.pathProps = pathProps;
        this._onDidChangeTreeData.fire(undefined);
    }

    public getTreeItem(pathId: NumId): TreeItem {
        return {
            label: this._label(pathId.id),
            command: {
                command: 'pytea.selectPath',
                arguments: [pathId.id],
                title: 'Select Specific Execution Path',
            },
        };
    }

    public getChildren(element?: NumId): ProviderResult<NumId[]> {
        if (element !== undefined) return [];
        return this.pathProps.map((_, i) => {
            return getId(i);
        });
    }

    private _label(pathId: number): string {
        if (pathId < 0 || this.pathProps.length <= pathId) return `Unknown Id ${pathId}`;
        const path = this.pathProps[pathId];
        switch (path.status) {
            case ExecutionPathStatus.Success:
                return `🟢 Success path ${pathId + 1}`;
            case ExecutionPathStatus.Stopped:
                return `🟡 Stopped path ${pathId + 1}`;
            case ExecutionPathStatus.Failed:
                return `🔴 Failed path ${pathId + 1}`;
        }
    }
}

export class VariableDataProvider implements TreeDataProvider<NumId>, PyteaTreeProvider {
    private _onDidChangeTreeData: EventEmitter<any> = new EventEmitter<any>();
    readonly onDidChangeTreeData: Event<any> = this._onDidChangeTreeData.event;

    constructor(private props?: ExecutionPathProps) {}

    public refresh(props?: ExecutionPathProps): void {
        this.props = props;
        this._onDidChangeTreeData.fire(undefined);
    }

    public getTreeItem(element: NumId): TreeItem {
        return {
            // resourceUri: "",
            label: `var ${element}`,
        };
    }

    public getChildren(element?: NumId): ProviderResult<NumId[]> {
        // return element ? this.model.getChildren(element) : this.model.roots;
        return undefined;
    }
}

export class CallStackProvider implements TreeDataProvider<NumId>, PyteaTreeProvider {
    private _onDidChangeTreeData: EventEmitter<any> = new EventEmitter<any>();
    readonly onDidChangeTreeData: Event<any> = this._onDidChangeTreeData.event;

    constructor(private props?: ExecutionPathProps) {}

    public refresh(props?: ExecutionPathProps): void {
        this.props = props;
        this._onDidChangeTreeData.fire(undefined);
    }

    public getTreeItem(element: NumId): TreeItem {
        return {
            // resourceUri: "",
            command: {
                command: 'pytea.selectPath',
                arguments: [element],
                title: 'Select Specific Execution Path',
            },
        };
    }

    public getChildren(element?: NumId): ProviderResult<NumId[]> {
        // return element ? this.model.getChildren(element) : this.model.roots;
        return undefined;
    }
}

const enum CtrType {
    Hard,
    Path,
    Soft,
}
export class ConstraintDataProvider implements TreeDataProvider<NumId>, PyteaTreeProvider {
    private _onDidChangeTreeData: EventEmitter<any> = new EventEmitter<any>();
    readonly onDidChangeTreeData: Event<any> = this._onDidChangeTreeData.event;

    private _ctrIdList: number[];

    constructor(private type: CtrType, private props?: ExecutionPathProps) {
        this._ctrIdList = [];
    }

    public refresh(props?: ExecutionPathProps): void {
        this.props = props;
        if (props) {
            switch (this.type) {
                case CtrType.Hard:
                    this._ctrIdList = props.hardCtr;
                    break;
                case CtrType.Path:
                    this._ctrIdList = props.pathCtr;
                    break;
                case CtrType.Soft:
                    this._ctrIdList = props.softCtr;
                    break;
            }
        } else {
            this._ctrIdList = [];
        }
        this._onDidChangeTreeData.fire(undefined);
    }

    public getTreeItem(nid: NumId): TreeItem {
        const ctrId = this._ctrIdList[nid.id];
        const ctr = this.props?.ctrPool[ctrId];
        if (!ctr) {
            return {};
        }

        let resourceUri: Uri | undefined;
        // if (ctr[1]) {
        //     const source = ctr[1]
        //     resourceUri = Uri.file(source[0] + '')
        // }
        return {
            resourceUri,
            label: ctr[0],
            command: {
                command: 'pytea.gotoConstraint',
                arguments: [ctrId],
                title: 'Go To The Generation Point Of The Constraint',
            },
        };
    }

    public getChildren(nid?: NumId): ProviderResult<NumId[]> {
        if (nid !== undefined) return undefined;
        return this._ctrIdList.map((_, i) => {
            return getId(i);
        });
    }
}

export interface PathManagerCallback {
    selectPath?: (pathId: number) => void;
}
export class PathManager {
    private _pathTree: TreeView<NumId>;
    private _varTree: TreeView<NumId>;
    private _stackTree: TreeView<NumId>;
    private _hardCtrTree: TreeView<NumId>;
    private _pathCtrTree: TreeView<NumId>;
    private _softCtrTree: TreeView<NumId>;

    private _pathProps: ExecutionPathProps[];
    private _mainProvider: PathSelectionProvider;
    private _providers: PyteaTreeProvider[];

    constructor(context: ExtensionContext, callback: PathManagerCallback) {
        this._pathProps = [];
        this._mainProvider = new PathSelectionProvider(this._pathProps);
        const providers = [
            new VariableDataProvider(),
            new CallStackProvider(),
            new ConstraintDataProvider(CtrType.Soft),
            new ConstraintDataProvider(CtrType.Path),
            new ConstraintDataProvider(CtrType.Hard),
        ];
        this._pathTree = window.createTreeView('executionPaths', {
            treeDataProvider: this._mainProvider,
            canSelectMany: false,
        });
        this._varTree = window.createTreeView('variables', {
            treeDataProvider: providers[0],
            showCollapseAll: true,
            canSelectMany: false,
        });
        this._stackTree = window.createTreeView('callStack', { treeDataProvider: providers[1] });
        this._softCtrTree = window.createTreeView('softConstraints', { treeDataProvider: providers[2] });
        this._pathCtrTree = window.createTreeView('pathConstraints', { treeDataProvider: providers[3] });
        this._hardCtrTree = window.createTreeView('hardConstraints', { treeDataProvider: providers[4] });
        this._providers = providers;

        context.subscriptions.push(this._pathTree);
        context.subscriptions.push(this._varTree);
        context.subscriptions.push(this._stackTree);
        context.subscriptions.push(this._softCtrTree);
        context.subscriptions.push(this._pathCtrTree);
        context.subscriptions.push(this._hardCtrTree);

        commands.registerCommand(Commands.selectPath, (pathId) => {
            if (callback.selectPath) callback.selectPath(pathId);

            if (pathId < 0 || pathId >= this._pathProps.length) {
                console.log(`selected ${pathId}`);
                this._providers.forEach((p) => p.refresh(this._pathProps[pathId]));
            }
        });
        // commands.registerCommand('pytea.gotoCallStack', (frameId) => this.reveal());
        // commands.registercommand('pytea.gotoconstraint', (ctrid) => this.reveal());
    }

    applyPathProps(results: ExecutionPathProps[]) {
        this._pathProps = results;
        this._mainProvider.refresh(results);
        let mainPath: ExecutionPathProps | undefined;
        if (results.length > 0) {
            mainPath = results[0];
        }
        this._providers.forEach((p) => p.refresh(mainPath));
    }
}
