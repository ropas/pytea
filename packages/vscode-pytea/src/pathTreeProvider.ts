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
    window,
} from 'vscode';

interface PyteaTreeProvider {
    refresh(props?: ExecutionPathProps): void;
}

// TODO: use ES6 Symbol?
const idCache: { [id: number]: { id: number } } = {};
function getId(id: number): { id: number } {
    if (id in idCache) return idCache[id];
    const newId = { id };
    idCache[id] = newId;
    return newId;
}

export class PathSelectionProvider implements TreeDataProvider<{ id: number }> {
    private _onDidChangeTreeData: EventEmitter<any> = new EventEmitter<any>();
    readonly onDidChangeTreeData: Event<any> = this._onDidChangeTreeData.event;

    constructor(private pathProps: ExecutionPathProps[]) {}

    public refresh(pathProps: ExecutionPathProps[]): void {
        this.pathProps = pathProps;
        console.log(`refresh with ${pathProps.length}`);
        this._onDidChangeTreeData.fire(undefined);
    }

    public getTreeItem(pathId: { id: number }): TreeItem {
        return {
            label: this._label(pathId.id),
            command: {
                command: 'pytea.selectPath',
                arguments: [pathId.id],
                title: 'Select Specific Execution Path',
            },
        };
    }

    public getChildren(element?: { id: number }): ProviderResult<{ id: number }[]> {
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

export class VariableDataProvider implements TreeDataProvider<string>, PyteaTreeProvider {
    private _onDidChangeTreeData: EventEmitter<any> = new EventEmitter<any>();
    readonly onDidChangeTreeData: Event<any> = this._onDidChangeTreeData.event;

    constructor(private props?: ExecutionPathProps) {}

    public refresh(props?: ExecutionPathProps): void {
        this.props = props;
        this._onDidChangeTreeData.fire(undefined);
    }

    public getTreeItem(element: string): TreeItem {
        return {
            // resourceUri: "",
            label: `var ${element}`,
        };
    }

    public getChildren(element?: string): ProviderResult<string[]> {
        // return element ? this.model.getChildren(element) : this.model.roots;
        return undefined;
    }
}

export class CallStackProvider implements TreeDataProvider<number>, PyteaTreeProvider {
    private _onDidChangeTreeData: EventEmitter<any> = new EventEmitter<any>();
    readonly onDidChangeTreeData: Event<any> = this._onDidChangeTreeData.event;

    constructor(private props?: ExecutionPathProps) {}

    public refresh(props?: ExecutionPathProps): void {
        this.props = props;
        this._onDidChangeTreeData.fire(undefined);
    }

    public getTreeItem(element: number): TreeItem {
        return {
            // resourceUri: "",
            command: {
                command: 'pytea.selectPath',
                arguments: [element],
                title: 'Select Specific Execution Path',
            },
        };
    }

    public getChildren(element?: number): ProviderResult<number[]> {
        // return element ? this.model.getChildren(element) : this.model.roots;
        return undefined;
    }
}

export class ConstraintDataProvider implements TreeDataProvider<number>, PyteaTreeProvider {
    private _onDidChangeTreeData: EventEmitter<any> = new EventEmitter<any>();
    readonly onDidChangeTreeData: Event<any> = this._onDidChangeTreeData.event;

    constructor(private props?: ExecutionPathProps) {}

    public refresh(props?: ExecutionPathProps): void {
        this.props = props;
        this._onDidChangeTreeData.fire(undefined);
    }

    public getTreeItem(element: number): TreeItem {
        return {
            // resourceUri: "",
            command: {
                command: 'pytea.selectPath',
                arguments: [element],
                title: 'Select Specific Execution Path',
            },
        };
    }

    public getChildren(element?: number): ProviderResult<number[]> {
        // return element ? this.model.getChildren(element) : this.model.roots;
        return undefined;
    }
}

export class PathManager {
    private _pathTree: TreeView<number>;
    private _varTree: TreeView<number>;
    private _stackTree: TreeView<number>;
    private _hardCtrTree: TreeView<number>;
    private _pathCtrTree: TreeView<number>;
    private _softCtrTree: TreeView<number>;

    private _pathProps: ExecutionPathProps[];
    private _mainProvider: PathSelectionProvider;
    private _providers: PyteaTreeProvider[];

    constructor(context: ExtensionContext) {
        this._pathProps = [];
        this._mainProvider = new PathSelectionProvider(this._pathProps);
        const providers = [
            new VariableDataProvider(),
            new CallStackProvider(),
            new ConstraintDataProvider(),
            new ConstraintDataProvider(),
            new ConstraintDataProvider(),
        ];
        this._pathTree = window.createTreeView('executionPaths', { treeDataProvider: this._mainProvider });
        this._varTree = window.createTreeView('variables', { treeDataProvider: providers[0] });
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

        commands.registerCommand('pytea.selectPath', (pathId) => {
            if (pathId < 0 || pathId >= this._pathProps.length) {
                console.log(`selected ${pathId}`);
                return;
            }
        });
        // commands.registerCommand('pytea.gotoCallStack', (frameId) => this.reveal());
        // commands.registercommand('pytea.gotoconstraint', (ctrid) => this.reveal());
    }

    applyPathProps(results: ExecutionPathProps[]) {
        this._mainProvider.refresh(results);
        let mainPath: ExecutionPathProps | undefined;
        if (results.length > 0) {
            mainPath = results[0];
        }
        this._providers.forEach((p) => p.refresh(mainPath));
    }
}
