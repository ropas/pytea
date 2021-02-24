/*
 * pathTreeProvider.ts
 * Copyright (c) Seoul National University
 * Licensed under the MIT license.
 *
 * Manage datas of sidebar
 */

import { ExecutionPathProps } from 'pytea/service/executionPaths';
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

export class PathSelectionProvider implements TreeDataProvider<number> {
    private _onDidChangeTreeData: EventEmitter<any> = new EventEmitter<any>();
    readonly onDidChangeTreeData: Event<any> = this._onDidChangeTreeData.event;

    constructor(private pathProps: ExecutionPathProps[]) {}

    public refresh(pathProps: ExecutionPathProps[]): void {
        this.pathProps = pathProps;
        this._onDidChangeTreeData.fire(undefined);
    }

    public getTreeItem(pathId: number): TreeItem {
        return {
            command: {
                command: 'pytea.selectPath',
                arguments: [pathId],
                title: 'Select Specific Execution Path',
            },
        };
    }

    public getChildren(element?: number): ProviderResult<number[]> {
        if (element !== undefined) return;
        return this.pathProps.map((_, i) => i);
    }
}

export class VariableDataProvider implements TreeDataProvider<number>, PyteaTreeProvider {
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
        return [];
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
        return [];
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
        return [];
    }
}

export class FtpExplorer {
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
        this._hardCtrTree = window.createTreeView('softConstraints', { treeDataProvider: providers[2] });
        this._pathCtrTree = window.createTreeView('pathConstraints', { treeDataProvider: providers[3] });
        this._softCtrTree = window.createTreeView('hardConstraints', { treeDataProvider: providers[4] });
        this._providers = providers;

        commands.registerCommand('pytea.selectPath', (pathId) => {
            if (pathId < 0 || pathId >= this._pathProps.length) {
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

// export class PathTreeProvider implements TreeDataProvider<number> {}

// function aNodeWithIdTreeDataProvider(msg: string): TreeDataProvider<{ key: string }> {
//     return {
//         getChildren: (element?: { key: string }): { key: string }[] => {
//             if (!element) return [{ key: msg }];
//             else if (element.key.length > 20) return [];
//             else return [{ key: msg.repeat(element.key.length + 1) }];
//         },
//         getTreeItem: (element: { key: string }): TreeItem => {
//             const treeItem = getTreeItem(element.key);
//             treeItem.id = element.key;
//             return treeItem;
//         },
//     };
// }

// function getTreeItem(key: string): TreeItem {
//     return {
//         label: <any>{ label: key },
//         tooltip: `Tooltip for ${key}`,
//         collapsibleState: TreeItemCollapsibleState.Collapsed,
//     };
// }
