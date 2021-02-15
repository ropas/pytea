/*
 * workspaceMap.ts
 *
 * Workspace management related functionality.
 */

import { PyteaService } from 'pytea/service/pyteaService';

import { createDeferred } from 'pyright-internal/common/deferred';
import { WorkspaceServiceInstance } from 'pyright-internal/languageServerBase';

import { PyteaServer } from './server';
import { defaultOptions, PyteaOptions } from 'pytea/service/pyteaOptions';

export interface PyteaWorkspaceInstance extends WorkspaceServiceInstance {
    pyteaService: PyteaService;
    pyteaOptions: PyteaOptions;
}

export class PyteaWorkspaceMap extends Map<string, PyteaWorkspaceInstance> {
    private _defaultWorkspacePath = '<default>';

    constructor(private _ls: PyteaServer) {
        super();
    }

    getNonDefaultWorkspaces(): PyteaWorkspaceInstance[] {
        const workspaces: PyteaWorkspaceInstance[] = [];
        this.forEach((workspace) => {
            if (workspace.rootPath) {
                workspaces.push(workspace);
            }
        });

        return workspaces;
    }

    getWorkspaceForFile(filePath: string): PyteaWorkspaceInstance {
        let bestRootPath: string | undefined;
        let bestInstance: PyteaWorkspaceInstance | undefined;

        this.forEach((workspace) => {
            if (workspace.rootPath) {
                // Is the file is under this workspace folder?
                if (filePath.startsWith(workspace.rootPath)) {
                    // Is this the fist candidate? If not, is this workspace folder
                    // contained within the previous candidate folder? We always want
                    // to select the innermost folder, since that overrides the
                    // outer folders.
                    if (bestRootPath === undefined || workspace.rootPath.startsWith(bestRootPath)) {
                        bestRootPath = workspace.rootPath;
                        bestInstance = workspace;
                    }
                }
            }
        });

        // If there were multiple workspaces or we couldn't find any,
        // create a default one to use for this file.
        if (bestInstance === undefined) {
            let defaultWorkspace = this.get(this._defaultWorkspacePath);
            if (!defaultWorkspace) {
                // If there is only one workspace, use that one.
                const workspaceNames = [...this.keys()];
                if (workspaceNames.length === 1) {
                    return this.get(workspaceNames[0])!;
                }

                // Create a default workspace for files that are outside
                // of all workspaces.
                const pyrightService = this._ls.createAnalyzerService(this._defaultWorkspacePath);
                const pyteaService = new PyteaService(pyrightService, undefined, this._ls.console);

                defaultWorkspace = {
                    workspaceName: '',
                    rootPath: '',
                    rootUri: '',
                    serviceInstance: pyrightService,
                    pyteaService: pyteaService,
                    pyteaOptions: defaultOptions,
                    disableLanguageServices: false,
                    disableOrganizeImports: false,
                    isInitialized: createDeferred<boolean>(),
                };
                this.set(this._defaultWorkspacePath, defaultWorkspace);
                this._ls.updateSettingsForWorkspace(defaultWorkspace).ignoreErrors();
            }

            return defaultWorkspace;
        }

        return bestInstance;
    }
}
