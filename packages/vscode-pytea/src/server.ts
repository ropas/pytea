/*
 * server.ts
 * Copyright (c) Seoul National University
 * Licensed under the MIT license.
 *
 * Implements pytea language server.
 */

import { ExecutionPath, ExecutionPathStatus } from 'pytea/service/executionPaths';
import { defaultOptions, PyteaLogLevel, PyteaOptions } from 'pytea/service/pyteaOptions';
import { PyteaService } from 'pytea/service/pyteaService';
import { buildPyteaOption } from 'pytea/service/pyteaUtils';
import { inspect } from 'util';
import { CancellationTokenSource } from 'vscode';
import {
    CancellationToken,
    ConfigurationItem,
    Connection,
    createConnection,
    Diagnostic,
    DiagnosticRelatedInformation,
    DiagnosticSeverity,
    DiagnosticTag,
    ExecuteCommandParams,
    InitializeParams,
    InitializeResult,
    Location,
    MarkupKind,
    WorkDoneProgressReporter,
    WorkspaceFolder,
} from 'vscode-languageserver/node';
import { TextDocumentSyncKind } from 'vscode-languageserver-protocol';

import { AnalysisResults } from 'pyright-internal/analyzer/analysis';
import { isPythonBinary } from 'pyright-internal/analyzer/pythonPathUtils';
import { AnalyzerService } from 'pyright-internal/analyzer/service';
import { CancelAfter } from 'pyright-internal/common/cancellationUtils';
import { getNestedProperty } from 'pyright-internal/common/collectionUtils';
import { ConsoleInterface, ConsoleWithLogLevel, LogLevel, NullConsole } from 'pyright-internal/common/console';
import { isString } from 'pyright-internal/common/core';
import { createDeferred } from 'pyright-internal/common/deferred';
import { Diagnostic as AnalyzerDiagnostic, DiagnosticCategory } from 'pyright-internal/common/diagnostic';
import { createFromRealFileSystem, FileSystem } from 'pyright-internal/common/fileSystem';
import { convertPathToUri, convertUriToPath, resolvePaths } from 'pyright-internal/common/pathUtils';
import { ServerOptions, ServerSettings } from 'pyright-internal/languageServerBase';
import { AnalyzerServiceExecutor } from 'pyright-internal/languageService/analyzerServiceExecutor';
import { Localizer } from 'pyright-internal/localization/localize';

import { PyteaCommandController } from './commandController';
import { PyteaWorkspaceInstance, PyteaWorkspaceMap } from './workspaceMap';

export interface PyteaServerSettings {
    pyrightSettings: ServerSettings;
    pyteaOptions: PyteaOptions;
}

export class PyteaServer {
    protected _connection: Connection;
    protected _workspaceMap: PyteaWorkspaceMap;
    protected _hasConfigurationCapability = false;
    protected _hasWorkspaceFoldersCapability = false;
    protected _hoverContentFormat: MarkupKind = MarkupKind.PlainText;
    protected _defaultClientConfig: any;

    private _serverOptions: ServerOptions;
    private _controller: PyteaCommandController;

    // We support running only one command at a time.
    private _pendingCommandCancellationSource: CancellationTokenSource | undefined;

    // Global root path - the basis for all global settings.
    rootPath = '';

    // File system abstraction.
    fs: FileSystem;

    readonly console: ConsoleInterface;

    constructor() {
        this._connection = createConnection();

        // eslint-disable-next-line @typescript-eslint/no-var-requires
        const version = require('../package.json').version || '';

        // When executed from CLI command (pyright-langserver), __rootDirectory is
        // already defined. When executed from VSCode extension, rootDirectory should
        // be __dirname.
        const rootDirectory = (global as any).__rootDirectory || __dirname;

        this.console = new ConsoleWithLogLevel(this._connection.console);

        this.console.info(`Pytea language server ${version && version + ' '}starting`);

        this.fs = createFromRealFileSystem(this.console);

        // Set the working directory to a known location within
        // the extension directory. Otherwise the execution of
        // python can have unintended and surprising results.
        const moduleDirectory = this.fs.getModulePath();
        if (moduleDirectory) {
            this.fs.chdir(moduleDirectory);
        }

        this._serverOptions = {
            productName: 'Pytea',
            rootDirectory,
            version,
        };

        // Stash the base directory into a global variable.
        (global as any).__rootDirectory = rootDirectory;
        this.console.info(`Server root directory: ${rootDirectory}`);

        // Create workspace map.
        this._workspaceMap = new PyteaWorkspaceMap(this);

        // Set up callbacks.
        this.setupConnection();

        // Connect command controller.
        this._controller = new PyteaCommandController(this);

        // Listen on the connection.
        this._connection.listen();
    }

    // Creates a service instance that's used for analyzing a
    // program within a workspace.
    createAnalyzerService(name: string): AnalyzerService {
        this.console.log(`Starting service instance "${name}"`);

        // ignore original pyright output.
        const output = new NullConsole();
        const service = new AnalyzerService(name, this.fs, output);

        service.setCompletionCallback((results) => this.onAnalysisCompletedHandler(results));

        return service;
    }

    updateSettingsForAllWorkspaces(): void {
        this._workspaceMap.forEach((workspace) => {
            this.updateSettingsForWorkspace(workspace).catch(() => {
                // ignore errors
            });
        });
    }

    analyze(entryPath: string) {
        this.console.info(`analyzing ${entryPath}...`);

        const workspace = this._workspaceMap.getWorkspaceForFile(entryPath);

        const baseOptions = workspace.pyteaOptions;
        baseOptions.entryPath = resolvePaths(
            workspace.rootPath,
            this.expandPathVariables(workspace.rootPath, entryPath)
        );

        const options = buildPyteaOption(undefined, workspace.rootPath, baseOptions);
        if (typeof options === 'string') {
            this.console.error(`Found error while setting Pytea options: ${options}`);
            return;
        }

        workspace.pyteaOptions = options;
        const pyteaService = workspace.pyteaService;
        pyteaService.setOptions(options);

        this.console.info(`options: ${inspect(options)}`);
        if (!pyteaService.validate()) {
            return;
        }

        const errMsg = pyteaService.translateAll(baseOptions.entryPath);
        if (errMsg) {
            this.console.error(`Found error while translating Python scripts: ${errMsg}`);
            return;
        }

        try {
            // TODO: remove global reference
            PyteaService.setGlobalService(pyteaService);
            const result = pyteaService.analyze();
            const success = result?.getList();
            const stopped = result?.getStopped();
            const failed = result?.getFailed();
            this.console.info(
                `Analyzing completed. Found ${success?.count() ?? 0} success paths / ${
                    failed?.count() ?? 0
                } failed paths`
            );

            if (result) {
                const executionPaths: ExecutionPath[] = [];
                let id = 0;
                success?.forEach((p) => {
                    const path = new ExecutionPath(
                        p,
                        id,
                        this.rootPath,
                        ExecutionPathStatus.Success,
                        pyteaService.getPathStore()
                    );
                    executionPaths.push(path);
                    id++;
                });

                stopped?.forEach((p) => {
                    const path = new ExecutionPath(
                        p,
                        id,
                        this.rootPath,
                        ExecutionPathStatus.Stopped,
                        pyteaService.getPathStore()
                    );
                    executionPaths.push(path);
                    id++;
                });

                failed?.forEach((p) => {
                    const path = new ExecutionPath(
                        p,
                        id,
                        this.rootPath,
                        ExecutionPathStatus.Failed,
                        pyteaService.getPathStore()
                    );
                    executionPaths.push(path);
                    id++;
                });

                setTimeout(() => {
                    //  async print log
                    pyteaService.printLog(result);
                }, 10);

                workspace.paths = executionPaths;

                return executionPaths.map((p) => p.props);
            } else {
                this.console.error('Analyzer returned undefined');
            }
        } catch (e) {
            this.console.error(`Found error while analyzing ${options.entryPath}\n  ${e}`);
        }
    }

    selectPath(pathId: number): void {
        // const error = p.retVal;
        // const sourceRange = pyteaService.getSourceRange(error.source);
        // if (sourceRange) {
        //     const [filePath, range] = sourceRange;
        //     this._connection.sendDiagnostics({
        //         uri: convertPathToUri(filePath),
        //         diagnostics: this._convertDiagnostics([
        //             new AnalyzerDiagnostic(DiagnosticCategory.Warning, error.reason, range),
        //         ]),
        //     });
        // }
    }

    restart() {
        this._workspaceMap.forEach((workspace) => {
            workspace.serviceInstance.restart();
        });
    }

    async getWorkspaceForFile(filePath: string): Promise<PyteaWorkspaceInstance> {
        const workspace = this._workspaceMap.getWorkspaceForFile(filePath);
        await workspace.isInitialized.promise;
        return workspace;
    }

    async getSettings(workspace: PyteaWorkspaceInstance): Promise<PyteaServerSettings> {
        const pyrightSettings: ServerSettings = {
            watchForSourceChanges: true,
            watchForLibraryChanges: false,
            openFilesOnly: true,
            useLibraryCodeForTypes: false,
            disableLanguageServices: true,
            disableOrganizeImports: true,
            typeCheckingMode: 'basic',
            diagnosticSeverityOverrides: {},
            logLevel: LogLevel.Log,
            autoImportCompletions: false,
        };

        const pyteaOptions: PyteaOptions = {
            configPath: '',
            pyteaLibPath: '',
            entryPath: '',
            pythonCmdArgs: {},
            pythonSubcommand: '',
            logLevel: 'result-only',
            immediateConstraintCheck: true,
            ignoreAssert: false,
            extractIR: false,
            variableRange: {},
        };

        try {
            const pythonSection = await this.getConfiguration(workspace.rootUri, 'python');
            if (pythonSection) {
                const pythonPath = pythonSection.pythonPath;
                if (pythonPath && isString(pythonPath) && !isPythonBinary(pythonPath)) {
                    pyrightSettings.pythonPath = resolvePaths(
                        workspace.rootPath,
                        this.expandPathVariables(workspace.rootPath, pythonPath)
                    );
                }

                const venvPath = pythonSection.venvPath;

                if (venvPath && isString(venvPath)) {
                    pyrightSettings.venvPath = resolvePaths(
                        workspace.rootPath,
                        this.expandPathVariables(workspace.rootPath, venvPath)
                    );
                }
            }

            const pythonPyteaSection = await this.getConfiguration(workspace.rootUri, 'python.pytea');
            if (pythonPyteaSection) {
                const configPath = pythonPyteaSection.configPath;
                if (configPath && typeof configPath === 'string') {
                    pyteaOptions.configPath = resolvePaths(
                        workspace.rootPath,
                        this.expandPathVariables(workspace.rootPath, configPath)
                    );
                }

                const pyteaLibPath = pythonPyteaSection.pyteaLibraryPath;
                if (pyteaLibPath && typeof pyteaLibPath === 'string') {
                    pyteaOptions.pyteaLibPath = resolvePaths(
                        workspace.rootPath,
                        this.expandPathVariables(workspace.rootPath, pyteaLibPath)
                    );
                }

                const pythonCmdArgs = pythonPyteaSection.pythonCommandLineArguments;
                if (pythonCmdArgs && typeof pythonCmdArgs === 'object') {
                    pyteaOptions.pythonCmdArgs = pythonCmdArgs;
                }

                const pythonSubcommand = pythonPyteaSection.pythonSubcommand;
                if (pythonSubcommand && typeof pythonSubcommand === 'string') {
                    pyteaOptions.pythonSubcommand = pythonSubcommand;
                }

                const immediateConstraintCheck = pythonPyteaSection.immediateConstraintCheck;
                if (immediateConstraintCheck !== undefined) {
                    pyteaOptions.immediateConstraintCheck = !!immediateConstraintCheck;
                }

                const ignoreAssert = pythonPyteaSection.ignoreAssert;
                if (ignoreAssert !== undefined) {
                    pyteaOptions.ignoreAssert = !!ignoreAssert;
                }

                const variableRange = pythonPyteaSection.variableRange;
                if (variableRange && typeof variableRange === 'object') {
                    pyteaOptions.variableRange = variableRange;
                }

                const logLevel = pythonPyteaSection.logLevel;
                const validLevels = ['none', 'result-only', 'reduced', 'full'];
                if (logLevel && typeof logLevel === 'string' && validLevels.includes(logLevel)) {
                    pyteaOptions.logLevel = logLevel as PyteaLogLevel;
                }
            }
        } catch (error) {
            this.console.error(`Error reading settings: ${error}`);
        }
        return {
            pyrightSettings,
            pyteaOptions,
        };
    }

    async updateSettingsForWorkspace(
        workspace: PyteaWorkspaceInstance,
        serverSettings?: PyteaServerSettings
    ): Promise<void> {
        serverSettings = serverSettings ?? (await this.getSettings(workspace));

        const pyrightSettings = serverSettings.pyrightSettings;

        // Set logging level first.
        (this.console as ConsoleWithLogLevel).level = pyrightSettings.logLevel ?? LogLevel.Info;

        AnalyzerServiceExecutor.runWithOptions(this.rootPath, workspace, pyrightSettings);
        workspace.disableLanguageServices = !!pyrightSettings.disableLanguageServices;
        workspace.disableOrganizeImports = !!pyrightSettings.disableOrganizeImports;
        workspace.pyteaOptions = serverSettings.pyteaOptions;
        this.console.info(`setting options to ${inspect(serverSettings.pyteaOptions)}`);

        // The workspace is now open for business.
        workspace.isInitialized.resolve(true);
    }

    protected executeCommand(params: ExecuteCommandParams, token: CancellationToken): Promise<any> {
        this.console.info(`got ${params.command}`);
        return this._controller.execute(params, token);
    }

    protected setupConnection(): void {
        // After the server has started the client sends an initialize request. The server receives
        // in the passed params the rootPath of the workspace plus the client capabilities.
        this._connection.onInitialize((params) => this.initialize(params));

        this._connection.onDidChangeConfiguration((params) => {
            this.console.log(`Received updated settings`);
            if (params?.settings) {
                this._defaultClientConfig = params?.settings;
            }
            this.updateSettingsForAllWorkspaces();
        });

        this._connection.onInitialized(() => {
            if (this._hasWorkspaceFoldersCapability) {
                this._connection.workspace.onDidChangeWorkspaceFolders((event) => {
                    event.removed.forEach((workspace) => {
                        const rootPath = convertUriToPath(workspace.uri);
                        this._workspaceMap.delete(rootPath);
                    });

                    event.added.forEach(async (workspace) => {
                        const rootPath = convertUriToPath(workspace.uri);
                        const newWorkspace = this.createWorkspaceServiceInstance(workspace, rootPath);
                        this._workspaceMap.set(rootPath, newWorkspace);
                        await this.updateSettingsForWorkspace(newWorkspace);
                    });
                });
            }
        });

        this._connection.onExecuteCommand(async (params, token, reporter) => {
            // Cancel running command if there is one.
            if (this._pendingCommandCancellationSource) {
                this._pendingCommandCancellationSource.cancel();
                this._pendingCommandCancellationSource = undefined;
            }

            // Create a progress dialog for long-running commands.
            const progress = await this._getProgressReporter(
                params.workDoneToken,
                reporter,
                Localizer.CodeAction.executingCommand()
            );
            const source = CancelAfter(token, progress.token);
            this._pendingCommandCancellationSource = source;

            try {
                return this.executeCommand(params, token);
            } finally {
                progress.reporter.done();
                source.dispose();
            }
        });
    }

    protected initialize(params: InitializeParams): InitializeResult {
        this.rootPath = params.rootPath || '';

        const capabilities = params.capabilities;
        this._hasWorkspaceFoldersCapability = !!capabilities.workspace?.workspaceFolders;

        // Create a service instance for each of the workspace folders.
        if (params.workspaceFolders) {
            params.workspaceFolders.forEach((folder) => {
                const path = convertUriToPath(folder.uri);
                this.console.info(`create ${path}`);
                this._workspaceMap.set(path, this.createWorkspaceServiceInstance(folder, path));
            });
        } else if (params.rootPath) {
            this.console.info(`create root path ${params.rootPath}`);
            this._workspaceMap.set(params.rootPath, this.createWorkspaceServiceInstance(undefined, params.rootPath));
        }

        const result: InitializeResult = {
            capabilities: {
                textDocumentSync: TextDocumentSyncKind.Incremental,
                executeCommandProvider: {
                    commands: [],
                    workDoneProgress: true,
                },
            },
        };

        return result;
    }

    protected convertLogLevel(logLevelValue?: string): LogLevel {
        if (!logLevelValue) {
            return LogLevel.Info;
        }

        switch (logLevelValue.toLowerCase()) {
            case 'error':
                return LogLevel.Error;

            case 'warning':
                return LogLevel.Warn;

            case 'information':
                return LogLevel.Info;

            case 'trace':
                return LogLevel.Log;

            default:
                return LogLevel.Info;
        }
    }

    protected createWorkspaceServiceInstance(
        workspace: WorkspaceFolder | undefined,
        rootPath: string
    ): PyteaWorkspaceInstance {
        const pyrightService = this.createAnalyzerService(workspace?.name ?? rootPath);
        const pyteaService = new PyteaService(pyrightService, undefined, this.console);
        return {
            workspaceName: workspace?.name ?? '',
            rootPath,
            rootUri: workspace?.uri ?? '',
            serviceInstance: pyrightService,
            pyteaService: pyteaService,
            pyteaOptions: defaultOptions,
            disableLanguageServices: true,
            disableOrganizeImports: true,
            isInitialized: createDeferred<boolean>(),
            selectedPath: 0,
            paths: [],
        };
    }

    protected onAnalysisCompletedHandler(results: AnalysisResults): void {
        // TODO: Set Diagnostic
        // Send the computed diagnostics to the client.
        // results.diagnostics.forEach((fileDiag) => {
        //     this._connection.sendDiagnostics({
        //         uri: convertPathToUri(fileDiag.filePath),
        //         diagnostics: this._convertDiagnostics(fileDiag.diagnostics),
        //     });
        // });
        this.console.info(`pyright analysis finished`);
    }

    protected async getConfiguration(scopeUri: string | undefined, section: string) {
        if (this._hasConfigurationCapability) {
            const item: ConfigurationItem = {
                scopeUri,
                section,
            };
            return this._connection.workspace.getConfiguration(item);
        }

        if (this._defaultClientConfig) {
            return getNestedProperty(this._defaultClientConfig, section);
        }

        return undefined;
    }

    // Expands certain predefined variables supported within VS Code settings.
    // Ideally, VS Code would provide an API for doing this expansion, but
    // it doesn't. We'll handle the most common variables here as a convenience.
    protected expandPathVariables(rootPath: string, value: string): string {
        const regexp = /\$\{(.*?)\}/g;
        return value.replace(regexp, (match: string, name: string) => {
            const trimmedName = name.trim();
            if (trimmedName === 'workspaceFolder') {
                return rootPath;
            }
            if (trimmedName === 'env:HOME' && process.env.HOME !== undefined) {
                return process.env.HOME;
            }

            return match;
        });
    }

    protected getDocumentationUrlForDiagnosticRule(rule: string): string | undefined {
        // For now, return the same URL for all rules. We can separate these
        // in the future.
        return 'https://github.com/microsoft/pyright/blob/master/docs/configuration.md';
    }

    private _getCompatibleMarkupKind(clientSupportedFormats: MarkupKind[] | undefined) {
        const serverSupportedFormats = [MarkupKind.PlainText, MarkupKind.Markdown];

        for (const format of clientSupportedFormats ?? []) {
            if (serverSupportedFormats.includes(format)) {
                return format;
            }
        }

        return MarkupKind.PlainText;
    }

    private _convertDiagnostics(diags: AnalyzerDiagnostic[]): Diagnostic[] {
        const convertedDiags: Diagnostic[] = [];

        diags.forEach((diag) => {
            const severity = convertCategoryToSeverity(diag.category);
            const rule = diag.getRule();
            const vsDiag = Diagnostic.create(diag.range, diag.message, severity, rule, this._serverOptions.productName);

            if (diag.category === DiagnosticCategory.UnusedCode) {
                vsDiag.tags = [DiagnosticTag.Unnecessary];
                vsDiag.severity = DiagnosticSeverity.Hint;
            }

            if (rule) {
                const ruleDocUrl = this.getDocumentationUrlForDiagnosticRule(rule);
                if (ruleDocUrl) {
                    vsDiag.codeDescription = {
                        href: ruleDocUrl,
                    };
                }
            }

            const relatedInfo = diag.getRelatedInfo();
            if (relatedInfo.length > 0) {
                vsDiag.relatedInformation = relatedInfo.map((info) => {
                    return DiagnosticRelatedInformation.create(
                        Location.create(convertPathToUri(info.filePath), info.range),
                        info.message
                    );
                });
            }

            convertedDiags.push(vsDiag);
        });

        function convertCategoryToSeverity(category: DiagnosticCategory) {
            switch (category) {
                case DiagnosticCategory.Error:
                    return DiagnosticSeverity.Error;
                case DiagnosticCategory.Warning:
                    return DiagnosticSeverity.Warning;
                case DiagnosticCategory.Information:
                    return DiagnosticSeverity.Information;
                case DiagnosticCategory.UnusedCode:
                    return DiagnosticSeverity.Hint;
            }
        }

        return convertedDiags;
    }

    private async _getProgressReporter(
        workDoneToken: string | number | undefined,
        clientReporter: WorkDoneProgressReporter,
        title: string
    ) {
        if (workDoneToken) {
            return { reporter: clientReporter, token: CancellationToken.None };
        }

        const serverInitiatedReporter = await this._connection.window.createWorkDoneProgress();
        serverInitiatedReporter.begin(title, undefined, undefined, true);

        return {
            reporter: serverInitiatedReporter,
            token: serverInitiatedReporter.token,
        };
    }
}

export function main() {
    if (process.env.NODE_ENV === 'production') {
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        require('source-map-support').install();
    }

    new PyteaServer();
}

// run this extension
main();
