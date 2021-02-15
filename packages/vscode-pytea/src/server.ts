/*
 * server.ts
 * Copyright (c) Seoul National University
 * Licensed under the MIT license.
 *
 * Implements pytea language server.
 */

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
    WorkspaceFolder,
} from 'vscode-languageserver/node';

import { AnalysisResults } from 'pyright-internal/analyzer/analysis';
import { isPythonBinary } from 'pyright-internal/analyzer/pythonPathUtils';
import { AnalyzerService } from 'pyright-internal/analyzer/service';
import { getNestedProperty } from 'pyright-internal/common/collectionUtils';
import { ConsoleInterface, ConsoleWithLogLevel, LogLevel, NullConsole } from 'pyright-internal/common/console';
import { isString } from 'pyright-internal/common/core';
import { createDeferred } from 'pyright-internal/common/deferred';
import { Diagnostic as AnalyzerDiagnostic, DiagnosticCategory } from 'pyright-internal/common/diagnostic';
import { DiagnosticRule } from 'pyright-internal/common/diagnosticRules';
import { createFromRealFileSystem, FileSystem } from 'pyright-internal/common/fileSystem';
import { convertPathToUri, convertUriToPath, resolvePaths } from 'pyright-internal/common/pathUtils';
import { Position } from 'pyright-internal/common/textRange';
import { ServerOptions, ServerSettings, WorkspaceServiceInstance } from 'pyright-internal/languageServerBase';
import { AnalyzerServiceExecutor } from 'pyright-internal/languageService/analyzerServiceExecutor';
import { convertHoverResults } from 'pyright-internal/languageService/hoverProvider';

import { defaultOptions, PyteaLogLevel, PyteaOptions } from 'pytea/service/pyteaOptions';
import { PyteaService } from 'pytea/service/pyteaService';

import { PyteaCommandController } from './commandController';
import { PyteaWorkspaceInstance, PyteaWorkspaceMap } from './workspaceMap';
import { buildPyteaOption } from 'pytea/service/pyteaUtils';

export interface PyteaServerSettings {
    pyrightSettings: ServerSettings;
    pyteaOptions: PyteaOptions;
}

export class PyteaServer {
    protected _connection: Connection = createConnection({});
    protected _workspaceMap: PyteaWorkspaceMap;
    protected _hasConfigurationCapability = false;
    protected _hasWorkspaceFoldersCapability = false;
    protected _hoverContentFormat: MarkupKind = MarkupKind.PlainText;
    protected _defaultClientConfig: any;

    private _serverOptions: ServerOptions;
    private _controller: PyteaCommandController;

    // Global root path - the basis for all global settings.
    rootPath = '';

    // File system abstraction.
    fs: FileSystem;

    readonly console: ConsoleInterface;

    constructor() {
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
            this.updateSettingsForWorkspace(workspace).ignoreErrors();
        });
    }

    analyze(entryPath: string) {
        const workspace = this._workspaceMap.getWorkspaceForFile(entryPath);

        const options = workspace.pyteaOptions;
        options.entryPath = resolvePaths(workspace.rootPath, this.expandPathVariables(workspace.rootPath, entryPath));

        const tempOptions = buildPyteaOption(undefined, workspace.rootPath, options);
        if (typeof tempOptions === 'string') {
            this.console.error(`Found error while setting Pytea options: ${tempOptions}`);
            return;
        }

        workspace.pyteaOptions = tempOptions;
        const pyteaService = workspace.pyteaService;
        pyteaService.setOptions(tempOptions);

        if (!pyteaService.validate()) {
            return;
        }

        const errMsg = pyteaService.translateAll(options.entryPath);
        if (errMsg) {
            this.console.error(`Found error while translating Python scripts: ${errMsg}`);
        }

        const result = pyteaService.analyze();
        this.console.info(`completed analyzing. found ${result?.getList()?.count ?? 0} paths`);
        // TODO: resulting!
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

            const pythonPyteaSection = await this.getConfiguration(workspace.rootUri, 'python.analysis');
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

        // The workspace is now open for business.
        workspace.isInitialized.resolve(true);
    }

    protected executeCommand(params: ExecuteCommandParams, token: CancellationToken): Promise<any> {
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

        this._connection.onHover(async (params, token) => {
            const filePath = convertUriToPath(params.textDocument.uri);

            const position: Position = {
                line: params.position.line,
                character: params.position.character,
            };

            // TODO: add pytea result.
            const workspace = await this.getWorkspaceForFile(filePath);
            const hoverResults = workspace.serviceInstance.getHoverForPosition(
                filePath,
                position,
                this._hoverContentFormat,
                token
            );
            return convertHoverResults(this._hoverContentFormat, hoverResults);
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
            await this.executeCommand(params, token);
        });
    }

    protected initialize(params: InitializeParams): InitializeResult {
        this.rootPath = params.rootPath || '';

        const capabilities = params.capabilities;
        this._hasWorkspaceFoldersCapability = !!capabilities.workspace?.workspaceFolders;
        this._hoverContentFormat = this._getCompatibleMarkupKind(capabilities.textDocument?.hover?.contentFormat);

        // Create a service instance for each of the workspace folders.
        if (params.workspaceFolders) {
            params.workspaceFolders.forEach((folder) => {
                const path = convertUriToPath(folder.uri);
                this._workspaceMap.set(path, this.createWorkspaceServiceInstance(folder, path));
            });
        } else if (params.rootPath) {
            this._workspaceMap.set(params.rootPath, this.createWorkspaceServiceInstance(undefined, params.rootPath));
        }

        const result: InitializeResult = {
            capabilities: {
                hoverProvider: { workDoneProgress: true },
                documentHighlightProvider: { workDoneProgress: true },
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

    private _getCompatibleMarkupKind(clientSupportedFormats: MarkupKind[] | undefined) {
        const serverSupportedFormats = [MarkupKind.PlainText, MarkupKind.Markdown];

        for (const format of clientSupportedFormats ?? []) {
            if (serverSupportedFormats.includes(format)) {
                return format;
            }
        }

        return MarkupKind.PlainText;
    }

    protected getDocumentationUrlForDiagnosticRule(rule: string): string | undefined {
        // For now, return the same URL for all rules. We can separate these
        // in the future.
        return 'https://github.com/microsoft/pyright/blob/master/docs/configuration.md';
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
