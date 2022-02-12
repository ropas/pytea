/*
 * server.ts
 *
 * Implements pyright language server.
 */

import { SVErrorLevel, SVType } from 'pytea/backend/sharpValues';
import { ExecutionPath, ExecutionPathStatus } from 'pytea/service/executionPaths';
import { defaultOptions, PyteaLogLevel, PyteaOptions } from 'pytea/service/pyteaOptions';
import { PyteaService } from 'pytea/service/pyteaService';
import { buildPyteaOption } from 'pytea/service/pyteaUtils';
import {
    CancellationToken,
    CodeAction,
    CodeActionKind,
    CodeActionParams,
    Command,
    Connection,
    ExecuteCommandParams,
    WorkDoneProgressServerReporter,
} from 'vscode-languageserver';

import { AnalysisResults } from 'pyright-internal/analyzer/analysis';
import { ImportResolver } from 'pyright-internal/analyzer/importResolver';
import { isPythonBinary } from 'pyright-internal/analyzer/pythonPathUtils';
import { BackgroundAnalysis } from 'pyright-internal/backgroundAnalysis';
import { BackgroundAnalysisBase } from 'pyright-internal/backgroundAnalysisBase';
import { getCancellationFolderName } from 'pyright-internal/common/cancellationUtils';
import { ConfigOptions } from 'pyright-internal/common/configOptions';
import { ConsoleWithLogLevel, LogLevel } from 'pyright-internal/common/console';
import { isDebugMode, isString } from 'pyright-internal/common/core';
import { Diagnostic as AnalyzerDiagnostic, DiagnosticCategory } from 'pyright-internal/common/diagnostic';
import { FileDiagnostics } from 'pyright-internal/common/diagnosticSink';
import { FileBasedCancellationProvider } from 'pyright-internal/common/fileBasedCancellationUtils';
import { FileSystem } from 'pyright-internal/common/fileSystem';
import { FullAccessHost } from 'pyright-internal/common/fullAccessHost';
import { Host } from 'pyright-internal/common/host';
import { resolvePaths } from 'pyright-internal/common/pathUtils';
import { ProgressReporter } from 'pyright-internal/common/progressReporter';
import { createFromRealFileSystem, WorkspaceFileWatcherProvider } from 'pyright-internal/common/realFileSystem';
import { LanguageServerBase, ServerSettings, WorkspaceServiceInstance } from 'pyright-internal/languageServerBase';
import { WorkspaceMap } from 'pyright-internal/workspaceMap';

import { PyteaCommandController } from './commandController';

const maxAnalysisTimeInForeground = { openFilesTimeInMs: 50, noOpenFilesTimeInMs: 200 };

export interface PyteaWorkspaceInstance extends WorkspaceServiceInstance {
    pyteaService: PyteaService;
    pyteaOptions: PyteaOptions;
    selectedPath: number;
    paths: ExecutionPath[];
}

export class PyteaServer extends LanguageServerBase {
    private _controller: PyteaCommandController;
    private _selectedWorkspace?: PyteaWorkspaceInstance;
    private _lastDiagMap?: Map<string, AnalyzerDiagnostic[]>;

    constructor(connection: Connection) {
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        const version = require('../package.json').version || '';

        // When executed from CLI command (pyright-langserver), __rootDirectory is
        // already defined. When executed from VSCode extension, rootDirectory should
        // be __dirname.
        const rootDirectory = (global as any).__rootDirectory || __dirname;

        const console = new ConsoleWithLogLevel(connection.console);
        const workspaceMap = new WorkspaceMap();
        const fileWatcherProvider = new WorkspaceFileWatcherProvider(workspaceMap, console);
        const fileSystem = createFromRealFileSystem(console, fileWatcherProvider);

        super(
            {
                productName: 'Pytea',
                rootDirectory,
                version,
                workspaceMap,
                fileSystem: fileSystem,
                fileWatcherProvider,
                cancellationProvider: new FileBasedCancellationProvider('bg'),
                maxAnalysisTimeInForeground,
                supportedCodeActions: [CodeActionKind.QuickFix, CodeActionKind.SourceOrganizeImports],
            },
            connection,
            console
        );

        this._controller = new PyteaCommandController(this);
    }

    async getSettings(workspace: WorkspaceServiceInstance): Promise<ServerSettings> {
        const serverSettings: ServerSettings = {
            watchForSourceChanges: true,
            watchForLibraryChanges: true,
            watchForConfigChanges: true,
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
            ...defaultOptions,
            logLevel: 'result-only',
        };

        try {
            const pythonSection = await this.getConfiguration(workspace.rootUri, 'python');
            if (pythonSection) {
                const pythonPath = pythonSection.pythonPath;
                if (pythonPath && isString(pythonPath) && !isPythonBinary(pythonPath)) {
                    serverSettings.pythonPath = resolvePaths(
                        workspace.rootPath,
                        this.expandPathVariables(workspace.rootPath, pythonPath)
                    );
                }

                const venvPath = pythonSection.venvPath;

                if (venvPath && isString(venvPath)) {
                    serverSettings.venvPath = resolvePaths(
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
        return serverSettings;
    }

    createBackgroundAnalysis(): BackgroundAnalysisBase | undefined {
        if (isDebugMode() || !getCancellationFolderName()) {
            // Don't do background analysis if we're in debug mode or an old client
            // is used where cancellation is not supported.
            return undefined;
        }

        return new BackgroundAnalysis(this.console);
    }

    async analyze(entryPath: string) {
        this.console.info(`analyzing ${entryPath}...`);

        this._clearDiagnostics();

        const ws = this._workspaceMap.getWorkspaceForFile(this, entryPath);
        this._selectedWorkspace = ws as PyteaWorkspaceInstance;
        const workspace = this._selectedWorkspace;

        // inject pytea service
        if (!('pyteaService' in ws)) {
            workspace.pyteaOptions = defaultOptions;
            workspace.paths = [];
            workspace.selectedPath = 0;
            workspace.pyteaService = new PyteaService(ws.serviceInstance, undefined, this.console);
        }

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

        // this.console.info(`options: ${inspect(options)}`);
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

            // TODO: resolve rejected
            const result = await pyteaService.analyze();

            const success = result.getList();
            const stopped = result.getStopped();
            const failed = result.getFailed();
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
        const service = this._selectedWorkspace?.pyteaService;
        const paths = this._selectedWorkspace?.paths;

        // TODO: pop-up error message
        if (!(service && paths && paths[pathId])) {
            return;
        }

        this._clearDiagnostics();

        const currPath = paths[pathId];
        const currProps = currPath.props;
        const ctx = currPath.ctx;
        const diagMap: Map<string, AnalyzerDiagnostic[]> = new Map();

        function severityMap(level: SVErrorLevel): DiagnosticCategory {
            switch (level) {
                case SVErrorLevel.Error:
                    return DiagnosticCategory.Error;
                case SVErrorLevel.Warning:
                    return DiagnosticCategory.Warning;
                case SVErrorLevel.Log:
                    return DiagnosticCategory.Information;
            }
        }

        // TODO: clear error log in diagnostics
        // TODO: for a succeed calculation, return empty array in diagnostics
        // TODO: clear every diagnostic on new analysis
        ctx.logs.forEach((log) => {
            if (log.type === SVType.Error && log.level === SVErrorLevel.Warning) {
                const sourceRange = service.getSourceRange(log.source);
                if (sourceRange) {
                    const [filePath, range] = sourceRange;
                    if (!diagMap.has(filePath)) {
                        diagMap.set(filePath, []);
                    }
                    const diagnostics = diagMap.get(filePath)!;
                    diagnostics.push(new AnalyzerDiagnostic(severityMap(log.level), log.reason, range));
                }
            }
        });

        if (currProps.status === ExecutionPathStatus.Stopped || currProps.status === ExecutionPathStatus.Failed) {
            const retVal = ctx.retVal;
            if (typeof retVal === 'object') {
                const sourceRange = service.getSourceRange(retVal.source);
                if (sourceRange) {
                    const [filePath, range] = sourceRange;
                    if (!diagMap.has(filePath)) {
                        diagMap.set(filePath, []);
                    }
                    const diagnostics = diagMap.get(filePath)!;
                    const reason = retVal.type === SVType.Error ? retVal.reason : 'unknown error';
                    diagnostics.push(new AnalyzerDiagnostic(DiagnosticCategory.Error, reason, range));
                }
            }
        }

        diagMap.forEach((diagnostics, filePath) => {
            const fileDiag: FileDiagnostics = { filePath, version: undefined, diagnostics };
            const publishParams = this.convertDiagnostics(fileDiag);

            for (const param of publishParams) {
                this._connection.sendDiagnostics(param);
            }
        });

        this._lastDiagMap = diagMap;
    }

    protected override onAnalysisCompletedHandler(results: AnalysisResults): void {
        // Ignore super send diagnostics
        const emptyResult = { ...results, diagnostics: [] };
        super.onAnalysisCompletedHandler(emptyResult);
    }

    protected override createHost() {
        return new FullAccessHost(this.fs);
    }

    protected override createImportResolver(fs: FileSystem, options: ConfigOptions, host: Host): ImportResolver {
        return new ImportResolver(fs, options, host);
    }

    protected executeCommand(params: ExecuteCommandParams, token: CancellationToken): Promise<any> {
        return this._controller.execute(params, token);
    }

    protected isLongRunningCommand(command: string): boolean {
        return this._controller.isLongRunningCommand(command);
    }

    protected async executeCodeAction(
        params: CodeActionParams,
        token: CancellationToken
    ): Promise<(Command | CodeAction)[] | undefined | null> {
        // this.recordUserInteractionTime();

        // const filePath = this._uriParser.decodeTextDocumentUri(params.textDocument.uri);
        // this.console.log(`codeaction on ${filePath}`);
        // const workspace = await this.getWorkspaceForFile(filePath);
        // return CodeActionProvider.getCodeActionsForPosition(workspace, filePath, params.range, token);

        return null;
    }

    protected createProgressReporter(): ProgressReporter {
        // The old progress notifications are kept for backwards compatibility with
        // clients that do not support work done progress.

        let workDoneProgress: Promise<WorkDoneProgressServerReporter> | undefined;
        return {
            isEnabled: (data: AnalysisResults) => true,
            begin: () => {
                if (this.client.hasWindowProgressCapability) {
                    workDoneProgress = this._connection.window.createWorkDoneProgress();
                    workDoneProgress
                        .then((progress) => {
                            progress.begin('');
                        })
                        .ignoreErrors();
                } else {
                    this._connection.sendNotification('pytea/beginProgress');
                }
            },
            report: (message: string) => {
                if (workDoneProgress) {
                    workDoneProgress
                        .then((progress) => {
                            progress.report(message);
                        })
                        .ignoreErrors();
                } else {
                    this._connection.sendNotification('pytea/reportProgress', message);
                }
            },
            end: () => {
                if (workDoneProgress) {
                    workDoneProgress
                        .then((progress) => {
                            progress.done();
                        })
                        .ignoreErrors();
                    workDoneProgress = undefined;
                } else {
                    this._connection.sendNotification('pytea/endProgress');
                }
            },
        };
    }

    private _clearDiagnostics() {
        this._lastDiagMap?.forEach((diagnostics, filePath) => {
            const fileDiag: FileDiagnostics = { filePath, version: undefined, diagnostics: [] };
            const publishParams = this.convertDiagnostics(fileDiag);

            for (const param of publishParams) {
                this._connection.sendDiagnostics(param);
            }
        });
    }
}
