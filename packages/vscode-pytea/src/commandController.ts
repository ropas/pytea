/*
 * commandController.ts
 * Copyright (c) Seoul National University
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo
 *
 * Implements language server commands execution functionality.
 */

import { CancellationToken, ExecuteCommandParams, ResponseError } from 'vscode-languageserver';

import { ServerCommand } from 'pyright-internal/commands/commandController';

import { PyteaCommands } from './commands';
import { PyteaServer } from './server';
import { Uri } from 'vscode';

export class RestartServerCommand implements ServerCommand {
    constructor(private _ls: PyteaServer) {}

    async execute(cmdParams: ExecuteCommandParams): Promise<any> {
        this._ls.restart();
    }
}

export class AnalyzeFileCommand implements ServerCommand {
    constructor(private _ls: PyteaServer) {}

    async execute(cmdParams: ExecuteCommandParams): Promise<any> {
        const args = cmdParams.arguments;
        if (args && args[0] && typeof args[0] === 'object') {
            const entryPath = (args[0] as Uri).fsPath;
            this._ls.analyze(entryPath);
        }
    }
}

export class PyteaCommandController implements ServerCommand {
    private _restartServer: RestartServerCommand;
    private _analyzeFile: AnalyzeFileCommand;

    constructor(ls: PyteaServer) {
        this._restartServer = new RestartServerCommand(ls);
        this._analyzeFile = new AnalyzeFileCommand(ls);
    }

    async execute(cmdParams: ExecuteCommandParams, token: CancellationToken): Promise<any> {
        switch (cmdParams.command) {
            case PyteaCommands.restartServer: {
                return this._restartServer.execute(cmdParams);
            }

            case PyteaCommands.analyzeFile: {
                return this._analyzeFile.execute(cmdParams);
            }

            default: {
                return new ResponseError<string>(1, 'Unsupported command');
            }
        }
    }
}
