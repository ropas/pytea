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
import { convertUriToPath } from 'pyright-internal/common/pathUtils';

import { PyteaCommands } from './commands';
import { PyteaServer } from './server';

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
        if (args && args[0]) {
            const entryPath = convertUriToPath(args[0]);
            return this._ls.analyze(entryPath);
        }
    }
}

export class PyteaCommandController implements ServerCommand {
    private _restartServer: RestartServerCommand;
    private _analyzeFile: AnalyzeFileCommand;

    constructor(private ls: PyteaServer) {
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

            case PyteaCommands.selectPath: {
                return this.ls.selectPath(cmdParams.arguments![0]);
            }

            default: {
                return new ResponseError<string>(1, 'Unsupported command');
            }
        }
    }
}
