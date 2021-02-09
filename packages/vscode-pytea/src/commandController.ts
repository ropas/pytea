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

export class RestartServerCommand implements ServerCommand {
    constructor(private _ls: PyteaServer) {}

    async execute(cmdParams: ExecuteCommandParams): Promise<any> {
        this._ls.restart();
    }
}

export class PyteaCommandController implements ServerCommand {
    private _restartServer: RestartServerCommand;

    constructor(ls: PyteaServer) {
        this._restartServer = new RestartServerCommand(ls);
    }

    async execute(cmdParams: ExecuteCommandParams, token: CancellationToken): Promise<any> {
        switch (cmdParams.command) {
            case PyteaCommands.restartServer: {
                return this._restartServer.execute(cmdParams);
            }

            default: {
                return new ResponseError<string>(1, 'Unsupported command');
            }
        }
    }
}
