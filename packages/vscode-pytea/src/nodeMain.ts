/*
 * nodeMain.ts
 * Copyright (c) Seoul National University.
 * Licensed under the MIT license.
 *
 * Provides the main entrypoint to the server when running in Node.
 */

// import { BackgroundAnalysisRunner } from './backgroundAnalysis';
import { Connection, ConnectionOptions } from 'vscode-languageserver';
import { createConnection } from 'vscode-languageserver/node';
import { isMainThread } from 'worker_threads';

import { getCancellationStrategyFromArgv } from 'pyright-internal/common/fileBasedCancellationUtils';

import { PyteaServer } from './nodeServer';

export function run(runServer: (connection: Connection) => void, runBackgroundThread: () => void) {
    if (process.env.NODE_ENV === 'production') {
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        require('source-map-support').install();
    }

    if (isMainThread) {
        console.log('running on main thread');
        runServer(createConnection(getConnectionOptions()));
    } else {
        console.log('running on background thread');
        runBackgroundThread();
    }
}

export function getConnectionOptions(): ConnectionOptions {
    return { cancellationStrategy: getCancellationStrategyFromArgv(process.argv) };
}

export function main() {
    run(
        (conn) => new PyteaServer(conn),
        () => {
            // const runner = new BackgroundAnalysisRunner();
            // runner.start();
        }
    );
}
