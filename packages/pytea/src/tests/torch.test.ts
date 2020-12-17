/*
 * torch.test.ts
 * Copyright (c) Seoul National University
 * Licensed under the MIT license.
 * Author: Ho Yount Jhoo
 *
 * Unit tests for pyright type checker and type analyzer.
 */

import * as fs from 'fs';
import * as path from 'path';

import { CommandLineOptions } from 'pyright-internal/common/commandLineOptions';
import { combinePaths, normalizePath } from 'pyright-internal/common/pathUtils';

import { PytOptionsPart } from '../pyt/pytOptions';
import { PytService } from '../pyt/pytService';
import { NodeConsole } from '../pyt/pytUtils';
import * as TorchTestUtils from './torchTestUtils';

describe('Interpreter Case Test', () => {
    const cwd = path.normalize(process.cwd());

    const entryName = process.argv[process.argv.length - 1];
    // const entryName = './test/scratch.py';
    const entryPath = normalizePath(combinePaths(cwd, entryName));

    if (!fs.existsSync(entryPath)) {
        fail(`${entryPath} does not exists.`);
    }

    const nodeConsole = new NodeConsole('torch_debug');

    const dirPath = path.dirname(entryPath);
    const cmdOptions = new CommandLineOptions(cwd, false);
    cmdOptions.configFilePath = path.join(dirPath, 'pyrightconfig.json');

    const configPath = path.join(dirPath, 'pyteaconfig.json');
    const pytOptions: PytOptionsPart = { configPath, entryPath };

    const service = new PytService(cmdOptions, pytOptions, nodeConsole, true);

    describe('Python to JS Constraint', () => {
        const validity = service.setEntryPath(entryPath);

        if (validity) {
            fail(validity);
        }

        const configOptions = service.getConfig();

        test('Build HTML', () => {
            const basePath = path.join(entryPath, '..');
            const fileName = path.basename(entryPath, path.extname(entryPath));
            TorchTestUtils.printAsHTML(entryPath, combinePaths(basePath, fileName + '.html'), configOptions);
        });

        test('Python Script Test', () => {
            service.checkWithLog();
        });
    });
});
