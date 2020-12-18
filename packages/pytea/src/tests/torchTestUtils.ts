/*
 * builtins.ts
 * Copyright (c) Seoul National University.
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo
 *
 * Utilities for PyTea entry point
 */

// deprecated
// import { exec } from 'child_process';
// import * as fs from 'fs';
// import * as os from 'os';
// import * as path from 'path';
// import { promisify } from 'util';

// import { ConfigOptions } from 'pyright-internal/common/configOptions';
// import { assert } from 'pyright-internal/common/debug';
// import { combinePaths, normalizePath } from 'pyright-internal/common/pathUtils';

// import { TreePrinter } from '../frontend/treePrinter';
// import { runProgram } from '../pyt/pytUtils';

// const torchEnvName = 'torch-venv';
// const execPromise = promisify(exec);

// export function resolveSampleFilePath(fileName: string): string {
//     return path.resolve(path.dirname(module.filename), `./samples/${fileName}`);
// }

// async function installTorchVenv(venvPath: string, makeVenv: boolean, isPy3: boolean) {
//     const pythonCmd = isPy3 ? 'python3' : 'python';
//     const pipCmd = isPy3 ? 'pip3' : 'pip';
//     if (makeVenv) {
//         console.log(`making virtualenv in ${venvPath} ...`);
//         await execPromise(`${pythonCmd} -m venv ${venvPath}`);
//     }

//     const isWin = os.platform().indexOf('win') > -1;
//     const isMac = os.platform() === 'darwin';
//     let installCommand: string;

//     if (isMac) {
//         installCommand = `${pipCmd} install torch torchvision`;
//     } else {
//         installCommand = `${pipCmd} install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html`;
//     }

//     console.log('installing pytorch...');
//     if (isWin) {
//         const activatePath = combinePaths(venvPath, 'Scripts', 'activate.bat');
//         await execPromise(`${activatePath}; ${installCommand}`);
//     } else {
//         const activatePath = combinePaths(venvPath, 'bin', 'activate');
//         await execPromise(`source ${activatePath}; ${installCommand}`, { shell: '/bin/bash' });
//     }

//     // TODO: should we deactivate venv? If then, how to deactivate it in Windows?
// }

// export async function checkOrInstallTorchVenv() {
//     // const isWin = os.platform().indexOf('win') > -1;

//     let foundPy3 = false;
//     try {
//         await execPromise(`python3 --version`);
//         foundPy3 = true;
//     } catch (err) {
//         // console.log('python3 is not installed. trying `python` command instead');
//     }

//     if (!foundPy3) {
//         try {
//             const { stdout } = await execPromise(`python --version`);
//             if (!stdout.startsWith('Python 3')) {
//                 throw new Error('Python 3 is not installed.');
//             }
//         } catch (err) {
//             throw new Error('Python 3 is not installed.');
//         }
//     }

//     const torchTestDirectory = normalizePath(combinePaths(process.cwd(), '../server/src/tests/samples/project_torch'));
//     const torchEnvPath = combinePaths(torchTestDirectory, torchEnvName);
//     if (!fs.existsSync(torchEnvPath)) {
//         await installTorchVenv(torchEnvPath, true, foundPy3);
//     } else if (!fs.existsSync(combinePaths(torchEnvPath, ''))) {
//         await installTorchVenv(torchEnvPath, false, foundPy3);
//     }
// }

// export function printAsHTML(filePath: string, saveHTMLPath: string, configOptions = new ConfigOptions('.')) {
//     const program = runProgram([filePath], configOptions);

//     // analyze single pytorch entry file
//     const sourceFile = program.getSourceFile(filePath);
//     const mainNode = sourceFile?.getParseResults()?.parseTree;

//     assert(mainNode !== undefined);

//     const printer = new TreePrinter(program, mainNode!);
//     fs.writeFileSync(saveHTMLPath, printer.makeHTML());
// }
