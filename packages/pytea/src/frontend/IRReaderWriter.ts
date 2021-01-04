/*
 * IRReaderWriter.ts
 * Copyright (c) Seoul National University.
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo (mersshs@gmail.com)
 *
 * String formatter for PyTea internal representation
 * See
 */

import { ThExpr, ThStmt } from './torchStatements';

export namespace IRWriter {
    export function exportIR(code: ThStmt | ThExpr): string {
        // TODO:

        return '';
    }
}

export namespace IRReader {
    export function parseIRStmt(code: string): ThStmt | undefined {
        // TODO:
        return;
    }

    export function parseIRExpr(code: string): ThStmt | undefined {
        // TODO:
        return;
    }
}
