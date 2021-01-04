/*
 * IRReaderWriter.ts
 * Copyright (c) Seoul National University.
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo (mersshs@gmail.com)
 *
 * String formatter for PyTea internal representation
 * See
 */

import { PyteaService } from 'src/service/pyteaService';

import { ParseNode } from 'pyright-internal/parser/parseNodes';

import { TEType, ThExpr, ThStmt, TSSeq, TSType } from './torchStatements';

export namespace IRWriter {
    export function makeIRString(stmt: ThStmt | ThExpr, service: PyteaService): string {
        let code: string;
        const sourceMap: string[] = new Map();

        function showStmt(stmt: ThStmt): string {
            const source = showSourcePos(service, sourceMap, stmt.source);
            switch (stmt.stype) {
                case TSType.Pass:
                    return source ? `(pass ${source})` : '(pass)';
                case TSType.Expr:
                    return showExpr(stmt.expr);
                case TSType.Seq: {
                    const flatten: ThStmt[] = [];
                    function push(subStmt: TSSeq): void {
                        if (subStmt.left.stype === TSType.Seq) {
                            push(subStmt.left);
                        } else {
                            flatten.push(subStmt.left);
                        }
                        if (subStmt.right.stype === TSType.Seq) {
                            push(subStmt.right);
                        } else {
                            flatten.push(subStmt.right);
                        }
                    }
                    push(stmt);
                    return `(${flatten.map(showStmt).join(' ')})`;
                }
                case TSType.Assign:
                case TSType.If:
                case TSType.ForIn:
                case TSType.Return:
                case TSType.Continue:
                case TSType.Break:
                case TSType.Let:
                case TSType.FunDef:
                    return ``;
            }
        }

        function showExpr(expr: ThExpr): string {
            const source = showSourcePos(service, sourceMap, stmt.source);
            switch (expr.etype) {
                case TEType.Object:
                    return source ? `(object ${source})` : '(object';
                case TEType.Const:
                case TEType.Tuple:
                case TEType.Call:
                case TEType.LibCall:
                case TEType.BinOp:
                case TEType.UnaryOp:
                case TEType.Name:
                case TEType.Attr:
                case TEType.Subscr:
                    return '';
            }
        }

        if ('stype' in stmt) {
            code = showStmt(stmt);
        } else {
            code = showExpr(stmt);
        }

        const sourceMapStr = `(source-map \n${sourceMap.map((path) => `(${path})`).join('\n')})`;

        return `${code}\n${sourceMapStr}`;
    }

    export function showSourcePos(service: PyteaService, sourceMap: string[], node?: ParseNode): string {
        if (!node) return ' ';
        // TODO
        return ' ';
    }
}

export namespace IRReader {
    export function parseIRString(code: string): ThStmt | undefined {
        // TODO
        return;
    }

    export function tokenize(code: string): Token[] {
        const tokens: Token[] = [];
        const len = code.length;

        // 0: normal / 1: parsing string / 2: parsing number
        let currType = 0;
        let charStack: string[] = [];

        for (let i = 0; i < len; i++) {
            const char = code[i];

            if (currType !== 1) {
                // parse normal tokens and numbers
                let shouldPush = true;
                let tokenAfter: Token | undefined = undefined;
                switch (char) {
                    case '"':
                        currType = 1;
                        break;
                    case '(':
                        tokenAfter = { type: TokenType.LPar };
                        break;
                    case ')':
                        tokenAfter = { type: TokenType.RPar };
                        break;
                    case '[':
                        tokenAfter = { type: TokenType.LBrak };
                        break;
                    case ']':
                        tokenAfter = { type: TokenType.RBrak };
                        break;
                    case ':':
                        tokenAfter = { type: TokenType.Colon };
                        break;
                    case '0':
                    case '1':
                    case '2':
                    case '3':
                    case '4':
                    case '5':
                    case '6':
                    case '7':
                    case '8':
                    case '9':
                    case '.':
                        currType = 2;
                        break;
                    case ' ':
                        break;
                    default:
                        shouldPush = false;
                        charStack.push(char);
                        break;
                }

                if (shouldPush && charStack.length > 0) {
                    const value = charStack.join();
                    if (currType === 2) {
                        tokens.push({ type: TokenType.Number, value: Number.parseFloat(value) });
                        currType = 0;
                    } else {
                        tokens.push({ type: TokenType.Command, value });
                    }

                    charStack = [];
                }

                if (tokenAfter) {
                    tokens.push(tokenAfter);
                }
            } else if (currType === 1) {
                // parsing string
                switch (char) {
                    case '"':
                        tokens.push({
                            type: TokenType.String,
                            value: charStack.join(),
                        });
                        charStack = [];
                        currType = 0;
                        break;
                    case '\\':
                        if (i + 1 < len && code[i + 1] === '"') {
                            i++;
                            charStack.push('"');
                            break;
                        }
                    // eslint-disable-next-line no-fallthrough
                    default:
                        charStack.push(char);
                        break;
                }
            }
        }

        return tokens;
    }

    const enum TokenType {
        LPar,
        RPar,
        LBrak,
        RBrak,
        Colon,
        Command,
        Number,
        String,
    }

    type Token = TkPar | TkCmd | TkNum | TkStr;
    interface TokenBase {
        type: TokenType;
    }

    interface TkPar extends TokenBase {
        type: TokenType.LPar | TokenType.RPar | TokenType.LBrak | TokenType.RBrak | TokenType.Colon;
    }

    interface TkCmd extends TokenBase {
        type: TokenType.Command;
        value: string;
    }

    interface TkNum extends TokenBase {
        type: TokenType.Number;
        value: number;
    }

    interface TkStr extends TokenBase {
        type: TokenType.String;
        value: string;
    }
}
