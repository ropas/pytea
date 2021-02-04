/*
 * IRReaderWriter.ts
 * Copyright (c) Seoul National University.
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo (mersshs@gmail.com)
 *
 * String formatter for PyTea internal representation
 * See doc/IR-format.md for more information
 */

import { ParseNode } from 'pyright-internal/parser/parseNodes';

import { TEBinOp, TEConst, TEConstType, TEType, TEUopType, ThExpr, ThStmt, TSSeq, TSType } from './torchStatements';

export namespace IRWriter {
    export function makeIRString(stmt: ThStmt | ThExpr, path: string): string {
        let code: string;

        if ('stype' in stmt) {
            code = showStmt(stmt);
        } else {
            code = showExpr(stmt);
        }

        return `(source-map ${showStr(path)} ${code})`;
    }

    export function showStmt(stmt: ThStmt): string {
        const source = ' ' + showSourcePos(stmt.source);
        switch (stmt.stype) {
            case TSType.Pass:
                return `(pass${source})`;
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
                return `(assign${source} ${showExpr(stmt.left)} ${showExpr(stmt.right)})`;
            case TSType.If:
                return `(if${source} ${showExpr(stmt.cond)} ${showStmt(stmt.thenStmt)} ${showStmt(stmt.elseStmt)})`;
            case TSType.ForIn:
                return `(for${source} ${showStr(stmt.ident)} ${showExpr(stmt.loopVal)} ${showStmt(stmt.loopBody)})`;
            case TSType.Return:
                return `(return${source})`;
            case TSType.Continue:
                return `(continue${source})`;
            case TSType.Break:
                return `(break${source})`;
            case TSType.Let:
                return `(let${source} (${showStr(stmt.name)} ${stmt.expr ? showExpr(stmt.expr) : ''}) ${showStmt(
                    stmt.scope
                )})`;
            case TSType.FunDef:
                return `(fundef${source} ${showStr(stmt.name)} (${stmt.params.map(showStr).join(' ')}) ${showStmt(
                    stmt.body
                )} ${showStmt(stmt.scope)})`;
        }
    }

    export function showExpr(expr: ThExpr): string {
        const source = ' ' + showSourcePos(expr.source);
        switch (expr.etype) {
            case TEType.Object:
                return `(object${source})`;
            case TEType.Tuple:
                return `(tuple${source} ${expr.values.map(showExpr).join(' ')})`;
            case TEType.Call:
                return `(call${source} ${showExpr(expr.func)} ${expr.params.map(showExpr).join(' ')})`;
            case TEType.LibCall:
                return `(libcall${source} ${showStr(expr.type)} ${expr.params
                    .map((p) => `(${showStr(p[0])} ${showExpr(p[1])})`)
                    .join(' ')})`;
            case TEType.BinOp:
                return `(bop${source} ${TEBinOp.toStringBop(expr.bopType)} ${showExpr(expr.left)} ${showExpr(
                    expr.right
                )})`;
            case TEType.UnaryOp: {
                const uop = expr.uopType === TEUopType.Neg ? '-' : 'not';
                return `(uop${source} ${uop} ${showExpr(expr.base)})`;
            }
            case TEType.Name:
                return `(var${source} ${showStr(expr.ident)})`;
            case TEType.Attr:
                return `(attr${source} ${showExpr(expr.left)} ${showStr(expr.right)})`;
            case TEType.Subscr:
                return `(subs${source} ${showExpr(expr.left)} ${showExpr(expr.right)})`;
            case TEType.Const:
                return showConst(expr);
        }
    }

    export function showConst(expr: TEConst): string {
        const source = ' ' + showSourcePos(expr.source);
        switch (expr.constType) {
            case TEConstType.Bool:
                return `(bool${source} ${(expr.value as boolean) ? 'True' : 'False'})`;
            case TEConstType.Int:
                return `(int${source} ${expr.value as number})`;
            case TEConstType.Float:
                return `(float${source} ${expr.value as number})`;
            case TEConstType.String:
                return `(str${source} ${showStr(expr.value as string)})`;
            case TEConstType.None:
                return `(none${source})`;
        }
    }

    export function showSourcePos(node?: ParseNode): string {
        if (!node) return '';
        return `[${node.start}:${node.start + node.length}]`;
    }

    export function showStr(str: string): string {
        const escaped = str
            .replace(/[\\"']/g, '\\$&')
            .replace(/\n/g, '\\n')
            .replace(/\r/g, '\\r')
            .replace(/\t/g, '\\t');
        return `"${escaped}"`;
    }
}

export namespace IRReader {
    // returns [sourceMap, top-level statements]
    // or returns error message
    export function parseIRString(code: string): [Map<string, ThStmt>, ThStmt[]] | string {
        const sourceMap: Map<string, ThStmt> = new Map();
        const stmts: ThStmt[] = [];

        const tokens = tokenize(code);
        const len = tokens.length;
        let pos = 0;
        while (pos < len) {
            const parsedMap = parseSourceMap(tokens, pos);
            if (typeof parsedMap === 'string') {
                const stmt = parseStmt(tokens, pos);
                if (typeof stmt === 'string') {
                    return formatErrorMessage(stmt, tokens, pos);
                }

                stmts.push(stmt[0]);
                pos = stmt[1];
            } else {
                sourceMap.set(parsedMap[0], parsedMap[1]);
                pos = parsedMap[2];
            }
        }

        return [sourceMap, stmts];
    }

    function formatErrorMessage(message: string, tokens: Token[], pos: number): string {
        const start = pos - 5 < 0 ? 0 : pos - 5;
        const surround = tokens
            .slice(start, pos + 5)
            .map(showToken)
            .join(' ');
        return `${message}\n  PARSE ERROR at token #${pos + 1}:\n  ... ${surround} ...`;
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
                    case ' ':
                    case '\n':
                    case '\r':
                    case '\t':
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
                    // eslint-disable-next-line no-fallthrough
                    default:
                        shouldPush = false;
                        charStack.push(char);
                        break;
                }

                if (shouldPush && charStack.length > 0) {
                    const value = charStack.join('');
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
                            value: charStack.join(''),
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

    export function showToken(tk: Token): string {
        switch (tk.type) {
            case TokenType.LPar:
                return '(';
            case TokenType.RPar:
                return ')';
            case TokenType.LBrak:
                return '[';
            case TokenType.RBrak:
                return ']';
            case TokenType.Colon:
                return ':';
            case TokenType.Command:
                return tk.value;
            case TokenType.Number:
                return tk.value.toString();
            case TokenType.String:
                return `"${tk.value}"`;
        }
    }

    // returns parsed sourceMap and next positoin / or return error message
    function parseSourceMap(tokens: Token[], pos: number): [string, ThStmt, number] | string {
        return 'not-implemented';
    }
    // returns parsed stmt and next position / or return error message
    function parseStmt(tokens: Token[], pos: number): [ThStmt, number] | string {
        return 'not-implemented';
    }

    // returns parsed expr and next position / or return error message
    function parseExpr(tokens: Token[], pos: number): [ThExpr, number] | string {
        return 'not-implemented';
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
