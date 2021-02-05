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

import {
    TEBinOp,
    TEConst,
    TEConstType,
    TEType,
    TEUopType,
    ThExpr,
    ThLeftExpr,
    ThStmt,
    TSAssign,
    TSExpr,
    TSSeq,
    TSType,
} from './torchStatements';

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
            try {
                const parsedMap = parseSourceMap(tokens, pos);
                sourceMap.set(parsedMap[0], parsedMap[1]);
                pos = parsedMap[2];
            } catch (msgPos) {
                // if the error is occured from the stmt in source-map, return it.
                if (pos + 1 >= len) {
                    const cmd = tokens[pos + 1];
                    if (cmd.type === TokenType.Command && cmd.value === 'source-map') {
                        return formatErrorMessage(msgPos[0] as string, tokens, msgPos[1]);
                    }
                }

                // else, fall back to stmt parser
                try {
                    const stmt = parseStmt(tokens, pos);
                    stmts.push(stmt[0]);
                    pos = stmt[1];
                } catch (msgPos) {
                    return formatErrorMessage(msgPos[0] as string, tokens, msgPos[1]);
                }
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
        return `${message}\n  PARSE ERROR at ... ${surround} ...`;
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
                        tokenAfter = tkLPar;
                        break;
                    case ')':
                        tokenAfter = tkRPar;
                        break;
                    case '[':
                        tokenAfter = tkLBrak;
                        break;
                    case ']':
                        tokenAfter = tkRBrak;
                        break;
                    case ':':
                        tokenAfter = tkColon;
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
    function parseSourceMap(tokens: Token[], pos: number): [string, ThStmt, number] {
        const next = consume(tokens, pos, [tkLPar, tkSourceMap]);
        if (next < 0) throw ['not a source map', pos];

        const path = parseStr(tokens, next);
        const [stmt, next2] = parseStmt(tokens, next);

        const next3 = consume(tokens, next2, [tkRPar]);
        if (next3 < 0) throw ['source-map is not closed by )', next2 + 1];

        return [path, stmt, next2];
    }
    // returns parsed stmt and next position / or return error message
    function parseStmt(tokens: Token[], pos: number): [ThStmt, number] {
        let next = pos;
        let stmt: ThStmt;
        if (tokens.length <= pos + 2) {
            throw ['stmt is too short', pos];
        }
        if (tokens[pos]?.type !== TokenType.LPar) {
            throw ['stmt is not started with (', pos];
        }

        const cmd = tokens[pos + 1];
        if (cmd.type === TokenType.Command) {
            next = skipSource(tokens, pos + 2);
            switch (cmd.value) {
                case 'assign': {
                    const [left, next2] = parseExpr(tokens, next);
                    const [right, next3] = parseExpr(tokens, next2);
                    next = next3;
                    stmt = TSAssign.create(left as ThLeftExpr, right);
                    break;
                }
                case 'let':
                case 'fundef':
                case 'if':
                case 'for':
                case 'pass':
                case 'return':
                case 'continue':
                case 'break':
                default: {
                    const [expr, next] = parseExpr(tokens, pos);
                    return [TSExpr.create(expr), next];
                }
            }
        } else {
            // seq
            const seq: ThStmt[] = [];
            next = pos + 1;
            try {
                const [seqStmt, seqNext] = parseStmt(tokens, next);
                seq.push(seqStmt);
                next = seqNext;
            } catch (e) {
                // parse stmt until it reaches end
            }

            if (seq.length === 0) {
                throw ['stmt-seq is empty', next];
            }
            let last: ThStmt = seq[seq.length - 1];
            for (let i = seq.length - 2; i >= 0; i--) {
                last = TSSeq.create(seq[i], last);
            }
            stmt = last;
        }

        const next2 = consume(tokens, next, [tkRPar]);
        if (next2 < 0) throw ['source-map is not closed by )', next + 1];

        return [stmt, next];
    }

    // returns parsed expr and next position / or return error message
    function parseExpr(tokens: Token[], pos: number): [ThExpr, number] {
        if (tokens.length <= pos + 2) {
            throw ['expr is too short', pos];
        }
        if (tokens[pos]?.type !== TokenType.LPar) {
            throw ['expr is not started with (', pos];
        }
    }

    // match tokens with matcher. if failed, return -1, or next position
    function consume(tokens: Token[], pos: number, matcher: Token[]): number {
        if (tokens.length < matcher.length + pos) {
            return -1;
        }

        for (let i = pos; i < pos + matcher.length; i++) {
            const [left, right] = [tokens[pos + i], matcher[pos + i]];
            if (left.type !== right.type) {
                return -1;
            }
            if (left.type === TokenType.Command && right.type === TokenType.Command && left.value !== right.value) {
                return -1;
            }
        }

        return pos + matcher.length;
    }

    function skipSource(tokens: Token[], pos: number): number {
        const next = consume(tokens, pos, [tkLBrak, tkDummyNum, tkColon, tkDummyNum, tkRBrak]);
        return next < 0 ? pos : next + pos;
    }

    // throwable
    function parseStr(tokens: Token[], pos: number): string {
        const token: Token | undefined = tokens[pos];
        if (token?.type !== TokenType.String) {
            throw [`expected string, got ${token ? showToken(token) : undefined}`, pos];
        }
        return token.value;
    }

    function parseNum(tokens: Token[], pos: number): number {
        const token: Token | undefined = tokens[pos];
        if (token?.type !== TokenType.Number) {
            throw [`expected number, got ${token ? showToken(token) : undefined}`, pos];
        }
        return token.value;
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

    const tkLPar: TkPar = { type: TokenType.LPar };
    const tkRPar: TkPar = { type: TokenType.RPar };
    const tkLBrak: TkPar = { type: TokenType.LBrak };
    const tkRBrak: TkPar = { type: TokenType.RBrak };
    const tkColon: TkPar = { type: TokenType.Colon };
    const tkSourceMap: TkCmd = { type: TokenType.Command, value: 'source-map' };
    const tkDummyNum: TkNum = { type: TokenType.Number, value: 0 };
}
