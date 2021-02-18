/*
 * IRReaderWriter.ts
 * Copyright (c) Seoul National University.
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo (mersshs@gmail.com)
 *
 * String formatter for PyTea internal representation
 * See doc/IR-format.md for more information
 */

import { CodeSource } from '../backend/sharpValues';
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

    export function showSourcePos(node?: CodeSource): string {
        if (!node) return '';
        if ('fileId' in node) {
            const { start, end } = node.range;
            return `[${start.line}:${start.character}-${end.line}:${end.character}]`;
        }

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
