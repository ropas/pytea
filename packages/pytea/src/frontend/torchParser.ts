/*
 * torchParser.ts
 * Copyright (c) Seoul National University.
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo
 *
 * PyTea Internal language script parser. (xxx.th.py)
 */

import { inspect } from 'util';

import {
    ArgumentCategory,
    AssignmentExpressionNode,
    AssignmentNode,
    BinaryOperationNode,
    BreakNode,
    CallNode,
    ConstantNode,
    ContinueNode,
    ExpressionNode,
    ForNode,
    FunctionNode,
    IfNode,
    IndexNode,
    MemberAccessNode,
    ModuleNode,
    NameNode,
    NumberNode,
    ParseNode,
    ParseNodeArray,
    ParseNodeType,
    PassNode,
    ReturnNode,
    StatementListNode,
    StringListNode,
    StringNode,
    TupleNode,
    UnaryOperationNode,
} from '../parser/parseNodes';
import { KeywordType, OperatorType } from '../parser/tokenizerTypes';
import { extractIds, flattenNodeArray, parseBinOp, parseUnaryOp } from './frontUtils';
import {
    LibCallType,
    TEAttr,
    TEBinOp,
    TECall,
    TEConst,
    TEConstType,
    TELibCall,
    TEName,
    TEObject,
    TESubscr,
    TETuple,
    TEType,
    TEUnaryOp,
    ThExpr,
    ThLeftExpr,
    ThStmt,
    TSAssign,
    TSBreak,
    TSContinue,
    TSExpr,
    TSForIn,
    TSFunDef,
    TSIf,
    TSLet,
    TSPass,
    TSReturn,
    TSSeq,
} from './torchStatements';

export interface ThStmtParser {
    parse(node: ParseNode): ThStmt;
}

export class TorchIRParser implements ThStmtParser {
    parse(node: ParseNode): ThStmt {
        const parser = new TorchIRParser();
        const stmt = parser.visitNode(node);

        if ('stype' in stmt) {
            return stmt;
        } else {
            return TSExpr.create(stmt);
        }
    }

    visitArray(nodes: ParseNodeArray, source?: ParseNode): ThStmt {
        const arr = flattenNodeArray(nodes);
        const localStack: ThStmt[] = [];

        for (const [idx, node] of arr.entries()) {
            if (!node) continue;

            if (node.nodeType === ParseNodeType.Function) {
                const [name, params, body] = this.visitFunction(node);
                const stmt = this.visitArray(arr.slice(idx + 1));
                localStack.push(TSFunDef.create(name, params, body, stmt, node));
                break;
            } else if (
                node.nodeType === ParseNodeType.Call &&
                node.leftExpression.nodeType === ParseNodeType.Name &&
                node.leftExpression.value.toUpperCase() === 'LET'
            ) {
                if (node.arguments.length !== 2 || node.arguments[0].valueExpression.nodeType !== ParseNodeType.Name) {
                    this.fail(node);
                }
                const name = (node.arguments[0].valueExpression as NameNode)?.value;
                const right = node.arguments[1].valueExpression;
                let expr: ThExpr | undefined;

                if (right.nodeType === ParseNodeType.Name && right.value.toLowerCase() === 'undef') {
                    expr = undefined;
                } else {
                    expr = this.visitExprNode(right);
                }

                const stmt = this.visitArray(arr.slice(idx + 1));
                localStack.push(TSLet.create(name, stmt, expr, node));
                break;
            }

            let stmt = this.visitNode(node);
            if ('etype' in stmt) {
                stmt = TSExpr.create(stmt);
            }
            localStack.push(stmt);
        }

        if (localStack.length === 0) {
            return TSPass.get(source);
        } else if (localStack.length === 1) {
            return localStack[0];
        } else {
            let stmt = localStack.pop()!;
            const rev = localStack.reverse();
            rev.forEach((s) => {
                stmt = TSSeq.create(s, stmt);
            });

            return stmt;
        }
    }

    visitModule(node: ModuleNode): ThStmt {
        const stmt = this.visitArray(node.statements, node);
        stmt.source = node;
        return stmt;
    }

    visitStatementList(node: StatementListNode): ThStmt {
        return this.visitArray(node.statements);
    }

    visitAssignment(node: AssignmentNode): ThStmt {
        return TSAssign.create(
            this.visitExprNode(node.leftExpression) as ThLeftExpr,
            this.visitExprNode(node.rightExpression),
            node
        );
    }

    visitBinaryOperation(node: BinaryOperationNode): ThExpr {
        let op = node.operator;
        let leftNode = node.leftExpression;
        let rightNode = node.rightExpression;

        // flip gt(e) to lt(e)
        if (op === OperatorType.GreaterThan || op === OperatorType.LessThanOrEqual) {
            leftNode = node.rightExpression;
            rightNode = node.leftExpression;
            if (op === OperatorType.GreaterThan) {
                op = OperatorType.LessThan;
            } else {
                op = OperatorType.LessThanOrEqual;
            }
        }

        const bop = parseBinOp(op);
        if (bop === undefined) {
            this.fail(node);
        }

        const left = this.visitExprNode(leftNode);
        const right = this.visitExprNode(rightNode);
        return TEBinOp.create(bop, left, right, node);
    }

    visitBreak(node: BreakNode): ThStmt {
        return TSBreak.create(node);
    }

    visitCall(node: CallNode): ThExpr {
        const args = node.arguments.map((arg) => this.visitExprNode(arg.valueExpression));

        if (
            node.leftExpression.nodeType === ParseNodeType.Name &&
            node.leftExpression.value.toUpperCase() === 'LIBCALL'
        ) {
            if (args.length === 0) {
                this.fail(node);
            }
            const nameExpr = args[0];
            if (nameExpr.etype !== TEType.Const || nameExpr.constType !== TEConstType.String) {
                this.fail(node);
            }

            const libKey = nameExpr.value as string;
            if (!(libKey in LibCallType)) {
                this.fail(node);
            }

            const paramKeys = node.arguments.map((arg) => (arg.name ? arg.name.value : ''));
            const params = args.map((v, i) => [paramKeys[i], v] as [string, ThExpr]).slice(1);

            return TELibCall.create(LibCallType[libKey as keyof typeof LibCallType], params, node);
        }

        return TECall.create(this.visitExprNode(node.leftExpression), args, node);
    }

    visitContinue(node: ContinueNode): ThStmt {
        return TSContinue.create(node);
    }

    visitConstant(node: ConstantNode): ThExpr {
        switch (node.constType) {
            case KeywordType.True:
                return TEConst.create(TEConstType.Bool, true, node);
            case KeywordType.False:
                return TEConst.create(TEConstType.Bool, false, node);
            case KeywordType.None:
                return TEConst.create(TEConstType.None, undefined, node);
            default:
                this.fail(node);
        }
    }

    visitIf(node: IfNode): ThStmt {
        const expr = this.visitExprNode(node.testExpression);
        const ifStmt = this.visitArray(node.ifSuite.statements, node.ifSuite);
        const elseSuite = node.elseSuite;

        let elseStmt: ThStmt;
        if (elseSuite) {
            if (elseSuite.nodeType === ParseNodeType.If) {
                elseStmt = this.visitIf(elseSuite);
            } else {
                elseStmt = this.visitArray(elseSuite.statements, elseSuite);
            }
        } else {
            elseStmt = TSPass.get();
        }

        return TSIf.create(expr, ifStmt, elseStmt, node);
    }

    visitIndex(node: IndexNode): ThExpr {
        const items = node.items.items.map((i) => this.visitExprNode(i));
        if (items.length === 0) {
            this.fail(node);
        } else if (items.length >= 2) {
            const tuple = TETuple.create(items, node);
            return TESubscr.create(this.visitExprNode(node.baseExpression), tuple, node);
        } else {
            return TESubscr.create(this.visitExprNode(node.baseExpression), items[0], node);
        }
    }

    visitFor(node: ForNode): ThStmt {
        const idx = extractIds(node.targetExpression);
        if (!idx) {
            this.fail(node);
        }
        const iter = this.visitExprNode(node.iterableExpression);
        const body = this.visitArray(node.forSuite.statements, node.forSuite);

        return TSForIn.create(idx, iter, body, node);
    }

    visitFunction(node: FunctionNode): [string, string[], ThStmt] {
        const name = node.name.value;
        const params = extractIds(node.parameters);
        if (!params) {
            this.fail(node);
        }
        const body = this.visitArray(node.suite.statements, node.suite);

        return [name, params, body];
    }

    visitName(node: NameNode): ThExpr {
        if (node.value === 'Object') {
            return TEObject.create(node);
        }
        return TEName.create(node.value);
    }

    visitNumber(node: NumberNode): ThExpr {
        if (node.isInteger) {
            return TEConst.create(TEConstType.Int, node.value, node);
        } else {
            return TEConst.create(TEConstType.Float, node.value, node);
        }
    }

    visitMemberAccess(node: MemberAccessNode): ThExpr {
        return TEAttr.create(this.visitExprNode(node.leftExpression), node.memberName.value, node);
    }

    visitPass(node: PassNode): ThStmt {
        return TSPass.get(node);
    }

    visitReturn(node: ReturnNode): ThStmt {
        const expr = node.returnExpression
            ? this.visitExprNode(node.returnExpression)
            : TEConst.create(TEConstType.None, undefined);
        return TSReturn.create(expr, node);
    }

    visitString(node: StringNode): ThExpr {
        return TEConst.create(TEConstType.String, node.value, node);
    }

    visitStringList(node: StringListNode): ThExpr {
        return TEConst.create(TEConstType.String, node.strings.map((str) => str.value).join(''), node);
    }

    visitTuple(node: TupleNode): ThExpr {
        return TETuple.create(
            node.expressions.map((e) => this.visitExprNode(e)),
            node
        );
    }

    visitUnaryOperation(node: UnaryOperationNode): ThExpr {
        const uop = parseUnaryOp(node.operator);
        if (!uop) {
            this.fail(node);
        }

        return TEUnaryOp.create(uop, this.visitExprNode(node.expression), node);
    }

    visitNode(node: ParseNode): ThStmt | ThExpr {
        switch (node.nodeType) {
            case ParseNodeType.Assignment:
            case ParseNodeType.Break:
            case ParseNodeType.Continue:
            case ParseNodeType.If:
            case ParseNodeType.For:
            case ParseNodeType.Module:
            case ParseNodeType.Pass:
            case ParseNodeType.Return:
            case ParseNodeType.StatementList:
                return this.visitStmtNode(node);
            case ParseNodeType.BinaryOperation:
            case ParseNodeType.UnaryOperation:
            case ParseNodeType.Call:
            case ParseNodeType.Constant:
            case ParseNodeType.Index:
            case ParseNodeType.MemberAccess:
            case ParseNodeType.Name:
            case ParseNodeType.Number:
            case ParseNodeType.String:
            case ParseNodeType.StringList:
            case ParseNodeType.Tuple:
                return this.visitExprNode(node as ExpressionNode);
            default:
                return TSPass.get(node);
        }
    }

    visitStmtNode(node: ParseNode): ThStmt {
        switch (node.nodeType) {
            case ParseNodeType.Assignment:
                return this.visitAssignment(node);
            case ParseNodeType.Break:
                return this.visitBreak(node);
            case ParseNodeType.Continue:
                return this.visitContinue(node);
            case ParseNodeType.If:
                return this.visitIf(node);
            case ParseNodeType.For:
                return this.visitFor(node);
            case ParseNodeType.Module:
                return this.visitModule(node);
            case ParseNodeType.Pass:
                return this.visitPass(node);
            case ParseNodeType.Return:
                return this.visitReturn(node);
            case ParseNodeType.StatementList:
                return this.visitStatementList(node);
            default:
                return TSPass.get(node);
        }
    }

    visitExprNode(node: ExpressionNode): ThExpr {
        switch (node.nodeType) {
            case ParseNodeType.BinaryOperation:
                return this.visitBinaryOperation(node);
            case ParseNodeType.UnaryOperation:
                return this.visitUnaryOperation(node);
            case ParseNodeType.Call:
                return this.visitCall(node);
            case ParseNodeType.Constant:
                return this.visitConstant(node);
            case ParseNodeType.Index:
                return this.visitIndex(node);
            case ParseNodeType.MemberAccess:
                return this.visitMemberAccess(node);
            case ParseNodeType.Name:
                return this.visitName(node);
            case ParseNodeType.Number:
                return this.visitNumber(node);
            case ParseNodeType.String:
                return this.visitString(node);
            case ParseNodeType.StringList:
                return this.visitStringList(node);
            case ParseNodeType.Tuple:
                return this.visitTuple(node);
            default:
                return this.fail(node);
        }
    }

    fail(node: ParseNode): never {
        fail('invalid node for TorchIR script: ' + inspect(node));
    }
}
