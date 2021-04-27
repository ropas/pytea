/*
 * frontUtils.ts
 * Copyright (c) Seoul National University.
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo (mersshs@gmail.com)
 *
 * Utility functions for frontend
 */

import { ParseTreeWalker } from 'pyright-internal/analyzer/parseTreeWalker';
import {
    AssignmentNode,
    ClassNode,
    ExpressionNode,
    ForNode,
    FunctionNode,
    GlobalNode,
    ImportAsNode,
    ImportFromAsNode,
    ModuleNameNode,
    NonlocalNode,
    ParameterNode,
    ParseNode,
    ParseNodeArray,
    ParseNodeType,
    WithItemNode,
} from 'pyright-internal/parser/parseNodes';
import { OperatorType } from 'pyright-internal/parser/tokenizerTypes';

import { TEAttr, TEBopType, TEType, TEUopType } from './torchStatements';

export function toQualPath(module: ModuleNameNode): string {
    return '.'.repeat(module.leadingDots) + module.nameParts.map((name) => name.value).join('.');
}

export function flattenNodeArray(nodes: ParseNodeArray): ParseNode[] {
    const arr: ParseNode[] = [];

    nodes.forEach((node) => {
        if (node) {
            if (node.nodeType === ParseNodeType.StatementList) {
                node.statements.forEach((stmt) => {
                    arr.push(stmt);
                });
            } else {
                arr.push(node);
            }
        }
    });

    return arr;
}

export function parseBinOp(binOp: OperatorType): TEBopType | undefined {
    switch (binOp) {
        case OperatorType.Add:
            return TEBopType.Add;
        case OperatorType.Subtract:
            return TEBopType.Sub;
        case OperatorType.Multiply:
            return TEBopType.Mul;
        case OperatorType.Power:
            return TEBopType.Pow;
        case OperatorType.Divide:
            return TEBopType.TrueDiv;
        case OperatorType.FloorDivide:
            return TEBopType.FloorDiv;
        case OperatorType.Mod:
            return TEBopType.Mod;
        case OperatorType.LessThan:
            return TEBopType.Lt;
        case OperatorType.LessThanOrEqual:
            return TEBopType.Lte;
        case OperatorType.Equals:
            return TEBopType.Eq;
        case OperatorType.NotEquals:
            return TEBopType.Neq;
        case OperatorType.And:
            return TEBopType.And;
        case OperatorType.Or:
            return TEBopType.Or;
        case OperatorType.Is:
            return TEBopType.Is;
        case OperatorType.IsNot:
            return TEBopType.IsNot;
        case OperatorType.In:
            return TEBopType.In;
        case OperatorType.NotIn:
            return TEBopType.NotIn;
        default:
            return undefined;
    }
}

export function getFullAttrPath(attr: TEAttr): string[] | undefined {
    let leftPath: string[] | undefined;
    switch (attr.left.etype) {
        case TEType.Name:
            return [attr.left.ident, attr.right];
        case TEType.Attr:
            leftPath = getFullAttrPath(attr.left);
            if (leftPath) {
                leftPath.push(attr.right);
            }
            return leftPath;
        default:
            return undefined;
    }
}

export function parseUnaryOp(unaryOp: OperatorType): TEUopType | undefined {
    switch (unaryOp) {
        case OperatorType.Subtract:
            return TEUopType.Neg;
        case OperatorType.Not:
            return TEUopType.Not;
        default:
            return;
    }
}

export function extractIds(node: ParseNode | ParameterNode[]): string[] | undefined {
    let exprs: ExpressionNode[];
    if (Array.isArray(node)) {
        const params = node.map((param) => param.name).filter((p) => p !== undefined);
        exprs = params as ExpressionNode[];
    } else {
        switch (node.nodeType) {
            case ParseNodeType.Tuple:
                exprs = node.expressions;
                break;
            case ParseNodeType.List:
                exprs = node.entries;
                break;
            case ParseNodeType.Name:
                return [node.value];
            default:
                return undefined;
        }
    }

    const ids = exprs.map(extractIds);
    if (ids.some((s) => s === undefined)) {
        return;
    }
    const idFlat = ids.flat();
    if (idFlat.some((s) => s === undefined)) return;
    return idFlat as string[];
}

/// extract new local variable names from ParseNodeArray
export function extractLocalDef(nodes: ParseNodeArray, exception?: (string | undefined)[]): Set<string> {
    const localSet = new LocalExtractor(nodes).extract();
    if (exception) {
        exception.forEach((e) => {
            if (e) {
                localSet.delete(e);
            }
        });
    }
    return localSet;
}

export function extractUnexportedGlobal(nodes: ParseNodeArray): Set<string> {
    return new UnexportedGlobalExtractor(nodes).extract();
}

class LocalExtractor extends ParseTreeWalker {
    private _names: Set<string>;
    private _nonlocal: Set<string>;

    constructor(nodes: ParseNodeArray) {
        super();
        this._names = new Set();
        this._nonlocal = new Set();
        this.walkMultiple(nodes);
    }

    extract(): Set<string> {
        if (this._nonlocal.size === 0) {
            return this._names;
        }

        return new Set([...this._names].filter((x) => !this._nonlocal.has(x)));
    }

    visitAssignment(node: AssignmentNode) {
        if (node.rightExpression.nodeType === ParseNodeType.Assignment) {
            this.visitAssignment(node.rightExpression);
        }

        const names = extractIds(node.leftExpression);
        names?.forEach((name) => {
            this._names.add(name);
        });

        return false;
    }

    visitClass(node: ClassNode) {
        this._names.add(node.name.value);
        return false;
    }

    visitGlobal(node: GlobalNode) {
        node.nameList.forEach((nn) => {
            this._nonlocal.add(nn.value);
        });
        return false;
    }

    visitNonlocal(node: NonlocalNode) {
        node.nameList.forEach((nn) => {
            this._nonlocal.add(nn.value);
        });
        return false;
    }

    visitFunction(node: FunctionNode) {
        this._names.add(node.name.value);
        return false;
    }

    visitFor(node: ForNode) {
        const idx = extractIds(node.targetExpression);
        idx?.forEach((name) => this._names.add(name));
        return true;
    }

    visitImportAs(node: ImportAsNode) {
        if (node.alias) {
            this._names.add(node.alias.value);
        } else if (node.module.nameParts.length >= 1) {
            this._names.add(node.module.nameParts[0].value);
        }
        return false;
    }

    visitImportFromAs(node: ImportFromAsNode) {
        if (node.alias) {
            this._names.add(node.alias.value);
        } else {
            this._names.add(node.name.value);
        }

        return false;
    }

    visitWithItem(node: WithItemNode) {
        if (node.target && node.target.nodeType === ParseNodeType.Name) {
            this._names.add(node.target.value);
        }
        return true;
    }
}

class UnexportedGlobalExtractor extends ParseTreeWalker {
    private _names: Set<string>;
    private _nonlocal: Set<string>;

    constructor(nodes: ParseNodeArray) {
        super();
        this._names = new Set();
        this._nonlocal = new Set();
        this.walkMultiple(nodes);
    }

    extract(): Set<string> {
        if (this._nonlocal.size === 0) {
            return this._names;
        }

        return new Set([...this._names].filter((x) => !this._nonlocal.has(x)));
    }

    visitImportAs(node: ImportAsNode) {
        if (!node.alias && node.module.nameParts.length === 1) {
            const name = node.module.nameParts[0].value;
            if (name) {
                this._names.add(name);
            }
        }
        return false;
    }

    visitFor(node: ForNode) {
        const idx = extractIds(node.targetExpression);
        idx?.forEach((name) => this._names.add(name));
        return true;
    }

    visitWithItem(node: WithItemNode) {
        if (node.target && node.target.nodeType === ParseNodeType.Name) {
            this._names.add(node.target.value);
        }
        return true;
    }
}
