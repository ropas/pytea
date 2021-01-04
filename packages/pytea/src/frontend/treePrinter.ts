/*
 * treePrinter.ts
 * Copyright (c) Seoul National University.
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo (mersshs@gmail.com)
 *
 * ParseNode HTML Printer for debugging
 */

import { AnalyzerFileInfo } from 'pyright-internal/analyzer/analyzerFileInfo';
import * as AnalyzerNodeInfo from 'pyright-internal/analyzer/analyzerNodeInfo';
import { FlowNode } from 'pyright-internal/analyzer/codeFlow';
import { FlowFlags } from 'pyright-internal/analyzer/codeFlow';
import { ImportResult } from 'pyright-internal/analyzer/importResult';
import { ParseTreeWalker } from 'pyright-internal/analyzer/parseTreeWalker';
import { Program } from 'pyright-internal/analyzer/program';
import { ModuleNode, ParseNode, ParseNodeType } from 'pyright-internal/parser/parseNodes';

// From https://www.w3schools.com/howto/tryit.asp?filename=tryhow_js_treeview
const HTMLHeader = `
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
ul, #myUL {
  list-style-type: none;
}

#myUL {
  margin: 0;
  padding: 0;
}

.caret {
  cursor: pointer;
  -webkit-user-select: none; /* Safari 3.1+ */
  -moz-user-select: none; /* Firefox 2+ */
  -ms-user-select: none; /* IE 10+ */
  user-select: none;
}

.caret::before {
  content: "\\25B6";
  color: black;
  display: inline-block;
  margin-right: 6px;
}

.caret-down::before {
  -ms-transform: rotate(90deg); /* IE 9 */
  -webkit-transform: rotate(90deg); /* Safari */'
  transform: rotate(90deg);
}

.nested {
  display: none;
}

.active {
  display: block;
}
</style>
</head>
<body>
<ul id="myUL">
`;

const HTMLFooter = `
</ul>
<button onclick="test()">test</button>
<script>
var toggler = document.getElementsByClassName("caret");
var i;

for (i = 0; i < toggler.length; i++) {
  toggler[i].addEventListener("click", function() {
    this.parentElement.querySelector(".nested").classList.toggle("active");
    this.classList.toggle("caret-down");
  });
}
function test() {
    for (i = 0; i < toggler.length; i++) {
        toggler[i].parentElement.querySelector(".nested").classList.toggle("active");
        toggler[i].classList.toggle("caret-down");
    }
}
</script>

</body>
</html>
`;

// For reverse mapping
enum ParseNodeTypeCopy {
    Error, // 0

    Argument,
    Assert,
    Assignment,
    AssignmentExpression,
    AugmentedAssignment,
    Await,
    BinaryOperation,
    Break,
    Call,

    Class, // 10
    Constant,
    Continue,
    Decorator,
    Del,
    Dictionary,
    DictionaryExpandEntry,
    DictionaryKeyEntry,
    Ellipsis,
    If,

    Import, // 20
    ImportAs,
    ImportFrom,
    ImportFromAs,
    Index,
    IndexItems,
    Except,
    For,
    FormatString,
    Function,

    Global, // 30
    Lambda,
    List,
    ListComprehension,
    ListComprehensionFor,
    ListComprehensionIf,
    MemberAccess,
    Module,
    ModuleName,
    Name,

    Nonlocal, // 40
    Number,
    Parameter,
    Pass,
    Raise,
    Return,
    Set,
    Slice,
    StatementList,
    StringList,

    String, // 50
    Suite,
    Ternary,
    Tuple,
    Try,
    TypeAnnotation,
    UnaryOperation,
    Unpack,
    While,
    With,

    WithItem, // 60
    Yield,
    YieldFrom,
}

type NodeWithName = [string, ParseNode | undefined];

export class TreePrinter extends ParseTreeWalker {
    private readonly _moduleNode: ModuleNode;
    private readonly _fileInfo: AnalyzerFileInfo;
    private readonly _program: Program;
    private _tempPrinter: string[];
    private _id: number;

    constructor(program: Program, node: ModuleNode) {
        super();
        this._moduleNode = node;
        this._program = program;
        this._fileInfo = AnalyzerNodeInfo.getFileInfo(node)!;
        this._tempPrinter = [];
        this._id = 0;
    }

    makeHTML(): string {
        const fi = this._fileInfo as any;
        const importLookup = this._fileInfo.importLookup;
        if (this._fileInfo.typingModulePath) {
            fi.lookupResult = importLookup(this._fileInfo.typingModulePath);
        }

        this._tempPrinter = [];
        this._tempPrinter.push(HTMLHeader);
        this._id = 0;
        // this._printObject('program', this._program, []);
        this._printInternally(this._moduleNode);
        this._tempPrinter.push(HTMLFooter);
        return this._tempPrinter.join('\n');
    }

    _printFlowNode(flowNode: FlowNode, flowIds: number[], name = 'flowNode') {
        if (flowIds.includes(flowNode.id)) {
            this._tempPrinter.push(`<li>${name}: ${FlowFlags[flowNode.flags]} Flag (${flowNode.id})</li>`);
            return;
        }

        this._tempPrinter.push(
            `<li><span class="caret">${name}: ${FlowFlags[flowNode.flags]} Flag (${
                flowNode.id
            })</span><ul class="nested">`
        );

        for (const [propKey, property] of Object.entries(flowNode)) {
            switch (propKey) {
                case 'id':
                case 'flags':
                    continue;
                case 'node':
                    this._tempPrinter.push(
                        `<li>node: ${ParseNodeTypeCopy[(property as ParseNode).nodeType]}(${
                            (property as ParseNode).id
                        })</li>`
                    );
                    break;
                case 'flowNode':
                case 'antecedent':
                    this._printFlowNode(property as FlowNode, [flowNode.id, ...flowIds], propKey);
                    break;
                case 'antecedents':
                    (property as FlowNode[]).forEach((node, index) => {
                        this._printFlowNode(node, [flowNode.id, ...flowIds], `antecedents[${index}]`);
                    });
                    break;
                case 'names':
                    (property as string[]).forEach((str, index) => {
                        this._tempPrinter.push(`<li>names[${index}]: ${str}</li>`);
                    });
                    break;
                default:
                    if (typeof property === 'number' || typeof property === 'string' || typeof property === 'boolean') {
                        this._tempPrinter.push(`<li>${propKey}: ${property}</li>`);
                    } else {
                        this._tempPrinter.push(`<li>${propKey}: ${property?.constructor.name}</li>`);
                    }
                    break;
            }
        }

        this._tempPrinter.push(`</ul></li>`);
    }

    _printObject(name: string, obj: object, travList: object[]) {
        if (!obj) {
            this._tempPrinter.push(`<li>${name}: - </li>`);
            return;
        }

        travList.push(obj);
        this._tempPrinter.push(`<li><span class="caret">&lt;${this._id}&gt;${name}</span><ul class="nested">`);
        this._id++;

        let kv;
        let printNest = true;
        if (obj instanceof Map || obj instanceof Set) {
            if (obj.size > 30) {
                printNest = false;
                kv = [['Map count', obj.size]];
            } else {
                kv = obj.entries();
            }
        } else {
            kv = Object.entries(obj);
        }

        for (const [propKey, property] of kv) {
            if (propKey === 'importInfo') {
                const importInfo = property as ImportResult;
                const ii = importInfo as any;

                if (importInfo.isImportFound) {
                    const lookupResults = [];
                    for (const path of importInfo.resolvedPaths) {
                        const result = this._fileInfo.importLookup(path);
                        if (result) {
                            lookupResults.push(result);
                        }
                    }
                    ii.lookupResults = lookupResults;
                }
            }

            switch (typeof property) {
                case 'number':
                case 'string':
                case 'boolean':
                case 'bigint':
                    this._tempPrinter.push(`<li>${propKey}: ${property}</li>`);
                    break;
                case 'symbol':
                    this._tempPrinter.push(`<li>${propKey}: ${String(property)}</li>`);
                    break;
                case 'object':
                    {
                        let propId = travList.findIndex((x) => x === property);
                        if (propId === -1 && printNest) {
                            if (Array.isArray(obj)) {
                                obj.forEach((childObj, index) => {
                                    propId = travList.findIndex((x) => x === childObj);
                                    if (propId === -1) {
                                        this._printObject(`${propKey}[${index}]`, childObj, travList);
                                    } else {
                                        this._tempPrinter.push(
                                            `<li>${propKey}[${index}]*${propId}*: ${String(property)}</li>`
                                        );
                                    }
                                });
                            } else {
                                this._printObject(`${propKey}`, property, travList);
                            }
                        } else {
                            this._tempPrinter.push(`<li>${propKey}*${propId}*: ${String(property)}</li>`);
                        }
                    }
                    break;
                default:
                    this._tempPrinter.push(`<li>${propKey}: ${property?.constructor.name}</li>`);
                    break;
            }
        }

        // travList.pop();

        this._tempPrinter.push('</ul></li>');
    }

    _printInternally(node: ParseNode, printSelf = true): void {
        // print this
        if (printSelf) {
            this._tempPrinter.push(
                `<li><span class="caret">${ParseNodeTypeCopy[node.nodeType]}Node (${node.id})</span><ul class="nested">`
            );
        }

        // print properties of this node
        const childrenToWalk = this.getChildEntries(node);
        const childKeys = childrenToWalk.map(([name, _]) => name.split('[')[0]);

        for (const [propKey, property] of Object.entries(node as object)) {
            if (propKey === 'id' || propKey === 'parent' || propKey === 'nodeType') {
                continue;
            } else if (childKeys.includes(propKey)) {
                continue;
            }

            if (typeof property === 'number' || typeof property === 'string' || typeof property === 'boolean') {
                this._tempPrinter.push(`<li>${propKey}: ${property}</li>`);
            } else if (property && (propKey === 'flowNode' || propKey === 'afterFlowNode')) {
                this._printFlowNode((property as any) as FlowNode, []);
            } else if (property && typeof property === 'object') {
                this._printObject(propKey, property, []);
            } else {
                this._tempPrinter.push(`<li>${propKey}: ${property?.constructor.name}</li>`);
            }
        }

        if (childrenToWalk.length > 0) {
            // print children
            for (const [propKey, childNode] of childrenToWalk) {
                if (childNode) {
                    this._tempPrinter.push(
                        `<li><span class="caret">${propKey}: ${ParseNodeTypeCopy[childNode.nodeType]}Node (${
                            childNode.id
                        })</span><ul class="nested">`
                    );
                    this._printInternally(childNode, false);
                } else {
                    this._tempPrinter.push(`<li><span class="caret">${propKey}: undefined</span><ul class="nested">`);
                }
                this._tempPrinter.push('</ul></li>');
            }
        }

        if (printSelf) {
            this._tempPrinter.push('</ul></li>');
        }
    }

    private _makeEntries(typeName: string, nodeList: ParseNode[]): NodeWithName[] {
        return nodeList.map((node, index) => [`${typeName}[${index}]`, node]);
    }

    getChildEntries(node: ParseNode): NodeWithName[] {
        switch (node.nodeType) {
            case ParseNodeType.Argument:
                if (this.visitArgument(node)) {
                    return [
                        ['name', node.name],
                        ['valueExpression', node.valueExpression],
                    ];
                }
                break;

            case ParseNodeType.Assert:
                if (this.visitAssert(node)) {
                    return [
                        ['testExpression', node.testExpression],
                        ['exceptionExpression', node.exceptionExpression],
                    ];
                }
                break;

            case ParseNodeType.Assignment:
                if (this.visitAssignment(node)) {
                    return [
                        ['leftExpression', node.leftExpression],
                        ['rightExpression', node.rightExpression],
                        ['typeAnnotationComment', node.typeAnnotationComment],
                    ];
                }
                break;

            case ParseNodeType.AssignmentExpression:
                if (this.visitAssignmentExpression(node)) {
                    return [
                        ['name', node.name],
                        ['rightExpression', node.rightExpression],
                    ];
                }
                break;

            case ParseNodeType.AugmentedAssignment:
                if (this.visitAugmentedAssignment(node)) {
                    return [
                        ['leftExpression', node.leftExpression],
                        ['rightExpression', node.rightExpression],
                    ];
                }
                break;

            case ParseNodeType.Await:
                if (this.visitAwait(node)) {
                    return [['expression', node.expression]];
                }
                break;

            case ParseNodeType.BinaryOperation:
                if (this.visitBinaryOperation(node)) {
                    return [
                        ['leftExpression', node.leftExpression],
                        ['rightExpression', node.rightExpression],
                    ];
                }
                break;

            case ParseNodeType.Break:
                if (this.visitBreak(node)) {
                    return [];
                }
                break;

            case ParseNodeType.Call:
                if (this.visitCall(node)) {
                    return [['leftExpression', node.leftExpression], ...this._makeEntries('arguments', node.arguments)];
                }
                break;

            case ParseNodeType.Class:
                if (this.visitClass(node)) {
                    return [
                        ...this._makeEntries('decorators', node.decorators),
                        ['name', node.name],
                        ...this._makeEntries('arguments', node.arguments),
                        ['suite', node.suite],
                    ];
                }
                break;

            case ParseNodeType.Ternary:
                if (this.visitTernary(node)) {
                    return [
                        ['ifExpression', node.ifExpression],
                        ['testExpression', node.testExpression],
                        ['elseExpression', node.elseExpression],
                    ];
                }
                break;

            case ParseNodeType.Constant:
                if (this.visitConstant(node)) {
                    return [];
                }
                break;

            case ParseNodeType.Continue:
                if (this.visitContinue(node)) {
                    return [];
                }
                break;

            case ParseNodeType.Decorator:
                if (this.visitDecorator(node)) {
                    return [['expression', node.expression]];
                }
                break;

            case ParseNodeType.Del:
                if (this.visitDel(node)) {
                    return this._makeEntries('expressions', node.expressions);
                }
                break;

            case ParseNodeType.Dictionary:
                if (this.visitDictionary(node)) {
                    return this._makeEntries('entries', node.entries);
                }
                break;

            case ParseNodeType.DictionaryKeyEntry:
                if (this.visitDictionaryKeyEntry(node)) {
                    return [
                        ['keyExpression', node.keyExpression],
                        ['valueExpression', node.valueExpression],
                    ];
                }
                break;

            case ParseNodeType.DictionaryExpandEntry:
                if (this.visitDictionaryExpandEntry(node)) {
                    return [['expandExpression', node.expandExpression]];
                }
                break;

            case ParseNodeType.Error:
                if (this.visitError(node)) {
                    return [['child', node.child]];
                }
                break;

            case ParseNodeType.If:
                if (this.visitIf(node)) {
                    return [
                        ['testExpression', node.testExpression],
                        ['ifSuite', node.ifSuite],
                        ['elseSuite', node.elseSuite],
                    ];
                }
                break;

            case ParseNodeType.Import:
                if (this.visitImport(node)) {
                    return this._makeEntries('list', node.list);
                }
                break;

            case ParseNodeType.ImportAs:
                if (this.visitImportAs(node)) {
                    return [
                        ['module', node.module],
                        ['alias', node.alias],
                    ];
                }
                break;

            case ParseNodeType.ImportFrom:
                if (this.visitImportFrom(node)) {
                    return [['module', node.module], ...this._makeEntries('imports', node.imports)];
                }
                break;

            case ParseNodeType.ImportFromAs:
                if (this.visitImportFromAs(node)) {
                    return [
                        ['name', node.name],
                        ['alias', node.alias],
                    ];
                }
                break;

            case ParseNodeType.Index:
                if (this.visitIndex(node)) {
                    return [
                        ['baseExpression', node.baseExpression],
                        ['items', node.items],
                    ];
                }
                break;

            case ParseNodeType.IndexItems:
                if (this.visitIndexItems(node)) {
                    return this._makeEntries('items', node.items);
                }
                break;

            case ParseNodeType.Ellipsis:
                if (this.visitEllipsis(node)) {
                    return [];
                }
                break;

            case ParseNodeType.Except:
                if (this.visitExcept(node)) {
                    return [
                        ['typeExpression', node.typeExpression],
                        ['name', node.name],
                        ['exceptSuite', node.exceptSuite],
                    ];
                }
                break;

            case ParseNodeType.For:
                if (this.visitFor(node)) {
                    return [
                        ['targetExpression', node.targetExpression],
                        ['iterableExpression', node.iterableExpression],
                        ['forSuite', node.forSuite],
                        ['elseSuite', node.elseSuite],
                    ];
                }
                break;

            case ParseNodeType.FormatString:
                if (this.visitFormatString(node)) {
                    return this._makeEntries('expressions', node.expressions);
                }
                break;

            case ParseNodeType.Function:
                if (this.visitFunction(node)) {
                    return [
                        ...this._makeEntries('decorators', node.decorators),
                        ['name', node.name],
                        ...this._makeEntries('parameters', node.parameters),
                        ['returnTypeAnnotation', node.returnTypeAnnotation],
                        ['suite', node.suite],
                    ];
                }
                break;

            case ParseNodeType.Global:
                if (this.visitGlobal(node)) {
                    return this._makeEntries('nameList', node.nameList);
                }
                break;

            case ParseNodeType.Lambda:
                if (this.visitLambda(node)) {
                    return [...this._makeEntries('parameters', node.parameters), ['expression', node.expression]];
                }
                break;

            case ParseNodeType.List:
                if (this.visitList(node)) {
                    return this._makeEntries('entries', node.entries);
                }
                break;

            case ParseNodeType.ListComprehension:
                if (this.visitListComprehension(node)) {
                    return [
                        ['expression', node.expression],
                        ...this._makeEntries('comprehensions', node.comprehensions),
                    ];
                }
                break;

            case ParseNodeType.ListComprehensionFor:
                if (this.visitListComprehensionFor(node)) {
                    return [
                        ['targetExpression', node.targetExpression],
                        ['iterableExpression', node.iterableExpression],
                    ];
                }
                break;

            case ParseNodeType.ListComprehensionIf:
                if (this.visitListComprehensionIf(node)) {
                    return [['testExpression', node.testExpression]];
                }
                break;

            case ParseNodeType.MemberAccess:
                if (this.visitMemberAccess(node)) {
                    return [
                        ['leftExpression', node.leftExpression],
                        ['memberName', node.memberName],
                    ];
                }
                break;

            case ParseNodeType.Module:
                if (this.visitModule(node)) {
                    return this._makeEntries('statements', node.statements);
                }
                break;

            case ParseNodeType.ModuleName:
                if (this.visitModuleName(node)) {
                    return this._makeEntries('nameParts', node.nameParts);
                }
                break;

            case ParseNodeType.Name:
                if (this.visitName(node)) {
                    return [];
                }
                break;

            case ParseNodeType.Nonlocal:
                if (this.visitNonlocal(node)) {
                    return this._makeEntries('nameList', node.nameList);
                }
                break;

            case ParseNodeType.Number:
                if (this.visitNumber(node)) {
                    return [];
                }
                break;

            case ParseNodeType.Parameter:
                if (this.visitParameter(node)) {
                    return [
                        ['name', node.name],
                        ['typeAnnotation', node.typeAnnotation],
                        ['defaultValue', node.defaultValue],
                    ];
                }
                break;

            case ParseNodeType.Pass:
                if (this.visitPass(node)) {
                    return [];
                }
                break;

            case ParseNodeType.Raise:
                if (this.visitRaise(node)) {
                    return [
                        ['typeExpression', node.typeExpression],
                        ['valueExpression', node.valueExpression],
                        ['tracebackExpression', node.tracebackExpression],
                    ];
                }
                break;

            case ParseNodeType.Return:
                if (this.visitReturn(node)) {
                    return [['returnExpression', node.returnExpression]];
                }
                break;

            case ParseNodeType.Set:
                if (this.visitSet(node)) {
                    return this._makeEntries('entries', node.entries);
                }
                break;

            case ParseNodeType.Slice:
                if (this.visitSlice(node)) {
                    return [
                        ['startValue', node.startValue],
                        ['endValue', node.endValue],
                        ['stepValue', node.stepValue],
                    ];
                }
                break;

            case ParseNodeType.StatementList:
                if (this.visitStatementList(node)) {
                    return this._makeEntries('statements', node.statements);
                }
                break;

            case ParseNodeType.String:
                if (this.visitString(node)) {
                    return [];
                }
                break;

            case ParseNodeType.StringList:
                if (this.visitStringList(node)) {
                    return [['typeAnnotation', node.typeAnnotation], ...this._makeEntries('strings', node.strings)];
                }
                break;

            case ParseNodeType.Suite:
                if (this.visitSuite(node)) {
                    return this._makeEntries('statements', node.statements);
                }
                break;

            case ParseNodeType.Tuple:
                if (this.visitTuple(node)) {
                    return this._makeEntries('expressions', node.expressions);
                }
                break;

            case ParseNodeType.Try:
                if (this.visitTry(node)) {
                    return [
                        ['trySuite', node.trySuite],
                        ...this._makeEntries('exceptClauses', node.exceptClauses),
                        ['elseSuite', node.elseSuite],
                        ['finallySuite', node.finallySuite],
                    ];
                }
                break;

            case ParseNodeType.TypeAnnotation:
                if (this.visitTypeAnnotation(node)) {
                    return [
                        ['valueExpression', node.valueExpression],
                        ['typeAnnotation', node.typeAnnotation],
                    ];
                }
                break;

            case ParseNodeType.UnaryOperation:
                if (this.visitUnaryOperation(node)) {
                    return [['expression', node.expression]];
                }
                break;

            case ParseNodeType.Unpack:
                if (this.visitUnpack(node)) {
                    return [['expression', node.expression]];
                }
                break;

            case ParseNodeType.While:
                if (this.visitWhile(node)) {
                    return [
                        ['testExpression', node.testExpression],
                        ['whileSuite', node.whileSuite],
                        ['elseSuite', node.elseSuite],
                    ];
                }
                break;

            case ParseNodeType.With:
                if (this.visitWith(node)) {
                    return [...this._makeEntries('withItems', node.withItems), ['suite', node.suite]];
                }
                break;

            case ParseNodeType.WithItem:
                if (this.visitWithItem(node)) {
                    return [
                        ['expression', node.expression],
                        ['target', node.target],
                    ];
                }
                break;

            case ParseNodeType.Yield:
                if (this.visitYield(node)) {
                    return [['expression', node.expression]];
                }
                break;

            case ParseNodeType.YieldFrom:
                if (this.visitYieldFrom(node)) {
                    return [['expression', node.expression]];
                }
                break;

            default:
                fail('Unexpected node type');
                break;
        }

        return [];
    }
}
