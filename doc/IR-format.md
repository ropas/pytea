# Format of PyTea Internal Representation

PyTea 내부에서 사용하는 IR은 LISP 형태로 변환되어 입출력을 수행할 수 있다.

각 statement 및 expression은 전부 `(<constructor> [<source>]? <args>*)`와 같은 형태를 따른다.

예를 들어 `x = 1 + 2`는 다음과 같은 IR로 번역된다.

```lisp
(source-map "/home/path/to/script/test.py"
(assign [0:0:9]
  (var [0:0:1] x)
  (binop [0:4:9] + (const [0:4:5] 1) (const [0:8:9] 2))))
```

이 형태를 기반으로 한 IR 입출력의 구현은 `IRReader.ts`에 있으며, LISP 형태가 아닌 Python 형태로 좀 더 알아보기 쉽게 출력하려면 `ThStmt.toString` 함수를 사용하면 된다.

## Syntax

```bnf
<output> ::= <source-map>*

<source-map> ::= (source-map <path> <stmt>)

<path> ::= <string>

<stmt> ::=
    | <stmt-assign>
    | <stmt-let>
    | <stmt-fundef>
    | <stmt-if>
    | <stmt-forin>
    | <stmt-pass>
    | <stmt-return>
    | <stmt-continue>
    | <stmt-break>
    | <stmt-seq>
    | <stmt-expr>

<expr> ::=
    | <expr-attr>
    | <expr-subscr>
    | <expr-call>
    | <expr-libcall>
    | <expr-object>
    | <expr-tuple>
    | <expr-binop>
    | <expr-unaryop>
    | <expr-name>
    | <expr-const>

<stmt-assign> ::= (assign <source>? <expr> <expr>) // a = b

<stmt-let> ::= (let <source>? (<string> <expr>?) <stmt>)  // let <string> = <expr> in <stmt>

<stmt-fundef> ::= (fundef <source>? <string> (<string>*) <stmt> <stmt>)  // let <string> <params> = <stmt> in <stmt>

<stmt-if> ::= (if <source>? <expr> <stmt> <stmt>)  // condition, if-stmt, else-stmt

<stmt-forin> ::= (for <source>? <expr-name> <expr> <stmt>)  // iter-var, iter-list, body

<stmt-pass> ::= (pass <source>?)
<stmt-return> ::= (return <source>? <expr>)
<stmt-continue> ::= (continue <source>?)
<stmt-break> ::= (break <source>?)
<stmt-expr> ::= <expr>
<stmt-seq> ::= (<stmt>+)

<expr-attr> ::= (attr <source>? <expr> <string>)  // a.b
<expr-subscr> ::= (subs <source>? <expr> <expr>)  // a[b]
<expr-call> ::= (call <source>? <expr> <expr>*)  // function, arguments
<expr-libcall> ::= (libcall <source>? <string> <argument>*) // libcall-name, keyworded-arguments
<expr-object> ::= (object <source>?)
<expr-tuple> ::= (tuple <source>? <expr>+)
<expr-binop> ::= (bop <source>? <binop-type> <expr> <expr>)
<expr-unaryop> ::= (uop <source>? <uop-type> <expr>)
<expr-name> ::= (var <source>? <string>)

<expr-const> ::=
    | <const-int>
    | <const-float>
    | <const-string>
    | <const-bool>
    | <const-none>

<argument> ::= (<string> <expr>) // argument-key, argument-value

<const-int> ::= (int <source>? <int>)
<const-float> ::= (float <source>? <float>)
<const-string> ::= (str <source>? <string>)
<const-bool> ::= (bool <source>? <bool>)
<const-none> ::= (none <source>?)

<binop-type> ::= < | <= | = | != | and | or | is | isnot | in | notin | + | - | * | % | ** | / | //
<uop-type> ::= not | -


<source> ::= [ <int> : <int> ] // start - end character position (0-start, exclusive)

<int> ::= (0-9)+
<float> ::= (0-9)+ . (0-9)*
<bool> ::= True | False
<string> ::= double-quoted string (with escaped chars like \", \', \\, \r, \n, \t, ...)
```