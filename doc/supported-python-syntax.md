# Supported Python syntax

## Supported

PyTea’s Python parsing is dependent on the Pyright type checker. From the Python syntax, we can analyze these statements or expressions below:

- Boolean, umber, (unformatted) String literal, Ellipsis (`...`), and its classes
- Assignment, Member access, Indexing
- Unary/Binary operation, operator overloading
- `if ...: ... elif ...: ... else: ...`
- `for ... in ...: ...`
- `break`, `continue`, `return`, `pass`
- Function/Closure definition
- Function call
- Variadic, keyword arguments (*args, **kwargs)
- Class (single-inheritance)
- `__getitem__`, `__init__`, `__call__`
- `global`, `nonlocal`
- `lambda`, ternary operator (`... if ... else ...`)
- Tuple, List, Dictionary initialization
- List comprehension, List/Dictionary unpacking, List slicing
- Import local script

## Unsupported

The statements below are unsupported by PyTea. These statements will be ignored or raise an error:

- Syntax after Python 3.8 (e.g. `:=` [PEP 572](https://www.python.org/dev/peps/pep-0572/))
- Overriding augmented assignment (e.g. overriding `__iadd__`)
- `async`, `await`
- `for ...: ... else: ...`
- Custom `__getattr__`, `__setattr__`, `__setitem__`
- Class (multiple-inheritance)
- Decorator, `@staticmethod`, `@classmethod`
- Generator (`yield`)
- Formatted string (e.g., `f"{...}"`)
- Keyword/Positional-only parameters (e.g., `def f(x, /, y, *, z)`)
- Set, Frozen set
- Type annotation
- Import 3rd-party (pip) library
- External side-effects (File I/O, Networking, . . .)

## Partially supported

The statements below can be parsed or analyzed, but we do not guarantee their analysis are correct. We will describe their behavior in PyTea.
- `while`: PyTea assumes that every loop is finite. The maximum iteration counts will be bound to 300.
- Iterator protocol (`__iter__`, `__next__`): Support of iterator protocol is still premature. Because of the finite loop
assumption, an iterator instance should have constant length too. (i.e., iterator should implement `__len__`)
- `raise`: Exception handling is not supported. If an exception is raised, the analyzer will be terminated with an
error.
- `try: A except ...: ... else: ... finally: B`: This will be translated to A; B
- `with A: B`: This will be translated to A; B
- `assert ...`: If assert condition is definitely false, PyTea will report an error without handling exception.
- `del ...`: Only removing variable is supported.

For the Python builtin and 3rd-party libraries, See bin/dist/pylib directory.
