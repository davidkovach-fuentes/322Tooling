from __future__ import print_function

import array as _array
import sys
import typing
from typing import Any, Dict, List, Optional, Tuple as PyTuple

try:
    from typing import TypedDict
except ImportError:

    class TypedDict(dict):
        pass


Meta = typing.NamedTuple("Meta", [("line", int), ("column", int)])


def _to_int64(val: int) -> int:
    val = val & 0xFFFFFFFFFFFFFFFF
    if val >= 0x8000000000000000:
        val -= 0x10000000000000000
    return val


class LanguageError(Exception):
    def __init__(self, line: int, column: int, message: str):
        self.line = line
        self.column = column
        self.message = message

    def __str__(self) -> str:
        return f"{self.line}:{self.column} [error] {self.message}"


class CallsDict(TypedDict):
    calls: List[PyTuple[int, float]]


class Token:
    kind = "token"
    data = "token"
    children: List[Any] = []

    def __init__(self, value: str, meta: Meta) -> None:
        self.value = value
        self.meta = meta

    def __eq__(self, other) -> bool:
        return self.value == other

    def __str__(self):
        return "Token({type}, {value}, {meta})".format(
            type=self.type, value=repr(self.value), meta=self.meta
        )


class Scope(object):
    __slots__ = ("scope_name", "parent_scope", "_values")

    def __init__(self, scope_name="", parent_scope=None):
        self.scope_name = scope_name
        self.parent_scope = parent_scope
        self._values = {}

    def __setitem__(self, key, value):
        self._values[key] = value

    def __getitem__(self, item):
        return self._values[item]

    def __contains__(self, key):
        return key in self._values


class Frame(object):
    __slots__ = ("frame_name", "current_scope", "scopes")

    def __init__(self, frame_name, global_scope):
        self.frame_name = frame_name
        self.current_scope = Scope(frame_name, global_scope)
        self.scopes = [self.current_scope]

    def new_scope(self):
        self.current_scope = Scope("", self.current_scope)
        self.scopes.append(self.current_scope)

    def del_scope(self):
        current_scope = self.current_scope
        self.current_scope = current_scope.parent_scope
        self.scopes.pop()

    def __contains__(self, key):
        return key in self.current_scope


class Stack(object):
    __slots__ = ("frames", "current_frame")

    def __init__(self):
        self.frames = []
        self.current_frame = None

    def __bool__(self):
        return bool(self.frames)

    def new_frame(self, frame_name, global_scope=None):
        frame = Frame(frame_name, global_scope=global_scope)
        self.frames.append(frame)
        self.current_frame = frame

    def del_frame(self):
        self.frames.pop()
        self.current_frame = self.frames[-1] if self.frames else None


class Memory(object):
    __slots__ = ("global_frame", "stack")

    def __init__(self):
        self.global_frame = Frame("GLOBAL_MEMORY", None)
        self.stack = Stack()

    def declare(self, key, value=0):
        curr_scope = (
            self.stack.current_frame.current_scope
            if self.stack.current_frame
            else self.global_frame.current_scope
        )
        curr_scope._values[key] = value

    def __setitem__(self, key, value):
        curr_scope = (
            self.stack.current_frame.current_scope
            if self.stack.current_frame
            else self.global_frame.current_scope
        )
        if key in curr_scope._values:
            curr_scope._values[key] = value
            return
        s = curr_scope
        while s is not None:
            if key in s._values:
                s._values[key] = value
                return
            s = s.parent_scope
        curr_scope._values[key] = value

    def __contains__(self, item):
        curr_scope = (
            self.stack.current_frame.current_scope
            if self.stack.current_frame
            else self.global_frame.current_scope
        )
        while curr_scope is not None:
            if item in curr_scope._values:
                return True
            curr_scope = curr_scope.parent_scope
        return False

    def __getitem__(self, item):
        curr_scope = (
            self.stack.current_frame.current_scope
            if self.stack.current_frame
            else self.global_frame.current_scope
        )
        if item in curr_scope._values:
            return curr_scope._values[item]
        s = curr_scope.parent_scope
        while s is not None:
            if item in s._values:
                return s._values[item]
            s = s.parent_scope
        return 0

    def new_frame(self, frame_name):
        self.stack.new_frame(frame_name, self.global_frame.current_scope)

    def del_frame(self):
        self.stack.del_frame()

    def new_scope(self):
        if self.stack.current_frame:
            self.stack.current_frame.new_scope()

    def del_scope(self):
        if self.stack.current_frame:
            self.stack.current_frame.del_scope()


class Node(object):
    def __init__(self, line):
        self.line = line


class NoOp(Node):
    pass


class Num(Node):
    def __init__(self, token, line):
        Node.__init__(self, line)
        self.token = token
        self.value = token.value


class String(Node):
    def __init__(self, token, line):
        Node.__init__(self, line)
        self.token = token
        self.value = token.value


class Type(Node):
    def __init__(self, token, line):
        Node.__init__(self, line)
        self.token = token
        self.value = token.value


class Var(Node):
    def __init__(self, token, line):
        Node.__init__(self, line)
        self.token = token
        self.value = token.value


class NDArray(Node):
    _tag = "s"

    def __init__(self, dims, line=0):
        Node.__init__(self, line)
        if not dims:
            raise LanguageError(line, 0, "array must have at least one dimension")
        clean_dims = []
        for d in dims:
            d_val = d.address if isinstance(d, NDArray) else int(d)
            if d_val < 0:
                raise LanguageError(line, 0, f"array size {d_val} must be non-negative")
            clean_dims.append(d_val)
        self.shape = tuple(clean_dims)
        self.strides = self._compute_strides(clean_dims)
        self.ndim = len(clean_dims)
        if self.ndim == 1:
            self.stride0 = 1
            self.dim0 = clean_dims[0]
        elif self.ndim == 2:
            self.stride0 = self.strides[0]
            self.stride1 = self.strides[1]
            self.dim0 = clean_dims[0]
            self.dim1 = clean_dims[1]

        total = 1
        for d in clean_dims:
            total *= d
        self.data = [0] * total
        self.address = (id(self) << 1) | 1

    @staticmethod
    def _compute_strides(dims):
        strides = [1] * len(dims)
        for i in range(len(dims) - 2, -1, -1):
            strides[i] = strides[i + 1] * dims[i + 1]
        return strides

    @property
    def size(self):
        return self.shape[0]

    def _flat_index(self, indices):
        if len(indices) != len(self.shape):
            raise LanguageError(
                self.line, 0,
                f"expected {len(self.shape)} index/indices, got {len(indices)}",
            )
        flat = 0
        for idx, dim, stride in zip(indices, self.shape, self.strides):
            idx_val = idx.address if isinstance(idx, NDArray) else int(idx)
            if not (0 <= idx_val < dim):
                raise LanguageError(
                    self.line, 0, f"array index {idx_val} out of bounds (size {dim})"
                )
            flat += idx_val * stride
        return flat

    def get(self, *indices):
        if len(indices) == 2:
            i0, i1 = indices
            i0 = i0.address if type(i0) is NDArray else i0
            i1 = i1.address if type(i1) is NDArray else i1
            if not (0 <= i0 < self.dim0 and 0 <= i1 < self.dim1):
                raise LanguageError(self.line, 0, "array index out of bounds")
            return self.data[i0 * self.stride0 + i1]
        elif len(indices) == 1:
            i0 = indices[0]
            i0 = i0.address if type(i0) is NDArray else i0
            if not (0 <= i0 < self.dim0):
                raise LanguageError(self.line, 0, "array index out of bounds")
            return self.data[i0]
        return self.data[self._flat_index(indices)]

    def set(self, *indices_and_value):
        if len(indices_and_value) == 3:
            i0, i1, value = indices_and_value
            i0 = i0.address if type(i0) is NDArray else i0
            i1 = i1.address if type(i1) is NDArray else i1
            if not (0 <= i0 < self.dim0 and 0 <= i1 < self.dim1):
                raise LanguageError(self.line, 0, "array index out of bounds")
            val = value if type(value) in (NDArray, Tuple) else _to_int64(int(value))
            self.data[i0 * self.stride0 + i1] = val
        elif len(indices_and_value) == 2:
            i0, value = indices_and_value
            i0 = i0.address if type(i0) is NDArray else i0
            if not (0 <= i0 < self.dim0):
                raise LanguageError(self.line, 0, "array index out of bounds")
            val = value if type(value) in (NDArray, Tuple) else _to_int64(int(value))
            self.data[i0] = val
        else:
            *indices, value = indices_and_value
            val = value if type(value) in (NDArray, Tuple) else _to_int64(int(value))
            self.data[self._flat_index(indices)] = val

    def length(self, dim=0):
        dim_val = dim.address if isinstance(dim, NDArray) else int(dim)
        if not (0 <= dim_val < len(self.shape)):
            return 0
        return self.shape[dim_val]

    def __repr__(self):
        slots = [str(d) for d in self.shape] + [
            repr(v) if isinstance(v, (NDArray, Tuple)) else str(v)
            for v in self.data
        ]
        return "{{{}:{}, {}}}".format(self._tag, len(slots), ", ".join(slots))


class Tuple(NDArray):
    _tag = "s"

    def __repr__(self):
        slots = [
            repr(v) if isinstance(v, (NDArray, Tuple)) else str(v)
            for v in self.data
        ]
        return "{{{}:{}, {}}}".format(self._tag, len(slots), ", ".join(slots))


class BinOp(Node):
    def __init__(self, left, op, right, line):
        Node.__init__(self, line)
        self.left = left
        self.token = self.op = op
        self.right = right


class UnOp(Node):
    def __init__(self, op, expr, line, prefix=True):
        Node.__init__(self, line)
        self.token = self.op = op
        self.expr = expr
        self.prefix = prefix


class TerOp(Node):
    def __init__(self, condition, texpression, fexpression, line):
        Node.__init__(self, line)
        self.condition = condition
        self.texpression = texpression
        self.fexpression = fexpression


class Assign(Node):
    def __init__(self, left, op, right, line):
        Node.__init__(self, line)
        self.left = left
        self.token = self.op = op
        self.right = right


class Expression(Node):
    def __init__(self, children, line):
        Node.__init__(self, line)
        self.children = children


class FunctionCall(Node):
    def __init__(self, name, args, line):
        Node.__init__(self, line)
        self.name = name
        self.args = args


class IfStmt(Node):
    def __init__(self, condition, tbody, line, fbody=None):
        Node.__init__(self, line)
        self.condition = condition
        self.tbody = tbody
        self.fbody = fbody


class WhileStmt(Node):
    def __init__(self, condition, body, line):
        Node.__init__(self, line)
        self.condition = condition
        self.body = body


class DoWhileStmt(WhileStmt):
    pass


class ReturnStmt(Node):
    def __init__(self, expression, line):
        Node.__init__(self, line)
        self.expression = expression


class BreakStmt(Node):
    pass


class ContinueStmt(Node):
    pass


class ForStmt(Node):
    def __init__(self, setup, condition, increment, body, line):
        Node.__init__(self, line)
        self.setup = setup
        self.condition = condition
        self.increment = increment
        self.body = body


class CompoundStmt(Node):
    def __init__(self, children, line):
        Node.__init__(self, line)
        self.children = children


class VarDecl(Node):
    def __init__(self, var_node, type_node, line):
        Node.__init__(self, line)
        self.var_node = var_node
        self.type_node = type_node


class IncludeLibrary(Node):
    def __init__(self, library_name, line):
        Node.__init__(self, line)
        self.library_name = library_name


class Param(Node):
    def __init__(self, type_node, var_node, line):
        Node.__init__(self, line)
        self.var_node = var_node
        self.type_node = type_node


class FunctionDecl(Node):
    def __init__(self, type_node, func_name, params, body, line):
        Node.__init__(self, line)
        self.type_node = type_node
        self.func_name = func_name
        self.params = params
        self.body = body


class FunctionBody(Node):
    def __init__(self, children, line):
        Node.__init__(self, line)
        self.children = children


class Program(Node):
    def __init__(self, declarations, line):
        Node.__init__(self, line)
        self.children = declarations


class NodeVisitor(object):
    def visit(self, node):
        method_name = "visit_" + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception("No visit_{} method".format(type(node).__name__))


def _flatten_stmts(stmts, out):
    for stmt in stmts:
        if stmt[0] == "scope":
            _flatten_stmts(stmt[1], out)
        else:
            out.append(stmt)


def _compile_fn(fn_ast):
    _, ret_type, name, params, body = fn_ast
    raw_stmts = body[1] if body[0] == "scope" else [body]
    flat = []
    _flatten_stmts(raw_stmts, flat)

    label_map = {}
    code = []
    for stmt in flat:
        if stmt[0] == "label":
            label_map[stmt[1]] = len(code)
        else:
            code.append(stmt)

    resolved_code = []
    for stmt in code:
        tag = stmt[0]
        if tag == "if":
            resolved_code.append(("if", stmt[1], label_map[stmt[2]], label_map[stmt[3]]))
        elif tag == "while":
            resolved_code.append(("while", stmt[1], label_map[stmt[2]], label_map[stmt[3]]))
        elif tag == "goto":
            resolved_code.append(("goto", label_map[stmt[1]]))
        else:
            resolved_code.append(stmt)

    return params, resolved_code


def eval_expr(expr, env, globals_dict, interpreter):
    t = type(expr)
    if t is int:
        return expr
    if t is str:
        return env[expr] if expr in env else globals_dict.get(expr, 0)
    if t is not tuple:
        return expr

    tag = expr[0]

    if tag == "var":
        v = expr[1]
        return env[v] if v in env else (globals_dict[v] if v in globals_dict else interpreter.memory[v])

    if tag == "int":
        return expr[1]

    if tag == "binary":
        lval = eval_expr(expr[1], env, globals_dict, interpreter)
        rval = eval_expr(expr[3], env, globals_dict, interpreter)
        op = expr[2]

        if type(lval) is int and type(rval) is int:
            if op == "+":
                res = lval + rval
                if res > 9223372036854775807 or res < -9223372036854775808:
                    res = (res & 0xFFFFFFFFFFFFFFFF)
                    if res >= 0x8000000000000000:
                        res -= 0x10000000000000000
                return res
            if op == "-":
                res = lval - rval
                if res > 9223372036854775807 or res < -9223372036854775808:
                    res = (res & 0xFFFFFFFFFFFFFFFF)
                    if res >= 0x8000000000000000:
                        res -= 0x10000000000000000
                return res
            if op == "*":
                res = lval * rval
                if res > 9223372036854775807 or res < -9223372036854775808:
                    res = (res & 0xFFFFFFFFFFFFFFFF)
                    if res >= 0x8000000000000000:
                        res -= 0x10000000000000000
                return res
            if op in ("==", "="):
                return 1 if lval == rval else 0
            if op == "<":
                return 1 if lval < rval else 0
            if op == "<=":
                return 1 if lval <= rval else 0
            if op == ">":
                return 1 if lval > rval else 0
            if op == ">=":
                return 1 if lval >= rval else 0
            if op == "!=":
                return 1 if lval != rval else 0
            if op == "&":
                return lval & rval
            if op == "/":
                return (lval // rval) if rval != 0 else 0
            if op == "%":
                return (lval % rval) if rval != 0 else 0
            if op == "|":
                return lval | rval
            if op == "^":
                return lval ^ rval
            if op == "<<":
                res = lval << (rval & 63)
                if res > 9223372036854775807 or res < -9223372036854775808:
                    res = (res & 0xFFFFFFFFFFFFFFFF)
                    if res >= 0x8000000000000000:
                        res -= 0x10000000000000000
                return res
            if op == ">>":
                return lval >> (rval & 63)

        return interpreter._eval_bop(lval, op, rval)

    if tag == "array_read":
        target = expr[1]
        if type(target) is str:
            arr = env[target] if target in env else (globals_dict[target] if target in globals_dict else interpreter.memory[target])
        else:
            arr = eval_expr(target, env, globals_dict, interpreter)
        idx_vals = [eval_expr(i, env, globals_dict, interpreter) for i in expr[2]]
        return arr.get(*idx_vals)

    if tag == "bool":
        return 1 if expr[1] else 0
    if tag == "string":
        return expr[1]

    if tag == "length":
        target = expr[2]
        if type(target) is str:
            arr = env[target] if target in env else (globals_dict[target] if target in globals_dict else interpreter.memory[target])
        else:
            arr = eval_expr(target, env, globals_dict, interpreter)
        dim = eval_expr(expr[3], env, globals_dict, interpreter) if expr[3] is not None else 0
        return arr.length(dim)

    if tag == "call":
        args = [eval_expr(e, env, globals_dict, interpreter) for e in expr[2]]
        return interpreter.call_function(expr[1], args)

    raise LanguageError(0, 0, f"Unknown expression tag {tag!r}")


def exec_code(code, env, globals_dict, interpreter):
    pc = 0
    n_instr = len(code)
    while pc < n_instr:
        stmt = code[pc]
        tag = stmt[0]

        if tag == "assign":
            var_name = stmt[1]
            val = eval_expr(stmt[2], env, globals_dict, interpreter)
            if var_name in env:
                env[var_name] = val
            elif var_name in globals_dict:
                globals_dict[var_name] = val
            else:
                env[var_name] = val
            pc += 1

        elif tag == "if":
            cond = eval_expr(stmt[1], env, globals_dict, interpreter)
            pc = stmt[2] if cond else stmt[3]

        elif tag == "while":
            cond = eval_expr(stmt[1], env, globals_dict, interpreter)
            pc = stmt[2] if cond else stmt[3]

        elif tag == "goto":
            pc = stmt[1]

        elif tag == "decl":
            env[stmt[2]] = 0
            pc += 1

        elif tag == "array_write":
            target = stmt[1]
            if type(target) is str:
                arr = env[target] if target in env else (globals_dict[target] if target in globals_dict else interpreter.memory[target])
            else:
                arr = eval_expr(target, env, globals_dict, interpreter)
            indices = [eval_expr(e, env, globals_dict, interpreter) for e in stmt[2]]
            val = eval_expr(stmt[3], env, globals_dict, interpreter)
            arr.set(*indices, val)
            pc += 1

        elif tag == "new_array":
            dims = [eval_expr(e, env, globals_dict, interpreter) for e in stmt[2]]
            env[stmt[1]] = NDArray(dims)
            pc += 1

        elif tag == "new_tuple":
            dims = [eval_expr(e, env, globals_dict, interpreter) for e in stmt[2]]
            env[stmt[1]] = Tuple(dims)
            pc += 1

        elif tag == "call":
            args = [eval_expr(e, env, globals_dict, interpreter) for e in stmt[2]]
            interpreter.call_function(stmt[1], args)
            pc += 1

        elif tag == "return":
            expr = stmt[1]
            return eval_expr(expr, env, globals_dict, interpreter) if expr is not None else None

        elif tag in ("break", "continue", "label"):
            pc += 1
        else:
            pc += 1
    return None


class Interpreter(object):
    def __init__(self, ast):
        self.ast = ast
        self.memory = Memory()
        self.compiled_functions = {}
        self.globals = self.memory.global_frame.current_scope._values
        self.runtime_functions = {
            "print": self._builtin_print,
            "input": self._builtin_input,
        }

    def run(self):
        _, fns = self.ast
        for fn in fns:
            fn_name = fn[2]
            self.compiled_functions[fn_name] = _compile_fn(fn)

        if "main" not in self.compiled_functions:
            raise LanguageError(0, 0, "Missing main function")

        return self.call_function("main", [])

    def _builtin_print(self, args):
        for arg in args:
            if isinstance(arg, (NDArray, Tuple)):
                print(repr(arg))
            elif isinstance(arg, bool):
                print("1" if arg else "0")
            else:
                print(_to_int64(arg))
        return 0

    def _builtin_input(self, args):
        val = input()
        return _to_int64(int(val.strip()))

    def call_function(self, name, args):
        if name in self.runtime_functions:
            return self.runtime_functions[name](args)

        if name not in self.compiled_functions:
            raise LanguageError(0, 0, f"Undefined function {name!r}")

        params, code = self.compiled_functions[name]
        if len(args) != len(params):
            raise LanguageError(
                0, 0, f"Function {name} expected {len(params)} args, got {len(args)}"
            )

        self.memory.new_frame(name)
        curr_scope = self.memory.stack.current_frame.current_scope
        env = curr_scope._values
        for (_, _, p_name), arg_val in zip(params, args):
            env[p_name] = arg_val

        try:
            return exec_code(code, env, self.globals, self)
        finally:
            self.memory.del_frame()

    def _eval_bop(self, lval, op, rval):
        lval = lval.address if isinstance(lval, NDArray) else int(lval)
        rval = rval.address if isinstance(rval, NDArray) else int(rval)

        if op == "+":
            return _to_int64(lval + rval)
        if op == "-":
            return _to_int64(lval - rval)
        if op == "*":
            return _to_int64(lval * rval)
        if op == "/":
            return _to_int64(lval // rval) if rval != 0 else 0
        if op == "%":
            return _to_int64(lval % rval) if rval != 0 else 0
        if op == "&":
            return _to_int64(lval & rval)
        if op == "|":
            return _to_int64(lval | rval)
        if op == "^":
            return _to_int64(lval ^ rval)
        if op == "<<":
            return _to_int64(lval << (rval & 63))
        if op == ">>":
            return _to_int64(lval >> (rval & 63))
        if op in ("==", "="):
            return 1 if lval == rval else 0
        if op == "!=":
            return 1 if lval != rval else 0
        if op == "<":
            return 1 if lval < rval else 0
        if op == "<=":
            return 1 if lval <= rval else 0
        if op == ">":
            return 1 if lval > rval else 0
        if op == ">=":
            return 1 if lval >= rval else 0
        raise LanguageError(0, 0, f"Unsupported binary operator {op!r}")
