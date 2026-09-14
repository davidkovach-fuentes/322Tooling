import random
import typing
from typing import Any, Dict, List, Optional, Tuple

try:
    from typing import TypedDict
except ImportError:

    class TypedDict(dict):
        pass


Meta = typing.NamedTuple("Meta", [("line", int), ("column", int)])


class LanguageError(Exception):
    def __init__(self, line: int, column: int, message: str):
        self.line = line
        self.column = column
        self.message = message

    def __str__(self) -> str:
        return f"{self.line}:{self.column} [error] {self.message}"


class CallsDict(TypedDict):
    calls: List[Tuple[int, float]]


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
        """String representation of the class instance.
        Examples:
            Token(INTEGER, 3)
            Token(PLUS, '+')
            Token(MUL, '*')
        """
        return "Token({type}, {value}, {meta})".format(
            type=self.type, value=repr(self.value), meta=self.meta
        )


class Scope(object):
    def __init__(self, scope_name, parent_scope=None):
        self.scope_name = scope_name
        self.parent_scope = parent_scope
        self._values = dict()

    def __setitem__(self, key, value):
        self._values[key] = value

    def __getitem__(self, item):
        return self._values[item]

    def __contains__(self, key):
        return key in self._values

    def __repr__(self):
        lines = ["{}:{}".format(key, val) for key, val in self._values.items()]
        title = "{}\n".format(self.scope_name)
        return title + "\n".join(lines)


class Frame(object):
    def __init__(self, frame_name, global_scope):
        self.frame_name = frame_name
        self.current_scope = Scope("{}.scope_00".format(frame_name), global_scope)
        self.scopes = [self.current_scope]

    def new_scope(self):
        self.current_scope = Scope(
            "{}{:02d}".format(
                self.current_scope.scope_name[:-2],
                int(self.current_scope.scope_name[-2:]) + 1,
            ),
            self.current_scope,
        )
        self.scopes.append(self.current_scope)

    def del_scope(self):
        current_scope = self.current_scope
        self.current_scope = current_scope.parent_scope
        self.scopes.pop(-1)
        del current_scope

    def __contains__(self, key):
        return key in self.current_scope

    def __repr__(self):
        lines = ["{}\n{}".format(scope, "-" * 40) for scope in self.scopes]

        title = "Frame: {}\n{}\n".format(self.frame_name, "*" * 40)

        return title + "\n".join(lines)


class Stack(object):
    def __init__(self):
        self.frames = list()
        self.current_frame = None

    def __bool__(self):
        return bool(self.frames)

    def new_frame(self, frame_name, global_scope=None):
        frame = Frame(frame_name, global_scope=global_scope)
        self.frames.append(frame)
        self.current_frame = frame

    def del_frame(self):
        self.frames.pop(-1)
        self.current_frame = len(self.frames) and self.frames[-1] or None

    def __repr__(self):
        lines = ["{}".format(frame) for frame in self.frames]
        return "\n".join(lines)


class Memory(object):
    def __init__(self):
        self.global_frame = Frame("GLOBAL_MEMORY", None)
        self.stack = Stack()

    def declare(self, key, value=0):
        ins_scope = (
            self.stack.current_frame.current_scope
            if self.stack.current_frame
            else self.global_frame.current_scope
        )
        ins_scope[key] = value

    def __setitem__(self, key, value):
        ins_scope = (
            self.stack.current_frame.current_scope
            if self.stack.current_frame
            else self.global_frame.current_scope
        )
        curr_scope = ins_scope
        while curr_scope and key not in curr_scope:
            urr_scope = curr_scope.parent_scope
        if curr_scope is None:
            raise LanguageError(0, 0, f"assignment to undeclared variable {key!r}")
        curr_scope[key] = value

    def __getitem__(self, item):
        curr_scope = (
            self.stack.current_frame.current_scope
            if self.stack.current_frame
            else self.global_frame.current_scope
        )
        while curr_scope and item not in curr_scope:
            curr_scope = curr_scope.parent_scope
        return curr_scope[item]

    def new_frame(self, frame_name):
        self.stack.new_frame(frame_name, self.global_frame.current_scope)

    def del_frame(self):
        self.stack.del_frame()

    def new_scope(self):
        self.stack.current_frame.new_scope()

    def del_scope(self):
        self.stack.current_frame.del_scope()

    def __repr__(self):
        return "{}\nStack\n{}\n{}".format(self.global_frame, "=" * 40, self.stack)

    def __str__(self):
        return self.__repr__()


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


# IMPLEMENTING ARRAYS
class ObjList(Node):
    """LB array. Multi-dimensional arrays are nested ObjLists: dims[0] is
    this array's own size, dims[1:] describes each element (itself an
    ObjList) when len(dims) > 1."""

    def __init__(self, dims, line=0):
        Node.__init__(self, line)
        if not dims:
            raise LanguageError(line, 0, "array must have at least one dimension")
        self.size = dims[0]
        if self.size < 0:
            raise LanguageError(line, 0, f"array size {self.size} must be non-negative")
        if len(dims) > 1:
            self.items = [ObjList(dims[1:], line) for _ in range(self.size)]
        else:
            self.items = [0] * self.size

    def _check_bounds(self, index):
        if not isinstance(index, int) or not (0 <= index < self.size):
            raise LanguageError(
                self.line, 0, f"array index {index} out of bounds (size {self.size})"
            )

    def get(self, index):
        self._check_bounds(index)
        return self.items[index]

    def set(self, index, value):
        self._check_bounds(index)
        self.items[index] = value

    def length(self, dim=0):
        if dim == 0:
            return self.size
        if self.size == 0 or not isinstance(self.items[0], ObjList):
            raise LanguageError(self.line, 0, f"array does not have dimension {dim}")
        return self.items[0].length(dim - 1)

    def __repr__(self):
        slots = [str(self.size)]
        for item in self.items:
            slots.append(repr(item) if isinstance(item, ObjList) else str(item))
        return "{{{}:{}, {}}}".format(self._tag, len(slots), ", ".join(slots))

    _tag = "s"


# IMPLEMENTING TUPLES
class Tuple(ObjList):
    """LB tuple. Backed by the same fixed-size storage as ObjList — same
    allocation, bounds-checked get/set, and length() — since a tuple is
    just a flat, non-nested fixed-size container in LB."""

    _tag = "t"


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
        self.args = args  # a list of Param nodes


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
        self.params = params  # a list of Param nodes
        self.body = body


class FunctionBody(Node):
    def __init__(self, children, line):
        Node.__init__(self, line)
        self.children = children


class Program(Node):
    def __init__(self, declarations, line):
        Node.__init__(self, line)
        self.children = declarations


###############################################################################
#                                                                             #
#  AST visitors (walkers)                                                     #
#                                                                             #
###############################################################################


class NodeVisitor(object):
    def visit(self, node):
        method_name = "visit_" + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception("No visit_{} method".format(type(node).__name__))
