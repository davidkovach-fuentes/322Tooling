from __future__ import print_function

import sys
from pathlib import Path

from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor
from interpreter.compiler import compile_program
from interpreter.vm import VM


_GRAMMAR_PATH = Path(__file__).with_name("LB.PEG")
grammar = Grammar(_GRAMMAR_PATH.read_text(encoding="utf-8"))


def parse(source):
    return grammar.parse(source)


def run(source):
    functions = compile_program(interpret(source))
    vm = VM(functions)
    if "main" not in functions:
        raise LanguageError(0, 0, "Missing main function")
    return vm.call_function("main", [])


def run_ast_interpreter(source):
    from interpreter.main import Interpreter

    return Interpreter(interpret(source)).run()


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: LBi.py <file.LB> [--ast]", file=sys.stderr)
        return 2
    path = argv[0]
    with open(path, "r") as f:
        source = f.read()
    if len(argv) > 1 and argv[1] == "--ast":
        print(interpret(source))
    else:
        run(source)
    return 0


class LanguageError(Exception):
    def __init__(self, line: int, column: int, message: str):
        self.line = line
        self.column = column
        self.message = message

    def __str__(self) -> str:
        return f"{self.line}:{self.column} [error] {self.message}"


def _flatten_choice(visited_children):
    if isinstance(visited_children, list) and len(visited_children) == 1:
        return visited_children[0]
    return visited_children


def _optional(visited_children):
    if isinstance(visited_children, list) and visited_children:
        return visited_children[0]
    return None


def _var_name(var):
    """Unwrap a visit_variable result (\"var\", name) to the bare name string."""
    if isinstance(var, tuple) and var[0] == "var":
        return var[1]
    return var


class LBVisitor(NodeVisitor):
    def visit_program(self, node, visited_children):
        _, fn_pairs, _ = visited_children
        functions = [pair[0] for pair in fn_pairs]
        return ("program", functions)

    def visit_function(self, node, visited_children):
        typ, _, name, _, _, _, params_opt, _, _, _, body = visited_children
        params = _optional(params_opt) or []
        return ("function", typ, name, params, body)

    def visit_type(self, node, visited_children):
        return _flatten_choice(visited_children)

    def visit_void(self, node, visited_children):
        return "void"

    def visit_int64(self, node, visited_children):
        return "int64"

    def visit_arraytype(self, node, visited_children):
        base, brackets = visited_children
        return ("arraytype", base, len(brackets))

    def visit_tupletype(self, node, visited_children):
        return "tuple"

    def visit_codetype(self, node, visited_children):
        return "code"

    def visit_index(self, node, visited_children):
        _, _, expr, _, _ = visited_children
        return expr

    def visit_array_access(self, node, visited_children):
        var, index_groups = visited_children
        indices = list(index_groups)
        return ("array_access", _var_name(var), indices)

    def visit_args(self, node, visited_children):
        first, rest = visited_children
        return [first] + [group[-1] for group in rest]

    def visit_lExpr(self, node, visited_children):
        return _flatten_choice(visited_children)

    def visit_rExpr(self, node, visited_children):
        return _flatten_choice(visited_children)

    def visit_newArrayExpr(self, node, visited_children):
        _, _, _, _, _, _, args_list, _, _ = visited_children
        return ("new_array_expr", args_list)

    def visit_newTupleExpr(self, node, visited_children):
        _, _, _, _, _, _, args_list, _, _ = visited_children
        return ("new_tuple_expr", args_list)

    def visit_lenExpr(self, node, visited_children):
        _, _, arr_var, dim_opt = visited_children
        dim_group = _optional(dim_opt)
        dim_expr = dim_group[-1] if dim_group else None
        return ("len_expr", _var_name(arr_var), dim_expr)

    def visit_assign_stmt(self, node, visited_children):
        left, _, _, _, right, _ = visited_children

        if isinstance(right, tuple) and right[:1] == ("new_array_expr",):
            return ("new_array", _var_name(left), right[1])
        if isinstance(right, tuple) and right[:1] == ("new_tuple_expr",):
            return ("new_tuple", _var_name(left), right[1])
        if isinstance(right, tuple) and right[:1] == ("len_expr",):
            _, arr_name, dim_expr = right
            return ("length", _var_name(left), arr_name, dim_expr)
        if isinstance(right, tuple) and right[:1] == ("array_access",):
            _, arr_name, indices = right
            return ("array_read", _var_name(left), arr_name, indices)
        if isinstance(left, tuple) and left[:1] == ("array_access",):
            _, arr_name, indices = left
            return ("array_write", arr_name, indices, right)
        return ("assign", _var_name(left), right)

    def visit_incdec(self, node, visited_children):
        return node.text

    def visit_unary_stmt(self, node, visited_children):
        var, _, op, _ = visited_children
        name = _var_name(var)
        bop = "+" if op == "++" else "-"
        return ("assign", name, ("binary", ("var", name), bop, ("int", 1)))

    def visit_parameters(self, node, visited_children):
        first, rest = visited_children
        return [first] + [group[-1] for group in rest]

    def visit_parameter(self, node, visited_children):
        typ, _, var = visited_children
        return ("param", typ, _var_name(var))

    def visit_decl(self, node, visited_children):
        typ, _, first_var, extra, _ = visited_children
        names = [_var_name(first_var)] + [_var_name(group[-1]) for group in extra]
        return [("decl", typ, name) for name in names]

    def visit_scope(self, node, visited_children):
        _, _, stmt_pairs, _ = visited_children
        stmts = []
        for pair in stmt_pairs:
            item = pair[0]
            if isinstance(item, list):
                stmts.extend(item)
            else:
                stmts.append(item)
        return ("scope", stmts)

    def visit_statement(self, node, visited_children):
        return _flatten_choice(visited_children)

    def visit_statements(self, node, visited_children):
        return _flatten_choice(visited_children)

    def visit_call_stmt(self, node, visited_children):
        call_val, _ = visited_children
        return call_val

    def visit_if(self, node, visited_children):
        _, _, _, _, cond, _, _, _, tlabel, _, flabel, _ = visited_children
        return ("if", cond, tlabel, flabel)

    def visit_while(self, node, visited_children):
        _, _, _, _, cond, _, _, _, blabel, _, elabel, _ = visited_children
        return ("while", cond, blabel, elabel)

    def visit_goto(self, node, visited_children):
        _, _, target, _ = visited_children
        return ("goto", target)

    def visit_label(self, node, visited_children):
        _, name = visited_children
        return ("label", name)

    def visit_label_stmt(self, node, visited_children):
        label_val, _ = visited_children
        return label_val

    def visit_break(self, node, visited_children):
        return ("break",)

    def visit_continue(self, node, visited_children):
        return ("continue",)

    def visit_call(self, node, visited_children):
        name, _, _, args_opt, _, _ = visited_children
        arglist = []
        inner = _optional(args_opt)
        if inner is not None:
            first, rest = inner
            arglist.append(first)
            for group in rest:
                arglist.append(group[-1])
        return ("call", name, arglist)

    def visit_return(self, node, visited_children):
        _, expr_opt, _ = visited_children
        group = _optional(expr_opt)
        expr_val = group[-1] if group is not None else None
        return ("return", expr_val)

    def visit_expression(self, node, visited_children):
        left, cop_opt = visited_children
        group = _optional(cop_opt)
        if group is not None:
            _, op, _, right = group
            return ("binary", left, op, right)
        return left

    def visit_arith(self, node, visited_children):
        left, rest = visited_children
        for group in rest:
            left = ("binary", left, group[1], group[3])
        return left

    def visit_operand(self, node, visited_children):
        val = _flatten_choice(visited_children)
        if isinstance(val, str):
            return ("var", val)
        return val

    def visit_bop(self, node, visited_children):
        return node.text

    def visit_cop(self, node, visited_children):
        return node.text

    def visit_boolean(self, node, visited_children):
        return _flatten_choice(visited_children)

    def visit_true(self, node, visited_children):
        return ("bool", True)

    def visit_false(self, node, visited_children):
        return ("bool", False)

    def visit_string(self, node, visited_children):
        return ("string", node.text[1:-1])

    def visit_variable(self, node, visited_children):
        return ("var", node.text)

    def visit_number(self, node, visited_children):
        return ("int", int(node.text.replace("_", "")))

    def visit_name(self, node, visited_children):
        return node.text

    def generic_visit(self, node, visited_children):
        return visited_children or node

def interpret(source):
    """Parse source and return the tuple AST."""
    return LBVisitor().visit(parse(source))


if __name__ == "__main__":
    sys.exit(main())
