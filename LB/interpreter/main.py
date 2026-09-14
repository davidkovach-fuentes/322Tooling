from .interpreter import Memory, ObjList, Tuple


def _label_name(node):
    return node[1] if isinstance(node, tuple) and node[0] == "label" else node


class FlattenError(Exception):
    def __init__(self, line, message):
        self.line = line
        self.message = message

    def __str__(self):
        return f"{self.line}: [flatten error] {self.message}"


class Flattener:
    def flatten_function(self, func_ast):
        _, typ, name, params, body = func_ast
        self.instrs = []
        self.label_table = {}
        self.patches = []
        self.loop_stack = []
        self._flatten_scope(body)
        self._backfill()
        return {
            "type": typ,
            "name": name,
            "params": params,
            "instrs": self.instrs,
            "labels": self.label_table,
        }

    def _emit(self, instr):
        self.instrs.append(instr)
        return len(self.instrs) - 1

    def _flatten_scope(self, scope_ast):
        _, stmts = scope_ast
        self._emit(("scope_enter",))
        for stmt in stmts:
            self._flatten_stmt(stmt)
        self._emit(("scope_exit",))

    def _flatten_stmt(self, stmt):
        kind = stmt[0]

        if kind == "scope":
            self._flatten_scope(stmt)

        elif kind == "label":
            name = _label_name(stmt)
            if name in self.label_table:
                raise FlattenError(0, f"duplicate label {name!r}")
            idx = self._emit(stmt)
            self.label_table[name] = idx
            if self.loop_stack and self.loop_stack[-1][1] == name:
                self.loop_stack.pop()

        elif kind == "if":
            _, cond, tlabel, flabel = stmt
            idx = self._emit(["if", cond, None, None])
            self.patches.append((idx, 2, _label_name(tlabel)))
            self.patches.append((idx, 3, _label_name(flabel)))

        elif kind == "while":
            _, cond, blabel, elabel = stmt
            idx = self._emit(["while", cond, None, None])
            exit_name = _label_name(elabel)
            self.patches.append((idx, 2, _label_name(blabel)))
            self.patches.append((idx, 3, exit_name))
            self.loop_stack.append((idx, exit_name))

        elif kind == "goto":
            idx = self._emit(["goto", None])
            self.patches.append((idx, 1, _label_name(stmt[1])))

        elif kind == "break":
            if not self.loop_stack:
                raise FlattenError(0, "break outside loop")
            _, exit_name = self.loop_stack[-1]
            idx = self._emit(["break", None])
            self.patches.append((idx, 1, exit_name))

        elif kind == "continue":
            if not self.loop_stack:
                raise FlattenError(0, "continue outside loop")
            while_idx, _ = self.loop_stack[-1]
            self._emit(("continue", while_idx))

        else:
            self._emit(stmt)

    def _backfill(self):
        for idx, field, label_name in self.patches:
            if label_name not in self.label_table:
                raise FlattenError(0, f"undefined label {label_name!r}")
            instr = list(self.instrs[idx])
            instr[field] = self.label_table[label_name]
            self.instrs[idx] = tuple(instr)


class Interpreter:
    def __init__(self, program_ast):
        flattener = Flattener()
        self.functions = {
            f["name"]: f
            for f in (flattener.flatten_function(fn) for fn in program_ast[1])
        }
        self.memory = Memory()

    def run(self, entry="main", args=None):
        return self.call(entry, args or [])

    def call(self, name, args):
        if name == "print":
            print(*args)
            return None
        func = self.functions[name]
        self.memory.new_frame(name)
        for param, arg in zip(func["params"], args):
            self.memory.declare(param[2], arg)
        result = self._exec(func)
        self.memory.del_frame()
        return result

    def _exec(self, func):
        instrs = func["instrs"]
        pc = 0
        while pc < len(instrs):
            instr = instrs[pc]
            kind = instr[0]

            if kind in ("scope_enter",):
                self.memory.new_scope()
            elif kind == "scope_exit":
                self.memory.del_scope()
            elif kind == "label":
                pass
            elif kind == "decl":
                self.memory.declare(instr[2], 0)
            elif kind == "assign":
                self.memory[instr[1]] = self._eval(instr[2])
            elif kind == "new_array":
                _, var, dim_exprs = instr
                dims = [self._eval(e) for e in dim_exprs]
                self.memory[var] = ObjList(dims)
            elif kind == "new_tuple":
                _, var, dim_exprs = instr
                dims = [self._eval(e) for e in dim_exprs]
                self.memory[var] = Tuple(dims)
            elif kind == "array_read":
                _, var, arr_name, indices = instr
                arr = self.memory[arr_name]
                for idx_expr in indices:
                    arr = arr.get(self._eval(idx_expr))
                self.memory[var] = arr
            elif kind == "array_write":
                _, arr_name, indices, expr = instr
                arr = self.memory[arr_name]
                for idx_expr in indices[:-1]:
                    arr = arr.get(self._eval(idx_expr))
                arr.set(self._eval(indices[-1]), self._eval(expr))
            elif kind == "length":
                _, var, arr_name, dim_expr = instr
                dim = self._eval(dim_expr) if dim_expr is not None else 0
                self.memory[var] = self.memory[arr_name].length(dim)
            elif kind == "if":
                pc = instr[2] if self._eval(instr[1]) else instr[3]
                continue
            elif kind == "while":
                pc = instr[2] if self._eval(instr[1]) else instr[3]
                continue
            elif kind in ("goto", "break"):
                pc = instr[1]
                continue
            elif kind == "continue":
                pc = instr[1]
                continue
            elif kind == "return":
                return self._eval(instr[1]) if instr[1] is not None else None
            elif kind == "call":
                self._eval(instr)
            else:
                raise ValueError(f"unknown instruction {instr!r}")

            pc += 1
        return None

    def _eval(self, expr):
        kind = expr[0]
        if kind == "int":
            return expr[1]
        if kind == "var":
            return self.memory[expr[1]]
        if kind == "call":
            _, fname, args = expr
            return self.call(fname, [self._eval(a) for a in args])
        if kind == "binary":
            _, left, op, right = expr
            l = self._eval(left)
            r = self._eval(right)
            if op == "+":
                return l + r
            if op == "-":
                return l - r
            if op == "*":
                return l * r
            if op == "/":
                return l // r
            if op == "%":
                return l % r
            if op == "&":
                return l & r
            if op == "|":
                return l | r
            if op == "^":
                return l ^ r
            if op == "<<":
                return l << r
            if op == ">>":
                return l >> r
            if op == "==":
                return l == r
            if op == "=":
                return l == r
            if op == "!=":
                return l != r
            if op == "<":
                return l < r
            if op == "<=":
                return l <= r
            if op == ">":
                return l > r
            if op == ">=":
                return l >= r
            raise ValueError("unknown operator {!r}".format(op))
        if kind == "bool":
            return expr[1]
        raise ValueError(f"unknown expression {expr!r}")
