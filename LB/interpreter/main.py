from .interpreter import NDArray, Tuple


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
        self.while_specs = []
        self.break_positions = []
        self.continue_positions = []
        self._flatten_scope(body)
        self._backfill()
        self._resolve_loops()
        return {
            "type": typ,
            "name": name,
            "params": params,
            "instrs": self.instrs,
            "labels": self.label_table,
            "depths": self._compute_depths(),
        }

    def _compute_depths(self):
        """Structural scope depth expected *before* each instruction runs,
        found by bracket-matching scope_enter (+1) / scope_exit (-1) over
        the flat instruction list. Used so a jump that lands mid-scope (a
        while loop's body label sits right after that scope's scope_enter)
        can reconcile the live scope stack to the right depth instead of
        relying on scope_enter/scope_exit always running in a perfect 1:1
        sequence — which backward jumps into the middle of a scope break."""
        depths = [0] * len(self.instrs)
        depth = 0
        for i, instr in enumerate(self.instrs):
            depths[i] = depth
            if instr[0] == "scope_enter":
                depth += 1
            elif instr[0] == "scope_exit":
                depth -= 1
        return depths

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

        elif kind == "if":
            _, cond, tlabel, flabel = stmt
            idx = self._emit(["if", cond, None, None])
            self.patches.append((idx, 2, _label_name(tlabel)))
            self.patches.append((idx, 3, _label_name(flabel)))

        elif kind == "while":
            _, cond, blabel, elabel = stmt
            idx = self._emit(["while", cond, None, None])
            blabel_name = _label_name(blabel)
            elabel_name = _label_name(elabel)
            self.patches.append((idx, 2, blabel_name))
            self.patches.append((idx, 3, elabel_name))
            self.while_specs.append((idx, blabel_name, elabel_name))

        elif kind == "goto":
            idx = self._emit(["goto", None])
            self.patches.append((idx, 1, _label_name(stmt[1])))

        elif kind == "break":
            idx = self._emit(["break", None])
            self.break_positions.append(idx)

        elif kind == "continue":
            idx = self._emit(["continue", None])
            self.continue_positions.append(idx)

        else:
            self._emit(stmt)

    def _backfill(self):
        for idx, field, label_name in self.patches:
            if label_name not in self.label_table:
                raise FlattenError(0, f"undefined label {label_name!r}")
            instr = list(self.instrs[idx])
            instr[field] = self.label_table[label_name]
            self.instrs[idx] = tuple(instr)

    def _resolve_loops(self):

        begin_of = {}
        end_of = {}
        for loop_id, (_, blabel_name, elabel_name) in enumerate(self.while_specs):
            begin_of[self.label_table[blabel_name]] = loop_id
            end_of[self.label_table[elabel_name]] = loop_id

        loop_stack = []
        instr_loop = [None] * len(self.instrs)
        for i in range(len(self.instrs)):
            if i in end_of and loop_stack and loop_stack[-1] == end_of[i]:
                loop_stack.pop()
            instr_loop[i] = loop_stack[-1] if loop_stack else None
            if i in begin_of:
                loop_stack.append(begin_of[i])

        for idx in self.break_positions:
            loop_id = instr_loop[idx]
            if loop_id is None:
                raise FlattenError(0, "break outside loop")
            _, _, elabel_name = self.while_specs[loop_id]
            self.instrs[idx] = ("break", self.label_table[elabel_name])

        for idx in self.continue_positions:
            loop_id = instr_loop[idx]
            if loop_id is None:
                raise FlattenError(0, "continue outside loop")
            while_idx, _, _ = self.while_specs[loop_id]
            self.instrs[idx] = ("continue", while_idx)


class Interpreter:
    def __init__(self, program_ast):
        flattener = Flattener()
        self.functions = {
            f["name"]: f
            for f in (flattener.flatten_function(fn) for fn in program_ast[1])
        }

    def run(self, entry="main", args=None):
        return self.call(entry, args or [])

    def call(self, name, args):
        if name == "print":
            print(*args)
            return None
        if name == "input":
            return int(input())
        func = self.functions[name]
        env = {}
        for param, arg in zip(func["params"], args):
            env[param[2]] = arg
        return self._exec(func, env)

    def _exec(self, func, env):
        instrs = func["instrs"]
        pc = 0
        n = len(instrs)
        _eval = self._eval
        while pc < n:
            instr = instrs[pc]
            kind = instr[0]

            if kind == "assign":
                env[instr[1]] = _eval(instr[2], env)
            elif kind == "decl":
                env[instr[2]] = 0
            elif kind == "array_read":
                _, var, arr_name, indices = instr
                env[var] = env[arr_name].get(*[_eval(e, env) for e in indices])
            elif kind == "array_write":
                _, arr_name, indices, expr = instr
                env[arr_name].set(
                    *([_eval(e, env) for e in indices] + [_eval(expr, env)])
                )
            elif kind == "if":
                pc = instr[2] if _eval(instr[1], env) else instr[3]
                continue
            elif kind == "while":
                pc = instr[2] if _eval(instr[1], env) else instr[3]
                continue
            elif kind in ("goto", "break", "continue"):
                pc = instr[1]
                continue
            elif kind == "new_array":
                _, var, dim_exprs = instr
                env[var] = NDArray([_eval(e, env) for e in dim_exprs])
            elif kind == "new_tuple":
                _, var, dim_exprs = instr
                env[var] = Tuple([_eval(e, env) for e in dim_exprs])
            elif kind == "length":
                _, var, arr_name, dim_expr = instr
                dim = _eval(dim_expr, env) if dim_expr is not None else 0
                env[var] = env[arr_name].length(dim)
            elif kind == "return":
                return _eval(instr[1], env) if instr[1] is not None else None
            elif kind == "call":
                _eval(instr, env)
            elif kind in ("scope_enter", "scope_exit", "label"):
                pass
            else:
                raise ValueError(f"unknown instruction {instr!r}")

            pc += 1
        return None

    def _eval(self, expr, env):
        kind = expr[0]
        if kind == "int":
            return expr[1]
        if kind == "string":
            return expr[1]
        if kind == "var":
            name = expr[1]
            if name in env:
                return env[name]
            if name in self.functions:
                return name
            return env[name]
        if kind == "call":
            _, fname, args = expr
            target = env[fname] if fname in env else fname
            return self.call(target, [self._eval(a, env) for a in args])
        if kind == "binary":
            _, left, op, right = expr
            l = self._eval(left, env)
            r = self._eval(right, env)
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
