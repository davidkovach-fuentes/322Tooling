from typing import Dict, List, Tuple, Any
from .bytecode import OpCode, Chunk, CompiledFunction

class CompileError(Exception):
    pass

def _label_name(node: Any) -> str:
    """Unwrap ('label', name) tuples produced by LBi.py into bare string names."""
    if isinstance(node, tuple) and len(node) >= 2 and node[0] == "label":
        return node[1]
    return node

class FunctionCompiler:
    def __init__(self, name: str, num_params: int):
        self.name = name
        self.num_params = num_params
        self.chunk = Chunk()
        
        self.locals: List[Tuple[str, int, int]] = []
        self.next_slot = num_params
        self.max_slots = num_params
        self.scope_depth = 0
        self.floor_stack: List[int] = []

        self.label_table: Dict[str, int] = {}
        self.patches: List[Tuple[int, str]] = []
        self.loops: List[Tuple[str, str]] = []
        self.pending_loop_jumps: List[Tuple[int, str]] = []
        self.while_instr_of: Dict[str, int] = {}

    def compile_statement(self, node: tuple):
        if not isinstance(node, tuple) or not node:
            raise CompileError(f"Invalid AST statement node: {node}")
            
        tag = node[0]
        visitor = getattr(self, f"visit_stmt_{tag}", None)
        if not visitor:
            raise CompileError(f"No statement visitor for AST tag: '{tag}'")
        
        visitor(*node[1:])

    def compile_expression(self, node: tuple):
        if not isinstance(node, tuple) or not node:
            raise CompileError(f"Invalid AST expression node: {node}")
            
        tag = node[0]
        visitor = getattr(self, f"visit_expr_{tag}", None)
        if not visitor:
            raise CompileError(f"No expression visitor for AST tag: '{tag}'")
        
        visitor(*node[1:])

    def resolve_local(self, name: str) -> int:
        for local_name, slot, depth in reversed(self.locals):
            if local_name == name:
                return slot
        raise CompileError(f"Undeclared variable '{name}'. Assignments must resolve to a known compile-time slot.")

    def resolve_local_or_global(self, name: str):
        for local_name, slot, depth in reversed(self.locals):
            if local_name == name:
                return True, slot
        return False, name


    def visit_stmt_decl(self, typ: str, name: str):
        slot = self.next_slot
        self.next_slot += 1
        self.max_slots = max(self.max_slots, self.next_slot)
        
        self.locals.append((name, slot, self.scope_depth))
        
        idx = self.chunk.add_constant(0)
        self.chunk.write(OpCode.CONST, idx)
        self.chunk.write(OpCode.SET_LOCAL, slot)

    def visit_stmt_assign(self, name: str, expr_node: tuple):
        self.compile_expression(expr_node)
        slot = self.resolve_local(name)
        self.chunk.write(OpCode.SET_LOCAL, slot)

    def visit_stmt_label(self, name: Any):
        label_str = _label_name(name)
        self.label_table[label_str] = len(self.chunk.code)

    def visit_stmt_goto(self, target: Any):
        target_str = _label_name(target)
        idx = self.chunk.write(OpCode.JUMP, None)
        self.patches.append((idx, target_str))

    def visit_stmt_if(self, cond: tuple, tlabel: Any, flabel: Any):
        t_str = _label_name(tlabel)
        f_str = _label_name(flabel)
        self.compile_expression(cond)
        
        idx_f = self.chunk.write(OpCode.JUMP_IF_FALSE, None)
        self.patches.append((idx_f, f_str))
        
        idx_t = self.chunk.write(OpCode.JUMP, None)
        self.patches.append((idx_t, t_str))

    def visit_stmt_while(self, cond: tuple, blabel: Any, elabel: Any):
        b_str = _label_name(blabel)
        e_str = _label_name(elabel)
        self.loops.append((b_str, e_str))
        self.while_instr_of[b_str] = len(self.chunk.code)

        self.compile_expression(cond)

        idx_e = self.chunk.write(OpCode.JUMP_IF_FALSE, None)
        self.patches.append((idx_e, e_str))

        idx_b = self.chunk.write(OpCode.JUMP, None)
        self.patches.append((idx_b, b_str))

    def visit_stmt_continue(self):
        idx = self.chunk.write(OpCode.JUMP, None)
        self.pending_loop_jumps.append((idx, "continue"))

    def visit_stmt_break(self):
        idx = self.chunk.write(OpCode.JUMP, None)
        self.pending_loop_jumps.append((idx, "break"))

    def visit_stmt_scope(self, stmts: list):
        floor = self.next_slot
        self.floor_stack.append(floor)
        self.scope_depth += 1
        
        for stmt in stmts:
            self.compile_statement(stmt)
            
        self.scope_depth -= 1
        self.locals = [l for l in self.locals if l[2] <= self.scope_depth]
        self.next_slot = self.floor_stack.pop()

    def visit_stmt_return(self, expr_opt=None):
        if expr_opt is not None:
            self.compile_expression(expr_opt)
        else:
            idx = self.chunk.add_constant(None)
            self.chunk.write(OpCode.CONST, idx)
        self.chunk.write(OpCode.RETURN)

    def visit_stmt_call(self, name: str, args: list):
        self.compile_expression(("call", name, args))
        self.chunk.write(OpCode.POP)

    def visit_stmt_new_array(self, name: str, args: list):
        for arg in args:
            self.compile_expression(arg)
        self.chunk.write(OpCode.ARR_NEW, len(args))
        slot = self.resolve_local(name)
        self.chunk.write(OpCode.SET_LOCAL, slot)

    def visit_stmt_new_tuple(self, name: str, args: list):
        for arg in args:
            self.compile_expression(arg)
        self.chunk.write(OpCode.TUP_NEW, len(args))
        slot = self.resolve_local(name)
        self.chunk.write(OpCode.SET_LOCAL, slot)

    def visit_stmt_length(self, name: str, arr_name: str, dim_expr):
        arr_slot = self.resolve_local(arr_name)
        self.chunk.write(OpCode.GET_LOCAL, arr_slot)
        if dim_expr is not None:
            self.compile_expression(dim_expr)
        else:
            idx = self.chunk.add_constant(0)
            self.chunk.write(OpCode.CONST, idx)
        self.chunk.write(OpCode.LEN)
        slot = self.resolve_local(name)
        self.chunk.write(OpCode.SET_LOCAL, slot)

    def visit_stmt_array_read(self, name: str, arr_name: str, indices: list):
        arr_slot = self.resolve_local(arr_name)
        self.chunk.write(OpCode.GET_LOCAL, arr_slot)
        for idx_expr in indices:
            self.compile_expression(idx_expr)
        self.chunk.write(OpCode.ARR_GET, len(indices))
        slot = self.resolve_local(name)
        self.chunk.write(OpCode.SET_LOCAL, slot)

    def visit_stmt_array_write(self, arr_name: str, indices: list, expr_node: tuple):
        arr_slot = self.resolve_local(arr_name)
        self.chunk.write(OpCode.GET_LOCAL, arr_slot)
        for idx_expr in indices:
            self.compile_expression(idx_expr)
        self.compile_expression(expr_node)
        self.chunk.write(OpCode.ARR_SET, len(indices))

    def backfill(self):
        """Resolves nested loop jumps and control-flow labels to absolute bytecode offsets."""
        for idx, jump_type in self.pending_loop_jumps:
            matching_loops = []
            for blabel, elabel in self.loops:
                if blabel not in self.label_table or elabel not in self.label_table:
                    continue
                b_off = self.label_table[blabel]
                e_off = self.label_table[elabel]
                if b_off <= idx <= e_off:
                    matching_loops.append((b_off, e_off, blabel, elabel))

            if not matching_loops:
                raise CompileError(f"Cannot '{jump_type}' outside of a loop.")

            matching_loops.sort(key=lambda item: item[1] - item[0])
            _, _, blabel, elabel = matching_loops[0]
            target = self.while_instr_of[blabel] if jump_type == "continue" else elabel
            self.patches.append((idx, target))

        for instruction_idx, target_label in self.patches:
            if isinstance(target_label, int):
                offset = target_label
            else:
                if target_label not in self.label_table:
                    raise CompileError(f"Undefined label: '{target_label}'")
                offset = self.label_table[target_label]
            self.chunk.patch_operand(instruction_idx, offset)


    def visit_expr_int(self, value: int):
        idx = self.chunk.add_constant(value)
        self.chunk.write(OpCode.CONST, idx)

    def visit_expr_bool(self, value: bool):
        idx = self.chunk.add_constant(1 if value else 0)
        self.chunk.write(OpCode.CONST, idx)

    def visit_expr_var(self, name: str):
        is_local, val = self.resolve_local_or_global(name)
        if is_local:
            self.chunk.write(OpCode.GET_LOCAL, val)
        else:
            idx = self.chunk.add_constant(val)
            self.chunk.write(OpCode.CONST, idx)

    def visit_expr_string(self, value: str):
        idx = self.chunk.add_constant(value)
        self.chunk.write(OpCode.CONST, idx)

    def visit_expr_call(self, name, args: list):
        if isinstance(name, tuple):
            if name[0] != "var":
                raise CompileError(f"Unsupported call target: {name!r}")
            name = name[1]
        is_local, val = self.resolve_local_or_global(name)
        if is_local:
            self.chunk.write(OpCode.GET_LOCAL, val)
            for arg in args:
                self.compile_expression(arg)
            self.chunk.write(OpCode.CALL_INDIRECT, len(args))
            return
        for arg in args:
            self.compile_expression(arg)
        self.chunk.write(OpCode.CALL, (name, len(args)))

    def visit_expr_binary(self, left: tuple, op: str, right: tuple):
        self.compile_expression(left)
        self.compile_expression(right)
        
        op_map = {
            "+": OpCode.ADD,
            "-": OpCode.SUB,
            "*": OpCode.MUL,
            "/": OpCode.DIV,
            "%": OpCode.MOD,
            "==": OpCode.EQ,
            "=":  OpCode.EQ,
            "!=": OpCode.NE,
            "<":  OpCode.LT,
            "<=": OpCode.LE,
            ">":  OpCode.GT,
            ">=": OpCode.GE,
            "&":  OpCode.BIT_AND,
            "|":  OpCode.BIT_OR,
            "^":  OpCode.BIT_XOR,
            "<<": OpCode.BIT_SHL,
            ">>": OpCode.BIT_SHR,
        }
        
        if op not in op_map:
            raise CompileError(f"Unsupported binary operator: '{op}'")
        
        self.chunk.write(op_map[op])

def compile_program(program_ast: tuple) -> Dict[str, CompiledFunction]:
    if not isinstance(program_ast, tuple) or program_ast[0] != "program":
        raise CompileError("Invalid program AST root.")
        
    _, functions = program_ast
    compiled_functions = {}

    for fn_node in functions:
        _, typ, name, params, body = fn_node
        
        num_params = len(params)
        compiler = FunctionCompiler(name=name, num_params=num_params)
        
        for idx, param in enumerate(params):
            _, p_typ, p_name = param
            compiler.locals.append((p_name, idx, 0))

        compiler.compile_statement(body)
        compiler.backfill()
        
        idx = compiler.chunk.add_constant(None)
        compiler.chunk.write(OpCode.CONST, idx)
        compiler.chunk.write(OpCode.RETURN)
        
        compiled_functions[name] = CompiledFunction(
            name=compiler.name,
            num_params=compiler.num_params,
            max_slots=compiler.max_slots,
            chunk=compiler.chunk
        )

    return compiled_functions
