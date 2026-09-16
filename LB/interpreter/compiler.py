from parsimonious.nodes import NodeVisitor

class BytecodeCompiler(NodeVisitor):
    def __init__(self):
        self.chunk = Chunk()
        self.locals = []  # Stack offset tracking for local variables
        self.scope_depth = 0

    def visit_number(self, node, visited_children):
        val = wrap_int64(int(node.text))
        idx = self.chunk.add_constant(val)
        self.chunk.emit_byte(OpCode.OP_CONSTANT)
        self.chunk.emit_byte(idx)

    def visit_binary_add(self, node, visited_children):
        # Visit left and right children (pushes operands onto VM stack)
        self.visit(node.children[0])
        self.visit(node.children[2])
        self.chunk.emit_byte(OpCode.OP_ADD)

    def visit_while_stmt(self, node, visited_children):
        loop_start = len(self.chunk.code)
        
        # 1. Condition expression
        self.visit(node.children[1])
        
        # 2. Emit jump if condition false (back-patched later)
        self.chunk.emit_byte(OpCode.OP_JUMP_IF_FALSE)
        jump_patch = len(self.chunk.code)
        self.chunk.emit_byte(0)  # Placeholder offset

        # 3. Body
        self.visit(node.children[3])
        
        # 4. Loop back to start
        self.chunk.emit_byte(OpCode.OP_JUMP)
        self.chunk.emit_byte(loop_start)

        # 5. Patch exit jump
        self.chunk.code[jump_patch] = len(self.chunk.code)
