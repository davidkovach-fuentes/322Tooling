from .bytecode import OpCode, CompiledFunction
from .interpreter import LanguageError, NDArray, Tuple

def _wrap_int64(val: int) -> int:
    """Wraps an integer to signed 64-bit range [-2^63, 2^63 - 1]."""
    val = val & 0xFFFFFFFFFFFFFFFF
    if val >= 0x8000000000000000:
        val -= 0x10000000000000000
    return val

class CallFrame:
    def __init__(self, function: CompiledFunction):
        self.function = function
        self.ip = 0
        self.locals = [0] * function.max_slots
        self.expr_stack = []

class VM:
    def __init__(self, functions: dict = None):
        """functions: dict[str, CompiledFunction] for every user-defined LB
        function, as produced by compiler.compile_program(). Needed so CALL
        can look up a callee by name at runtime."""
        self.functions = functions or {}

    def call_function(self, name: str, args: list):
        if name == "print":
            for a in args:
                if isinstance(a, (NDArray, Tuple)):
                    print(repr(a))
                elif isinstance(a, bool):
                    print("1" if a else "0")
                else:
                    print(_wrap_int64(a))
            return None
        if name == "input":
            return _wrap_int64(int(input().strip()))
        if name not in self.functions:
            raise LanguageError(0, 0, f"Undefined function {name!r}")
        return self.run(self.functions[name], args)

    def run(self, function: CompiledFunction, args: list = None):
        frame = CallFrame(function)
        for i, val in enumerate(args or []):
            frame.locals[i] = val
        code = function.chunk.code
        constants = function.chunk.constants

        while frame.ip < len(code):
            opcode, operand = code[frame.ip]
            frame.ip += 1

            if opcode == OpCode.CONST:
                frame.expr_stack.append(constants[operand])

            elif opcode == OpCode.GET_LOCAL:
                frame.expr_stack.append(frame.locals[operand])

            elif opcode == OpCode.SET_LOCAL:
                val = frame.expr_stack.pop()
                frame.locals[operand] = val

            elif opcode == OpCode.ADD:
                b = frame.expr_stack.pop()
                a = frame.expr_stack.pop()
                frame.expr_stack.append(_wrap_int64(a + b))

            elif opcode == OpCode.SUB:
                b = frame.expr_stack.pop()
                a = frame.expr_stack.pop()
                frame.expr_stack.append(_wrap_int64(a - b))

            elif opcode == OpCode.MUL:
                b = frame.expr_stack.pop()
                a = frame.expr_stack.pop()
                frame.expr_stack.append(_wrap_int64(a * b))

            elif opcode == OpCode.DIV:
                b = frame.expr_stack.pop()
                a = frame.expr_stack.pop()
                if b == 0:
                    raise LanguageError(0, 0, "Division by zero")
                frame.expr_stack.append(_wrap_int64(a // b))

            elif opcode == OpCode.MOD:
                b = frame.expr_stack.pop()
                a = frame.expr_stack.pop()
                if b == 0:
                    raise LanguageError(0, 0, "Modulo by zero")
                frame.expr_stack.append(_wrap_int64(a % b))

            elif opcode == OpCode.BIT_AND:
                b = frame.expr_stack.pop()
                a = frame.expr_stack.pop()
                frame.expr_stack.append(_wrap_int64(a & b))

            elif opcode == OpCode.BIT_OR:
                b = frame.expr_stack.pop()
                a = frame.expr_stack.pop()
                frame.expr_stack.append(_wrap_int64(a | b))

            elif opcode == OpCode.BIT_XOR:
                b = frame.expr_stack.pop()
                a = frame.expr_stack.pop()
                frame.expr_stack.append(_wrap_int64(a ^ b))

            elif opcode == OpCode.BIT_SHL:
                b = frame.expr_stack.pop()
                a = frame.expr_stack.pop()
                frame.expr_stack.append(_wrap_int64(a << (b & 63)))

            elif opcode == OpCode.BIT_SHR:
                b = frame.expr_stack.pop()
                a = frame.expr_stack.pop()
                frame.expr_stack.append(_wrap_int64(a >> (b & 63)))

            elif opcode == OpCode.JUMP:
                frame.ip = operand

            elif opcode == OpCode.JUMP_IF_FALSE:
                cond = frame.expr_stack.pop()
                if not cond:
                    frame.ip = operand

            elif opcode == OpCode.EQ:
                b = frame.expr_stack.pop()
                a = frame.expr_stack.pop()
                frame.expr_stack.append(int(a == b))

            elif opcode == OpCode.NE:
                b = frame.expr_stack.pop()
                a = frame.expr_stack.pop()
                frame.expr_stack.append(int(a != b))

            elif opcode == OpCode.LT:
                b = frame.expr_stack.pop()
                a = frame.expr_stack.pop()
                frame.expr_stack.append(int(a < b))

            elif opcode == OpCode.LE:
                b = frame.expr_stack.pop()
                a = frame.expr_stack.pop()
                frame.expr_stack.append(int(a <= b))

            elif opcode == OpCode.GT:
                b = frame.expr_stack.pop()
                a = frame.expr_stack.pop()
                frame.expr_stack.append(int(a > b))

            elif opcode == OpCode.GE:
                b = frame.expr_stack.pop()
                a = frame.expr_stack.pop()
                frame.expr_stack.append(int(a >= b))

            elif opcode == OpCode.POP:
                frame.expr_stack.pop()

            elif opcode == OpCode.ARR_NEW:
                dims = [frame.expr_stack.pop() for _ in range(operand)]
                dims.reverse()
                frame.expr_stack.append(NDArray(dims))

            elif opcode == OpCode.TUP_NEW:
                dims = [frame.expr_stack.pop() for _ in range(operand)]
                dims.reverse()
                frame.expr_stack.append(Tuple(dims))

            elif opcode == OpCode.ARR_GET:
                indices = [frame.expr_stack.pop() for _ in range(operand)]
                indices.reverse()
                arr = frame.expr_stack.pop()
                frame.expr_stack.append(arr.get(*indices))

            elif opcode == OpCode.ARR_SET:
                value = frame.expr_stack.pop()
                indices = [frame.expr_stack.pop() for _ in range(operand)]
                indices.reverse()
                arr = frame.expr_stack.pop()
                arr.set(*indices, value)

            elif opcode == OpCode.LEN:
                dim = frame.expr_stack.pop()
                arr = frame.expr_stack.pop()
                frame.expr_stack.append(arr.length(dim))

            elif opcode == OpCode.CALL:
                name, argc = operand
                call_args = [frame.expr_stack.pop() for _ in range(argc)]
                call_args.reverse()
                frame.expr_stack.append(self.call_function(name, call_args))

            elif opcode == OpCode.CALL_INDIRECT:
                call_args = [frame.expr_stack.pop() for _ in range(operand)]
                call_args.reverse()
                name = frame.expr_stack.pop()
                frame.expr_stack.append(self.call_function(name, call_args))

            elif opcode == OpCode.RETURN:
                return frame.expr_stack.pop() if frame.expr_stack else None

        return None
