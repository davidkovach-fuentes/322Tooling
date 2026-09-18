from enum import IntEnum, auto

class OpCode(IntEnum):
    CONST = 0
    POP = auto()
    
    GET_LOCAL = auto()
    SET_LOCAL = auto()
    
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()
    
    BIT_AND = auto()
    BIT_OR = auto()
    BIT_XOR = auto()
    BIT_SHL = auto()
    BIT_SHR = auto()
    
    EQ = auto()
    NE = auto()
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()
    
    ARR_NEW = auto()
    TUP_NEW = auto()
    ARR_GET = auto()
    ARR_SET = auto()
    LEN = auto()
    
    JUMP = auto()
    JUMP_IF_FALSE = auto()
    
    CALL = auto()
    CALL_INDIRECT = auto()
    RETURN = auto()

class Chunk:
    def __init__(self):
        self.code = []
        self.constants = []

    def write(self, opcode: OpCode, operand=None) -> int:
        self.code.append((opcode, operand))
        return len(self.code) - 1

    def patch_operand(self, index: int, new_operand):
        """Updates the operand of an existing instruction (e.g., jump backpatching)."""
        opcode, _ = self.code[index]
        self.code[index] = (opcode, new_operand)

    def add_constant(self, value) -> int:
        """Adds a value to the constant pool, deduplicating if it already exists."""
        if value in self.constants:
            return self.constants.index(value)
        self.constants.append(value)
        return len(self.constants) - 1

class CompiledFunction:
    """Wraps a Chunk with the metadata required to construct a CallFrame."""
    def __init__(self, name: str, num_params: int, max_slots: int, chunk: Chunk):
        self.name = name
        self.num_params = num_params
        self.max_slots = max_slots
        self.chunk = chunk
