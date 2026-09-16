from enum import IntEnum, auto

class OpCode(IntEnum):
    OP_CONSTANT = auto()
    OP_ADD = auto()
    OP_SUB = auto()
    OP_MUL = auto()
    OP_SHL = auto()
    OP_SHR = auto()
    OP_JUMP_IF_FALSE = auto()
    OP_JUMP = auto()
    OP_GET_LOCAL = auto()
    OP_SET_LOCAL = auto()
    OP_NEW_ARRAY = auto()
    OP_GET_SUBSCRIPT = auto()
    OP_SET_SUBSCRIPT = auto()
    OP_CALL = auto()
    OP_RETURN = auto()

def wrap_int64(val: int) -> int:
    """Enforces standard 64-bit signed two's complement integer wrapping."""
    val = val & 0xFFFFFFFFFFFFFFFF
    return val if val < 0x8000000000000000 else val - 0x10000000000000000

class Obj:
    """Base heap object tracked by the VM GC."""
    def __init__(self, vm):
        self.is_marked = False
        # Link into the VM's global object list for sweeping
        self.next_obj = vm.objects
        vm.objects = self

class ObjArray(Obj):
    def __init__(self, vm, dimensions: list[int]):
        super().__init__(vm)
        self.dimensions = dimensions
        # Flat element storage or nested references
        total_size = 1
        for d in dimensions:
            total_size *= d
        self.elements = [0] * total_size
