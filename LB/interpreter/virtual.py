from ops import *

class VM:
    def __init__(self):
        self.stack = []
        self.frames = []
        self.globals = {}
        self.objects = None  # Head of linked list of all allocated Obj instances
        self.temp_roots = []  # Safeguard for in-flight allocations
        self.bytes_allocated = 0
        self.next_gc = 1024 * 1024  # Trigger threshold (1 MB)

    def push_root(self, obj: Obj):
        """Root an object temporarily during nested allocation routines."""
        self.temp_roots.append(obj)

    def pop_root(self):
        self.temp_roots.pop()

    def collect_garbage(self):
        self._mark_roots()
        self._trace_references()
        self._sweep()
        self.next_gc = max(1024 * 1024, self.bytes_allocated * 2)

    def _mark_roots(self):
        # 1. Mark objects on the evaluation stack
        for val in self.stack:
            if isinstance(val, Obj):
                self._mark_object(val)

        # 2. Mark temporary allocation roots (prevents the allocation trap)
        for obj in self.temp_roots:
            self._mark_object(obj)

        # 3. Mark globals
        for val in self.globals.values():
            if isinstance(val, Obj):
                self._mark_object(val)

    def _mark_object(self, obj: Obj):
        if obj is None or obj.is_marked:
            return
        obj.is_marked = True
        # If the object contains references to other Objs (e.g., nested arrays), mark them
        if isinstance(obj, ObjArray):
            for elem in obj.elements:
                if isinstance(elem, Obj):
                    self._mark_object(elem)

    def _trace_references(self):
        # Additional gray-stack tracing if handling complex object graphs
        pass

    def _sweep(self):
        previous = None
        current = self.objects
        while current is not None:
            if current.is_marked:
                current.is_marked = False  # Reset for next GC cycle
                previous = current
                current = current.next_obj
            else:
                # Unreachable object: unhook from list and reclaim
                unreached = current
                current = current.next_obj
                if previous is not None:
                    previous.next_obj = current
                else:
                    self.objects = current
                # Explicitly break references to aid Python's host collection
                unreached.elements = []
