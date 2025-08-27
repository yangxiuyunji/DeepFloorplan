from collections import deque

class UndoCommand:
    def __init__(self, redo_fn, undo_fn, desc=""):
        self.redo_fn = redo_fn
        self.undo_fn = undo_fn
        self.desc = desc
    def undo(self):
        self.undo_fn()
    def redo(self):
        self.redo_fn()

class UndoStack:
    def __init__(self, limit=100):
        self.limit = limit
        self._stack = deque()
        self._redo = deque()
    def push(self, cmd: UndoCommand):
        if len(self._stack) >= self.limit:
            self._stack.popleft()
        self._stack.append(cmd)
        self._redo.clear()
    def undo(self):
        if not self._stack:
            return
        cmd = self._stack.pop()
        cmd.undo()
        self._redo.append(cmd)
    def redo(self):
        if not self._redo:
            return
        cmd = self._redo.pop()
        cmd.redo()
        self._stack.append(cmd)
    def clear(self):
        self._stack.clear(); self._redo.clear()
