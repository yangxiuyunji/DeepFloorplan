from PySide6.QtWidgets import QWidget, QDockWidget, QVBoxLayout, QLabel, QLineEdit, QComboBox, QPushButton, QFormLayout
from PySide6.QtCore import Signal, QObject

class PropertySignals(QObject):
    roomFieldEdited = Signal(str, str, object)  # room_id, field, value
    deleteRoom = Signal(str)
    addRoom = Signal()
    saveRequest = Signal()
    toggleMask = Signal(bool)

class PropertyPanel:
    def __init__(self):
        self.dock = QDockWidget("属性", None)
        self.widget = QWidget()
        self.dock.setWidget(self.widget)
        self.signals = PropertySignals()
        layout = QVBoxLayout(self.widget)
        self.info_lbl = QLabel("未选中")
        layout.addWidget(self.info_lbl)
        form = QFormLayout()
        self.type_combo = QComboBox(); self.type_combo.setEditable(True)
        self.text_edit = QLineEdit()
        self.bbox_lbl = QLabel("-")
        form.addRow("类别", self.type_combo)
        form.addRow("原始文本", self.text_edit)
        form.addRow("BBox", self.bbox_lbl)
        layout.addLayout(form)
        self.btn_add = QPushButton("新增房间")
        self.btn_del = QPushButton("删除房间")
        self.btn_mask = QPushButton("切换Mask")
        self.btn_save = QPushButton("保存")
        layout.addWidget(self.btn_add)
        layout.addWidget(self.btn_del)
        layout.addWidget(self.btn_mask)
        layout.addWidget(self.btn_save)
        layout.addStretch()
        self.current_room = None
        # 事件
        self.type_combo.currentTextChanged.connect(self._on_type_change)
        self.text_edit.editingFinished.connect(self._on_text_change)
        self.btn_del.clicked.connect(self._on_delete)
        self.btn_add.clicked.connect(lambda: self.signals.addRoom.emit())
        self.btn_save.clicked.connect(lambda: self.signals.saveRequest.emit())
        self.btn_mask.clicked.connect(lambda: self.signals.toggleMask.emit(True))

    def load_categories(self, cats):
        self.type_combo.clear(); self.type_combo.addItems(sorted(cats))

    def show_room(self, room):
        self.current_room = room
        if not room:
            self.info_lbl.setText("未选中")
            return
        self.info_lbl.setText(f"ID: {room.id}")
        if self.type_combo.findText(room.type) < 0:
            self.type_combo.addItem(room.type)
        self.type_combo.setCurrentText(room.type)
        self.text_edit.setText(room.text_raw)
        x1,y1,x2,y2 = room.bbox
        self.bbox_lbl.setText(f"({x1},{y1})-({x2},{y2})")

    def refresh(self, room):
        if self.current_room and room and self.current_room.id == room.id:
            self.show_room(room)

    def _on_type_change(self):
        if self.current_room:
            self.signals.roomFieldEdited.emit(self.current_room.id, "type", self.type_combo.currentText())

    def _on_text_change(self):
        if self.current_room:
            self.signals.roomFieldEdited.emit(self.current_room.id, "text_raw", self.text_edit.text())

    def _on_delete(self):
        if self.current_room:
            self.signals.deleteRoom.emit(self.current_room.id)
