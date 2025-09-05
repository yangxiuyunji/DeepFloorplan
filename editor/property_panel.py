from PySide6.QtWidgets import QWidget, QDockWidget, QVBoxLayout, QLabel, QLineEdit, QComboBox, QPushButton, QFormLayout, QSpinBox
from PySide6.QtCore import Signal, QObject

class PropertySignals(QObject):
    roomFieldEdited = Signal(str, str, object)  # room_id, field, value
    deleteRoom = Signal(str)
    addRoom = Signal()
    saveRequest = Signal()
    toggleMask = Signal(bool)
    globalFieldEdited = Signal(str, object)  # field, value

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
        # 全局属性控件
        self.orientation_combo = QComboBox(); self.orientation_combo.addItems([
            "坐北朝南","坐南朝北","坐东朝西","坐西朝东",
            "坐东南朝西北","坐西北朝东南","坐西南朝东北","坐东北朝西南"
        ])
        self.north_angle_spin = QSpinBox(); self.north_angle_spin.setRange(0,359); self.north_angle_spin.setValue(0)
        self.magnetic_declination_spin = QSpinBox(); self.magnetic_declination_spin.setRange(-180,180); self.magnetic_declination_spin.setValue(0)
        form.addRow("类别", self.type_combo)
        form.addRow("原始文本", self.text_edit)
        form.addRow("BBox", self.bbox_lbl)
        form.addRow("房屋朝向", self.orientation_combo)
        form.addRow("北向角度", self.north_angle_spin)
        form.addRow("磁偏角", self.magnetic_declination_spin)
        layout.addLayout(form)
        
        # 新增模式说明
        self.mode_label = QLabel("模式: 未选中任何房间")
        self.mode_label.setStyleSheet("color: blue; font-weight: bold;")
        layout.addWidget(self.mode_label)
        
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
        self.orientation_combo.currentTextChanged.connect(lambda txt: self.signals.globalFieldEdited.emit("house_orientation", txt))
        self.north_angle_spin.valueChanged.connect(lambda v: self.signals.globalFieldEdited.emit("north_angle", v))
        self.magnetic_declination_spin.valueChanged.connect(lambda v: self.signals.globalFieldEdited.emit("magnetic_declination", v))

    def load_categories(self, cats):
        # room.type现在已经是基础类型，直接使用即可
        self.type_combo.clear()
        self.type_combo.addItems(sorted(cats))
    
    def _extract_base_type(self, room_type: str) -> str:
        """提取房间的基础类型，去除数字编号"""
        if not room_type:
            return room_type
        
        # 从后往前找，移除末尾的连续数字
        i = len(room_type) - 1
        while i >= 0 and room_type[i].isdigit():
            i -= 1
        
        # 如果整个字符串都是数字，返回原字符串
        if i < 0:
            return room_type
        
        # 返回去除数字后的基础类型
        return room_type[:i+1]

    def show_room(self, room):
        self.current_room = room
        if not room:
            self.info_lbl.setText("未选中")
            self.mode_label.setText("模式: 新增房间模式 - 选择类型后点击'新增房间'")
            self.mode_label.setStyleSheet("color: green; font-weight: bold;")
            # 在新增房间模式下，不清空用户的类型选择！
            # 只清空其他字段
            self.text_edit.clear()  # 清空文本
            self.bbox_lbl.setText("")  # 清空边界框信息
            return
        
        self.info_lbl.setText(f"ID: {room.id}")
        self.mode_label.setText(f"模式: 编辑房间 - 修改属性会影响当前房间")
        self.mode_label.setStyleSheet("color: orange; font-weight: bold;")
        
        # room.type现在已经是基础类型，直接使用
        if self.type_combo.findText(room.type) < 0:
            self.type_combo.addItem(room.type)
        self.type_combo.setCurrentText(room.type)
        self.text_edit.setText(room.text_raw)
        x1,y1,x2,y2 = room.bbox
        self.bbox_lbl.setText(f"({x1},{y1})-({x2},{y2})")

    def refresh(self, room):
        if self.current_room and room and self.current_room.id == room.id:
            self.show_room(room)

    def set_global_fields(self, orientation: str, north_angle: int, magnetic_declination: int = 0):
        # 更新全局字段显示（不触发信号）
        idx = self.orientation_combo.findText(orientation)
        if idx >= 0:
            self.orientation_combo.blockSignals(True)
            self.orientation_combo.setCurrentIndex(idx)
            self.orientation_combo.blockSignals(False)
        else:
            self.orientation_combo.blockSignals(True)
            self.orientation_combo.addItem(orientation)
            self.orientation_combo.setCurrentText(orientation)
            self.orientation_combo.blockSignals(False)
        self.north_angle_spin.blockSignals(True)
        self.north_angle_spin.setValue(int(north_angle))
        self.north_angle_spin.blockSignals(False)
        self.magnetic_declination_spin.blockSignals(True)
        self.magnetic_declination_spin.setValue(int(magnetic_declination))
        self.magnetic_declination_spin.blockSignals(False)

    def _on_type_change(self):
        if self.current_room:
            self.signals.roomFieldEdited.emit(self.current_room.id, "type", self.type_combo.currentText())

    def _on_text_change(self):
        if self.current_room:
            self.signals.roomFieldEdited.emit(self.current_room.id, "text_raw", self.text_edit.text())

    def _on_delete(self):
        if self.current_room:
            self.signals.deleteRoom.emit(self.current_room.id)
