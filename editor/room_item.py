from PySide6.QtWidgets import QGraphicsRectItem, QGraphicsItem, QGraphicsSimpleTextItem
from PySide6.QtGui import QPen, QBrush, QColor
from PySide6.QtCore import Qt, QRectF, QPointF, Signal, QObject

HANDLE_SIZE = 12
GRID_SIZE = 8  # 网格尺寸（像素）
MIN_W = 12     # 最小宽度
MIN_H = 12     # 最小高度

class RoomGraphicsSignals(QObject):
    bboxChanged = Signal(str, tuple)   # room_id, bbox
    selected = Signal(str)

class ResizeHandle(QGraphicsRectItem):
    """真正的单边/角拖拽：句柄本身不作为可移动 item，而是根据鼠标移动 delta 调整父矩形。
    """
    def __init__(self, parent, pos_flag: str):
        super().__init__(-HANDLE_SIZE/2, -HANDLE_SIZE/2, HANDLE_SIZE, HANDLE_SIZE, parent)
        # 颜色：边=红，角=白
        if pos_flag in ("l", "r", "t", "b"):
            self.setBrush(QBrush(QColor(255, 100, 100)))
        else:
            self.setBrush(QBrush(QColor(255, 255, 255)))
        self.setPen(QPen(Qt.black, 2))
        # 不让 Qt 自动移动句柄，完全由我们控制
        self.setFlag(QGraphicsItem.ItemIsMovable, False)
        self.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self.setAcceptHoverEvents(True)
        self.setZValue(100)
        self.pos_flag = pos_flag
        if pos_flag in ("l","r"):
            self.setCursor(Qt.SizeHorCursor)
        elif pos_flag in ("t","b"):
            self.setCursor(Qt.SizeVerCursor)
        elif pos_flag in ("tl","br"):
            self.setCursor(Qt.SizeFDiagCursor)
        else:
            self.setCursor(Qt.SizeBDiagCursor)
        self.setToolTip(f"拖动{pos_flag}句柄调整大小")
        # 交互状态记录
        self._press_scene_pos = None  # 场景坐标下按下位置

    # -------- Hover --------
    def hoverEnterEvent(self, event):
        super().hoverEnterEvent(event)
        self.setOpacity(0.85)

    def hoverLeaveEvent(self, event):
        super().hoverLeaveEvent(event)
        self.setOpacity(1.0)

    # -------- Resize 手势 --------
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            parent: RoomRectItem = self.parentItem()  # type: ignore
            if parent:
                # 进入 resize 模式，禁止父项被当作整体拖动
                parent.setFlag(QGraphicsItem.ItemIsMovable, False)
                parent._active_handle_flag = self.pos_flag
                parent._orig_rect = parent.rect()
                self._press_scene_pos = event.scenePos()
                parent._resize_in_progress = True
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        parent: RoomRectItem = self.parentItem()  # type: ignore
        if parent and parent._resize_in_progress and self._press_scene_pos is not None:
            # 基于初始矩形 + delta 计算
            orig: QRectF = parent._orig_rect
            # 将鼠标位移转换到父局部坐标（避免场景缩放影响）
            delta_parent = parent.mapFromScene(event.scenePos()) - parent.mapFromScene(self._press_scene_pos)
            # 起始数值
            left = orig.left(); right = orig.right(); top = orig.top(); bottom = orig.bottom()
            flag = self.pos_flag
            dx = delta_parent.x(); dy = delta_parent.y()
            # 根据 flag 调整对应边
            if 'l' in flag:
                left += dx
            if 'r' in flag:
                right += dx
            if 't' in flag:
                top += dy
            if 'b' in flag:
                bottom += dy
            # 约束最小尺寸
            MIN_W = 5; MIN_H = 5
            if right - left < MIN_W:
                if 'l' in flag:
                    left = right - MIN_W
                else:
                    right = left + MIN_W
            if bottom - top < MIN_H:
                if 't' in flag:
                    top = bottom - MIN_H
                else:
                    bottom = top + MIN_H
            # 处理反转（用户拖反）
            if left > right:
                left, right = right, left
            if top > bottom:
                top, bottom = bottom, top
            parent.apply_new_rect(QRectF(left, top, right - left, bottom - top))
            event.accept()
            return
        # 不调用 super() 避免 Qt 位移句柄
        # super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        parent: RoomRectItem = self.parentItem()  # type: ignore
        if parent and parent._resize_in_progress:
            parent._resize_in_progress = False
            r = parent.rect()
            bbox = (int(r.left()), int(r.top()), int(r.right()), int(r.bottom()))
            parent.signals.bboxChanged.emit(parent.room.id, bbox)
            # 恢复整体可移动
            parent.setFlag(QGraphicsItem.ItemIsMovable, True)
            event.accept()
            return
        super().mouseReleaseEvent(event)

class RoomRectItem(QGraphicsRectItem):
    def __init__(self, room, color, signals: RoomGraphicsSignals):
        super().__init__()
        self.room = room
        x1, y1, x2, y2 = room.bbox
        self.setRect(QRectF(x1, y1, x2 - x1 + 1, y2 - y1 + 1))
        self.signals = signals
        self.color = color
        self.setBrush(QBrush(QColor(color[0], color[1], color[2], 90)))
        self.setPen(QPen(QColor(*color), 2))
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setAcceptHoverEvents(True)
        self.handles = []
        base_text = f"{room.type}{room.index}"
        self.label = QGraphicsSimpleTextItem(base_text, self)
        self._label_base_text = base_text
        self.label.setPos(x1 + 4, y1 + 4)
        self._lock_handle_update = False
        self._active_handle_flag = None
        self._orig_rect = self.rect()
        self._resize_in_progress = False
        self.create_handles()
        for handle in self.handles:
            handle.setVisible(False)

    # --------- 句柄 ---------
    def create_handles(self):
        self.handles.clear()
        for flag in ["tl","tr","bl","br","l","r","t","b"]:
            h = ResizeHandle(self, flag)
            h.setVisible(True)  # 确保句柄可见
            self.handles.append(h)
        self.update_handles()

    def update_handles(self):
        r = self.rect()
        cx = r.center().x(); cy = r.center().y()
        self._lock_handle_update = True
        try:
            for h in self.handles:
                if h.pos_flag == "tl": h.setPos(r.left(), r.top())
                elif h.pos_flag == "tr": h.setPos(r.right(), r.top())
                elif h.pos_flag == "bl": h.setPos(r.left(), r.bottom())
                elif h.pos_flag == "br": h.setPos(r.right(), r.bottom())
                elif h.pos_flag == "l": h.setPos(r.left(), cy)
                elif h.pos_flag == "r": h.setPos(r.right(), cy)
                elif h.pos_flag == "t": h.setPos(cx, r.top())
                elif h.pos_flag == "b": h.setPos(cx, r.bottom())
        finally:
            self._lock_handle_update = False

    def update_rect_from_handles(self):
        """综合四角与边句柄位置，允许单边拖拽。"""
        # 旧的基于句柄位置反推逻辑已弃用，保留函数以兼容其它调用路径
        pass

    # 新的统一应用矩形方法
    def apply_new_rect(self, new_rect: QRectF):
        old = self.rect()
        if (abs(old.left()-new_rect.left()) < 0.01 and
            abs(old.top()-new_rect.top()) < 0.01 and
            abs(old.width()-new_rect.width()) < 0.01 and
            abs(old.height()-new_rect.height()) < 0.01):
            return
        self.prepareGeometryChange()
        self.setRect(new_rect)
        # 实时尺寸展示
        if self._resize_in_progress:
            w = int(new_rect.width()); h = int(new_rect.height())
            self.label.setText(f"{self._label_base_text} {w}x{h}")
        else:
            self.label.setText(self._label_base_text)
        self.label.setPos(new_rect.left()+4, new_rect.top()+4)
        self.update_handles()

    def set_selected(self, selected: bool):
        """设置房间选中状态并控制句柄可见性"""
        self.setSelected(selected)
        for handle in self.handles:
            handle.setVisible(selected)
        
        # 更新边框样式提供视觉反馈
        if selected:
            pen = QPen(QColor(255, 100, 100), 3, Qt.SolidLine)  # 选中时红色粗边框
        else:
            pen = QPen(QColor(100, 100, 100), 2, Qt.SolidLine)  # 未选中时灰色细边框
        self.setPen(pen)

    # --------- 事件 ---------
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.signals.selected.emit(self.room.id)
            # 所有房间都可以移动
            self.setFlag(QGraphicsItem.ItemIsMovable, True)
            # 显示当前房间的句柄，隐藏其他房间的句柄
            scene = self.scene()
            if scene:
                for item in scene.items():
                    if isinstance(item, RoomRectItem):
                        item.set_selected(item == self)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        was_resize = self._resize_in_progress  # 理论上已在句柄里置 False
        # 如果是普通移动（非 resize），需要把 item 的平移折叠进 rect 以维持“rect=绝对坐标，pos=0”约定
        if not was_resize and not self._resize_in_progress:
            if not self.pos().isNull():
                dx = self.pos().x(); dy = self.pos().y()
                r = self.rect()
                new_rect = QRectF(r.left()+dx, r.top()+dy, r.width(), r.height())
                self.setPos(0,0)
                self.prepareGeometryChange()
                self.setRect(new_rect)
                self.update_handles()
                self.label.setPos(new_rect.left()+4, new_rect.top()+4)
        super().mouseReleaseEvent(event)
        # 统一发出 bbox 事件（此时 rect 已是绝对坐标）
        r = self.rect()
        bbox = (int(r.left()), int(r.top()), int(r.right()), int(r.bottom()))
        self.signals.bboxChanged.emit(self.room.id, bbox)
        # 保持可移动
        self.setFlag(QGraphicsItem.ItemIsMovable, True)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self.update_handles()
        self.label.setPos(self.rect().left() + 4, self.rect().top() + 4)

    def refresh_from_room(self):
        x1,y1,x2,y2 = self.room.bbox
        self.setRect(QRectF(x1, y1, x2 - x1 + 1, y2 - y1 + 1))
        self.label.setText(f"{self.room.type}{self.room.index}")
        self.label.setPos(x1 + 4, y1 + 4)
        self.update_handles()
