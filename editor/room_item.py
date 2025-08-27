from PySide6.QtWidgets import QGraphicsRectItem, QGraphicsItem, QGraphicsSimpleTextItem
from PySide6.QtGui import QPen, QBrush, QColor, QCursor
from PySide6.QtCore import Qt, QRectF, QPointF, Signal, QObject

HANDLE_SIZE = 10

class RoomGraphicsSignals(QObject):
    bboxChanged = Signal(str, tuple)   # room_id, bbox
    selected = Signal(str)

class ResizeHandle(QGraphicsRectItem):
    def __init__(self, parent, pos_flag: str):
        super().__init__(-HANDLE_SIZE/2, -HANDLE_SIZE/2, HANDLE_SIZE, HANDLE_SIZE, parent)
        self.setBrush(QBrush(QColor(255,255,255)))
        self.setPen(QPen(Qt.black,1))
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setZValue(100)  # 确保句柄在最上层
        self.pos_flag = pos_flag
        # 根据位置设置合适的鼠标光标
        if pos_flag in ("l","r"):
            self.setCursor(Qt.SizeHorCursor)
        elif pos_flag in ("t","b"):
            self.setCursor(Qt.SizeVerCursor)
        elif pos_flag in ("tl","br"):
            self.setCursor(Qt.SizeFDiagCursor)
        else:  # tr, bl
            self.setCursor(Qt.SizeBDiagCursor)
        self.setToolTip("拖动白色方块调整大小，拖动房间主体移动位置")

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and self.parentItem():
            parent = self.parentItem()
            # 轴向约束：边句柄仅沿一个轴移动
            if self.pos_flag == 'l' or self.pos_flag == 'r':
                # 锁定 y 到父矩形中心
                value.setY(parent.rect().center().y())
            elif self.pos_flag == 't' or self.pos_flag == 'b':
                value.setX(parent.rect().center().x())
            if not getattr(parent, "_lock_handle_update", False):
                parent._lock_handle_update = True
                parent.update_rect_from_handles()
                parent._lock_handle_update = False
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        parent = self.parentItem()
        if parent and isinstance(parent, RoomRectItem):
            r = parent.rect()
            bbox = (int(r.left()), int(r.top()), int(r.right()), int(r.bottom()))
            parent.signals.bboxChanged.emit(parent.room.id, bbox)

class RoomRectItem(QGraphicsRectItem):
    def __init__(self, room, color, signals: RoomGraphicsSignals):
        super().__init__()
        self.room = room
        x1,y1,x2,y2 = room.bbox
        self.setRect(QRectF(x1, y1, x2 - x1 + 1, y2 - y1 + 1))
        self.signals = signals
        self.color = color
        self.setBrush(QBrush(QColor(color[0], color[1], color[2], 90)))
        self.setPen(QPen(QColor(*color), 2))
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        # 所有房间都允许整体移动，但可以通过设置区分行为
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setAcceptHoverEvents(True)
        self.handles = []
        self.label = QGraphicsSimpleTextItem(f"{room.type}{room.index}", self)
        self.label.setPos(x1 + 4, y1 + 4)
        self._lock_handle_update = False
        self.create_handles()

    # --------- 句柄 ---------
    def create_handles(self):
        self.handles.clear()
        for flag in ["tl","tr","bl","br","l","r","t","b"]:
            h = ResizeHandle(self, flag)
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
        # 收集位置
        pos_map = {h.pos_flag: h.pos() for h in self.handles}
        r = self.rect()
        # 初值：若四角齐全用四角，否则用当前矩形
        if all(k in pos_map for k in ("tl","tr","bl","br")):
            left = min(pos_map['tl'].x(), pos_map['bl'].x())
            right = max(pos_map['tr'].x(), pos_map['br'].x())
            top = min(pos_map['tl'].y(), pos_map['tr'].y())
            bottom = max(pos_map['bl'].y(), pos_map['br'].y())
        else:
            left, top, right, bottom = r.left(), r.top(), r.right(), r.bottom()
        # 边句柄覆盖（单边移动）
        if 'l' in pos_map:
            left = pos_map['l'].x()
        if 'r' in pos_map:
            right = pos_map['r'].x()
        if 't' in pos_map:
            top = pos_map['t'].y()
        if 'b' in pos_map:
            bottom = pos_map['b'].y()
        # 规范与最小尺寸
        if right < left:
            left, right = right, left
        if bottom < top:
            top, bottom = bottom, top
        if right - left < 5:
            right = left + 5
        if bottom - top < 5:
            bottom = top + 5
        self.prepareGeometryChange()
        self.setRect(QRectF(left, top, right - left, bottom - top))
        self.label.setPos(self.rect().left() + 4, self.rect().top() + 4)
        # 重新同步所有句柄位置（防止继续拖动基于旧坐标）
        self.update_handles()

    # --------- 事件 ---------
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.signals.selected.emit(self.room.id)
            # 所有房间都可以移动
            self.setFlag(QGraphicsItem.ItemIsMovable, True)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        # 保持所有房间可移动
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        r = self.rect()
        bbox = (int(r.left()), int(r.top()), int(r.right()), int(r.bottom()))
        self.signals.bboxChanged.emit(self.room.id, bbox)

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
