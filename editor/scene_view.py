from PySide6.QtWidgets import QGraphicsView, QGraphicsScene
from PySide6.QtCore import Qt, Signal, QObject, QTimer
from PySide6.QtGui import QPixmap, QImage
import cv2
from .room_item import RoomRectItem, RoomGraphicsSignals
from .category_manager import CategoryManager


class SceneSignals(QObject):
    roomSelected = Signal(str)
    roomBBoxChanged = Signal(str, tuple)


class FloorplanSceneView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.signals = SceneSignals()
        self._room_items = {}
        self._doc = None
        self._bg_item = None
        self._mask_item = None
        self._show_mask = True
        self.scale_factor = 1.0
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

    # ---------- 视图适配 ----------
    def fit_to_scene(self):
        br = self.scene.itemsBoundingRect().adjusted(-10, -10, 10, 10)
        if br.isValid() and br.width() > 0 and br.height() > 0:
            self.fitInView(br, Qt.KeepAspectRatio)
            self.scale_factor = 1.0

    # ---------- 交互 ----------
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 1.25 if delta > 0 else 0.8
        self.scale(factor, factor)
        self.scale_factor *= factor

    # ---------- 文档加载 ----------
    def load_document(self, doc):
        self.scene.clear()
        self._room_items.clear()
        self._doc = doc
        if doc.image_path:
            self._bg_item = self._add_pixmap(doc.image_path, z=-20, opacity=1.0)
        if doc.mask_path and self._show_mask:
            self._mask_item = self._add_pixmap(doc.mask_path, z=-10, opacity=0.35)
        self.cat_mgr = CategoryManager(doc.categories)
        for r in doc.rooms:
            color = self.cat_mgr.get_color(r.type)
            sigs = RoomGraphicsSignals()
            item = RoomRectItem(r, color, sigs)
            sigs.selected.connect(self.signals.roomSelected)
            sigs.bboxChanged.connect(self.signals.roomBBoxChanged)
            self.scene.addItem(item)
            self._room_items[r.id] = item
        self.setSceneRect(self.scene.itemsBoundingRect())
        # 下一事件循环再 fit，保证窗口尺寸已确定
        QTimer.singleShot(0, self.fit_to_scene)

    # ---------- 工具 ----------
    def _add_pixmap(self, path: str, z=-5, opacity=1.0):
        img = cv2.imread(path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        qimg = QImage(img.data, w, h, w * 3, QImage.Format_RGB888)
        pm = QPixmap.fromImage(qimg)
        item = self.scene.addPixmap(pm)
        item.setZValue(z)
        item.setOpacity(opacity)
        return item

    def toggle_mask(self, show: bool):
        self._show_mask = show
        if not self._doc or not self._doc.mask_path:
            return
        if show and not self._mask_item:
            self._mask_item = self._add_pixmap(self._doc.mask_path, z=-10, opacity=0.35)
        elif not show and self._mask_item:
            self.scene.removeItem(self._mask_item)
            self._mask_item = None

    def refresh_room(self, room):
        item = self._room_items.get(room.id)
        if item:
            item.refresh_from_room()

    def add_room_item(self, room):
        """添加单个房间项，不重新加载整个文档"""
        if not self._doc:
            return
        color = self.cat_mgr.get_color(room.type)
        sigs = RoomGraphicsSignals()
        item = RoomRectItem(room, color, sigs)
        sigs.selected.connect(self.signals.roomSelected)
        sigs.bboxChanged.connect(self.signals.roomBBoxChanged)
        self.scene.addItem(item)
        self._room_items[room.id] = item

    def remove_room_item(self, room_id):
        """移除单个房间项"""
        item = self._room_items.get(room_id)
        if item:
            self.scene.removeItem(item)
            del self._room_items[room_id]
