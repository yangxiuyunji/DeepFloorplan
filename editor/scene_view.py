from PySide6.QtWidgets import QGraphicsView, QGraphicsScene
from PySide6.QtCore import Qt, Signal, QObject, QTimer
from PySide6.QtGui import QPixmap, QImage, QPen, QBrush, QColor, QFont, QFontInfo, QPainterPath, QPainter
import math
import cv2
from .room_item import RoomRectItem, RoomGraphicsSignals
from .category_manager import CategoryManager


class SceneSignals(QObject):
    roomSelected = Signal(str)
    roomBBoxChanged = Signal(str, tuple)
    roomDeselected = Signal()  # 新增：取消选择信号
    mousePositionChanged = Signal(float, float)  # 新增：鼠标坐标信号


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
        self._has_fitted_once = False
        self.scale_factor = 1.0
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)  # 启用鼠标跟踪，即使没有按下按钮也能追踪鼠标移动
        # 指北针交互状态
        self._compass_size = 120  # 从88增加到120，更大的指示盘
        self._compass_margin = 80  # 从12增加到80，往左移动更多
        self._compass_custom_pos = None  # 视口坐标 (x,y) 顶左，None 表示右上角锚定
        self._compass_dragging = False
        self._compass_drag_offset = (0, 0)

    # ---------- 视图适配 ----------
    def fit_to_scene(self):
        # 直接计算最优缩放比例，确保图像铺满视口
        br = self.scene.itemsBoundingRect()
        if not (br.isValid() and br.width() > 0 and br.height() > 0):
            return
        
        # 获取视口尺寸，减去可能的滚动条空间
        vp_w = self.viewport().width() - 10
        vp_h = self.viewport().height() - 10
        if vp_w <= 0 or vp_h <= 0:
            return
        
        # 计算缩放比例以铺满视口
        scale_x = vp_w / br.width()
        scale_y = vp_h / br.height()
        # 选择较小的缩放比例以保持宽高比且不裁剪，留少量边距
        scale_ratio = min(scale_x, scale_y) * 0.9
        
        # 重置所有变换并应用新的缩放
        self.resetTransform()
        self.scale(scale_ratio, scale_ratio)
        self.scale_factor = scale_ratio
        
        # 居中显示
        self.centerOn(br.center())
        self._has_fitted_once = True

    # ---------- 交互 ----------
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 1.25 if delta > 0 else 0.8
        self.scale(factor, factor)
        self.scale_factor *= factor

    # ---------- 文档加载 ----------
    def load_document(self, doc):
        try:
            print(f"SceneView.load_document: Starting with {len(doc.rooms)} rooms")
            
            # 明确清理之前的背景和遮罩项
            print(f"SceneView.load_document: Clearing previous items")
            if hasattr(self, '_bg_item') and self._bg_item:
                self.scene.removeItem(self._bg_item)
                self._bg_item = None
            if hasattr(self, '_mask_item') and self._mask_item:
                self.scene.removeItem(self._mask_item)
                self._mask_item = None
                
            self.scene.clear()
            self._room_items.clear()
            self._doc = doc
            
            print(f"SceneView.load_document: Adding background image")
            if doc.image_path:
                self._bg_item = self._add_pixmap(doc.image_path, z=-20, opacity=1.0)
            if doc.mask_path and self._show_mask:
                self._mask_item = self._add_pixmap(doc.mask_path, z=-10, opacity=0.35)
            
            print(f"SceneView.load_document: Creating category manager")
            self.cat_mgr = CategoryManager(doc.categories)
            
            print(f"SceneView.load_document: Adding room items")
            for i, r in enumerate(doc.rooms):
                print(f"  Adding room {i}: {r.id} ({r.type}{r.index}) bbox={r.bbox}")
                
                # 检查房间ID是否重复
                if r.id in self._room_items:
                    print(f"  WARNING: Room ID {r.id} already exists in _room_items")
                
                try:
                    color = self.cat_mgr.get_color(r.type)
                    sigs = RoomGraphicsSignals()
                    item = RoomRectItem(r, color, sigs)
                    sigs.selected.connect(self.signals.roomSelected)
                    sigs.bboxChanged.connect(self.signals.roomBBoxChanged)
                    self.scene.addItem(item)
                    self._room_items[r.id] = item
                    print(f"  Room {r.id} added successfully")
                except Exception as room_err:
                    print(f"  ERROR adding room {r.id}: {room_err}")
                    import traceback
                    traceback.print_exc()
                    raise
            
            print(f"SceneView.load_document: Setting scene rect")
            self.setSceneRect(self.scene.itemsBoundingRect())
            
            print(f"SceneView.load_document: Scheduling fit_to_scene")
            # 延迟 fit 以确保窗口尺寸已确定，并增加延迟时间确保渲染完成
            QTimer.singleShot(100, self.fit_to_scene)
            
            print(f"SceneView.load_document: Completed successfully")
            
        except Exception as e:
            print(f"FATAL ERROR in SceneView.load_document: {e}")
            import traceback
            traceback.print_exc()
            raise

    # ---------- 工具 ----------
    def _add_pixmap(self, path: str, z=-5, opacity=1.0):
        try:
            print(f"_add_pixmap: Loading image from {path}")
            img = cv2.imread(path)
            if img is None:
                print(f"_add_pixmap: Failed to load image from {path}")
                return None
            print(f"_add_pixmap: Image loaded successfully, shape: {img.shape}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = img.shape
            print(f"_add_pixmap: Converting to QImage, size: {w}x{h}")
            
            qimg = QImage(img.data, w, h, w * 3, QImage.Format_RGB888)
            print(f"_add_pixmap: QImage created successfully")
            
            pm = QPixmap.fromImage(qimg)
            print(f"_add_pixmap: QPixmap created successfully")
            
            item = self.scene.addPixmap(pm)
            print(f"_add_pixmap: Pixmap added to scene")
            
            item.setZValue(z)
            item.setOpacity(opacity)
            print(f"_add_pixmap: Z-value and opacity set")
            
            return item
        except Exception as e:
            print(f"FATAL ERROR in _add_pixmap: {e}")
            import traceback
            traceback.print_exc()
            raise

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

    # ---------- 指北针 ----------
    def drawForeground(self, painter, rect):
        super().drawForeground(painter, rect)
        if not self._doc:
            return
        angle = getattr(self._doc, 'north_angle', 0)  # 0=北 90=东 顺时针
        orientation_text = getattr(self._doc, 'house_orientation', '')
        painter.save()
        try:
            painter.resetTransform()  # 使用视口坐标，避免随缩放变化
            w = self.viewport().width()
            h = self.viewport().height()
            size = self._compass_size
            # 计算位置
            if self._compass_custom_pos is None:
                x = w - self._compass_margin - size
                y = self._compass_margin
            else:
                x, y = self._compass_custom_pos
                if x + size > w: x = w - size - 2
                if y + size > h: y = h - size - 2
                self._compass_custom_pos = (x, y)
            cx = x + size / 2
            cy = y + size / 2
            painter.setRenderHint(QPainter.Antialiasing, True)
            # 阴影
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(0, 0, 0, 90)))
            painter.drawEllipse(cx - size/2 + 3, cy - size/2 + 3, size, size)
            # 背景圆
            painter.setPen(QPen(QColor(30, 30, 30, 200), 1))
            painter.setBrush(QBrush(QColor(255, 255, 255, 210)))
            painter.drawEllipse(cx - size/2, cy - size/2, size, size)
            # 刻度
            for a in range(0, 360, 45):
                rad = math.radians(a)
                ox = cx + math.cos(rad) * (size * 0.42)
                oy = cy - math.sin(rad) * (size * 0.42)
                ix = cx + math.cos(rad) * (size * 0.32)
                iy = cy - math.sin(rad) * (size * 0.32)
                painter.drawLine(ox, oy, ix, iy)
            # 箭头 - 在新系统中0°=北(上方)，需要转换为数学坐标系
            # 数学坐标系：0°=右，90°=上，所以需要转换
            display_angle = (90 - angle) % 360  # 转换：0°(北)→90°(上)
            rad_n = math.radians(display_angle)
            nx = math.cos(rad_n)
            ny = math.sin(rad_n)
            tip_len = size * 0.36
            tip = (cx + nx * tip_len, cy - ny * tip_len)
            perp_x = -ny
            perp_y = nx
            half_w = size * 0.10
            base_center = (cx, cy)
            p1 = (base_center[0] + perp_x * half_w, base_center[1] - perp_y * half_w)
            p2 = (base_center[0] - perp_x * half_w, base_center[1] + perp_y * half_w)
            path = QPainterPath()
            path.moveTo(*tip)
            path.lineTo(*p1)
            path.lineTo(*p2)
            path.closeSubpath()
            painter.setBrush(QBrush(QColor(220, 30, 30)))
            painter.setPen(QPen(QColor(120, 0, 0), 1))
            painter.drawPath(path)
            # 中心点
            painter.setBrush(QBrush(QColor(50, 50, 50)))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(cx - 3, cy - 3, 6, 6)
            # 8个方位中文标记 - 移到指示盘外面，使用楷体
            font = QFont()
            # 设置字体族，优先使用楷体，回退到其他中文字体
            font_families = ["KaiTi", "楷体", "SimKai", "DFKai-SB", "Microsoft YaHei", "SimHei", "SimSun"]
            for family in font_families:
                font.setFamily(family)
                if QFontInfo(font).family() == family:
                    break  # 找到可用字体就停止
            
            font.setPointSize(11)    # 适中的字体大小
            font.setBold(True)         # 保持加粗以提高可读性
            painter.setFont(font)
            
            # 设置文字颜色和阴影效果以提高可见性
            painter.setPen(QPen(QColor(0, 0, 0), 2))  # 黑色文字，较粗线条
            
            def _place(label, ang_deg):
                # 根据方位类型调整半径，斜方位文字更长需要更大半径
                if label in ['东南', '东北', '西南', '西北']:
                    r = size * 0.7  # 斜方位使用更大半径避免被遮挡
                else:
                    r = size * 0.65  # 正方位使用标准半径
                
                # 转换到数学坐标系
                display_deg = (90 - ang_deg) % 360
                rad = math.radians(display_deg)
                tx = cx + math.cos(rad) * r
                ty = cy - math.sin(rad) * r
                
                # 更精确的文字尺寸计算
                text_width = len(label) * 11  # 根据字体大小调整
                text_height = 20
                
                # 绘制文字阴影效果（白色描边）提高可见性
                painter.setPen(QPen(QColor(255, 255, 255), 4))
                painter.drawText(int(tx - text_width/2), int(ty - text_height/2), text_width, text_height, Qt.AlignCenter, label)
                
                # 绘制主文字（黑色）
                painter.setPen(QPen(QColor(0, 0, 0), 2))
                painter.drawText(int(tx - text_width/2), int(ty - text_height/2), text_width, text_height, Qt.AlignCenter, label)
            
            # 8个方位：北、东北、东、东南、南、西南、西、西北
            _place('北', angle)                    # 北方按当前角度
            _place('东北', (angle + 45) % 360)     # 东北 = 北方 + 45°
            _place('东', (angle + 90) % 360)       # 东方 = 北方 + 90°
            _place('东南', (angle + 135) % 360)    # 东南 = 北方 + 135°
            _place('南', (angle + 180) % 360)      # 南方 = 北方 + 180°
            _place('西南', (angle + 225) % 360)    # 西南 = 北方 + 225°
            _place('西', (angle + 270) % 360)      # 西方 = 北方 + 270°
            _place('西北', (angle + 315) % 360)    # 西北 = 北方 + 315°
            # 角度与朝向文本
            painter.drawText(x, y + size + 2, size, 16, Qt.AlignCenter, f"{angle}°")
            if orientation_text:
                # 增加"坐X朝X"文字的向下偏移，从18改为35
                painter.drawText(x - 20, y + size + 35, size + 40, 16, Qt.AlignCenter, orientation_text)
        finally:
            painter.restore()

    # ---------- 指北针拖动 ----------
    def _compass_hit(self, pos):
        size = self._compass_size
        if self._compass_custom_pos is None:
            w = self.viewport().width()
            x = w - self._compass_margin - size
            y = self._compass_margin
        else:
            x, y = self._compass_custom_pos
        return (x <= pos.x() <= x + size) and (y <= pos.y() <= y + size)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._compass_hit(event.pos()):
            # 开始拖动指北针
            size = self._compass_size
            if self._compass_custom_pos is None:
                w = self.viewport().width()
                x = w - self._compass_margin - size
                y = self._compass_margin
            else:
                x, y = self._compass_custom_pos
            self._compass_dragging = True
            self._compass_drag_offset = (event.pos().x() - x, event.pos().y() - y)
            event.accept()
            return
        
        # 检查是否点击了房间项
        scene_pos = self.mapToScene(event.pos())
        clicked_item = self.scene.itemAt(scene_pos, self.transform())
        
        # 如果点击的不是房间项（即点击空白区域），发送取消选择信号
        if event.button() == Qt.LeftButton:
            is_room_item = False
            current_item = clicked_item
            # 检查是否点击了房间或房间的子项（如标签）
            while current_item:
                if isinstance(current_item, RoomRectItem):
                    is_room_item = True
                    break
                current_item = current_item.parentItem()
            
            if not is_room_item:
                # 点击空白区域，取消选择
                self.signals.roomDeselected.emit()
                # 清除场景中所有房间的选择状态
                for item in self.scene.items():
                    if hasattr(item, 'setSelected'):
                        item.setSelected(False)
        
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # 发射鼠标坐标信号
        scene_pos = self.mapToScene(event.pos())
        self.signals.mousePositionChanged.emit(scene_pos.x(), scene_pos.y())
        
        if self._compass_dragging:
            ox, oy = self._compass_drag_offset
            nx = event.pos().x() - ox
            ny = event.pos().y() - oy
            # 限制在视口内
            w = self.viewport().width()
            h = self.viewport().height()
            size = self._compass_size
            nx = max(2, min(nx, w - size - 2))
            ny = max(2, min(ny, h - size - 2))
            self._compass_custom_pos = (nx, ny)
            self.viewport().update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._compass_dragging and event.button() == Qt.LeftButton:
            self._compass_dragging = False
            event.accept()
            return
        super().mouseReleaseEvent(event)
