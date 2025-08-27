import sys, uuid, argparse, os
from pathlib import Path
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt
from .json_io import load_floorplan_json, save_floorplan_json
from .models import RoomModel
from .scene_view import FloorplanSceneView
from .property_panel import PropertyPanel
from .undo_stack import UndoStack, UndoCommand

BASE_TYPES = ["厨房","卫生间","客厅","卧室","阳台","书房"]

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("户型 JSON 编辑器 (MVP)")
        self.view = FloorplanSceneView()
        self.setCentralWidget(self.view)
        self.prop = PropertyPanel()
        self.addDockWidget(Qt.RightDockWidgetArea, self.prop.dock)
        self.doc = None
        self.undo = UndoStack(limit=100)
        self._connect()
        self._build_menu()

    # ---------- UI 构建 ----------
    def _build_menu(self):
        # 顶部菜单栏
        m_file = self.menuBar().addMenu("文件")
        act_open = QAction("打开 JSON", self)
        act_open.triggered.connect(self.open_json)
        act_save = QAction("保存", self)
        act_save.triggered.connect(self.save_json)
        act_toggle_bg = QAction("切换原图/结果图", self)
        act_toggle_bg.triggered.connect(self.toggle_background_image)
        act_undo = QAction("撤销", self)
        act_undo.triggered.connect(self.on_undo)
        act_redo = QAction("重做", self)
        act_redo.triggered.connect(self.on_redo)

        m_file.addAction(act_open)
        m_file.addAction(act_save)
        m_file.addAction(act_toggle_bg)

        m_edit = self.menuBar().addMenu("编辑")
        m_edit.addAction(act_undo)
        m_edit.addAction(act_redo)

        # 视图菜单
        m_view = self.menuBar().addMenu("视图")
        act_reset_view = QAction("重置视图", self)
        act_reset_view.setShortcut("Ctrl+0")
        def _reset_view():
            self.view._has_fitted_once = False  # 允许再次 fit
            self.view.fit_to_scene()
        act_reset_view.triggered.connect(_reset_view)
        m_view.addAction(act_reset_view)

        # 快捷键
        act_save.setShortcut("Ctrl+S")
        act_undo.setShortcut("Ctrl+Z")
        act_redo.setShortcut("Ctrl+Y")

    def _connect(self):
        self.view.signals.roomSelected.connect(self.on_room_selected)
        self.view.signals.roomBBoxChanged.connect(self.on_room_bbox_changed)
        self.prop.signals.roomFieldEdited.connect(self.on_room_field_edited)
        self.prop.signals.deleteRoom.connect(self.delete_room)
        self.prop.signals.addRoom.connect(self.add_room_default)
        self.prop.signals.saveRequest.connect(self.save_json)
        self.prop.signals.toggleMask.connect(self.toggle_mask)

    # ---------- 文件 ----------
    def open_json(self):
        fn, _ = QFileDialog.getOpenFileName(self, "选择 JSON", "", "JSON (*.json)")
        if not fn:
            return
        self._load_json_path(fn)

    def _load_json_path(self, fn: str):
        try:
            self.doc = load_floorplan_json(fn)
            self._auto_locate_mask(self.doc)
            self._guess_original_image(self.doc, Path(fn))
            # 默认：如果同时有原图与结果图，优先显示原图（相当于自动点了一次“切换原图”）
            if self.doc.original_image_path and self.doc.result_image_path:
                self.doc.image_path = self.doc.original_image_path
            self.view.load_document(self.doc)
            self.prop.load_categories(self.doc.categories)
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))

    def _auto_locate_mask(self, doc):
        if not doc.image_path:
            return
        p = Path(doc.image_path)
        candidates = [
            p.with_name(p.stem.replace("_result", "") + "_boundary_mask.png"),
            p.with_name(p.stem + "_boundary_mask.png"),
            p.with_name(p.stem + "_coordinate_result.png"),
            p,
        ]
        for c in candidates:
            if c.exists() and c.suffix.lower() in (".png", ".jpg", ".jpeg"):
                if "boundary_mask" in c.name and not doc.mask_path:
                    doc.mask_path = str(c)
                if not doc.image_path or c == p:
                    # 将首次识别到的结果图记录为 result_image_path
                    doc.result_image_path = str(c)
                    doc.image_path = str(c)
        if doc.image_path and not Path(doc.image_path).exists():
            maybe = Path(doc.json_path).parent / Path(doc.image_path).name
            if maybe.exists():
                doc.image_path = str(maybe)
        # 结果底图兜底：如果没有找到明确的 result 图，再尝试按 JSON 文件名推测 *_result.png
        if not doc.result_image_path and doc.json_path:
            jp = Path(doc.json_path)
            base = jp.stem
            if base.endswith('_edited'):
                base = base[:-7]
            guess_list = [
                jp.parent / f"{base}.png",
                jp.parent / f"{base}.jpg",
                jp.parent / f"{base}_result.png",
                jp.parent / f"{base}_result.jpg",
            ]
            for g in guess_list:
                if g.exists():
                    doc.result_image_path = str(g)
                    if not doc.image_path:
                        doc.image_path = str(g)
                    break

    def _guess_original_image(self, doc, json_path: Path):
        """尝试根据 JSON 名称推测原始输入图 (jpg/png)。"""
        if doc.original_image_path:
            return
        # 基名去掉 _result/_edited 等后缀
        stem = json_path.stem
        for suffix in ["_edited", "_result"]:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
        search_dirs = [json_path.parent, json_path.parent.parent / 'demo', json_path.parent.parent]
        candidates = []
        for d in search_dirs:
            if not d.exists():
                continue
            candidates += [d / f"{stem}.jpg", d / f"{stem}.png", d / f"{stem}.jpeg"]
        for c in candidates:
            if c.exists():
                doc.original_image_path = str(c)
                # 仅记录原图，不覆盖当前默认（保持 result 图为默认底图）
                break

    def toggle_background_image(self):
        if not self.doc:
            return
        # 如果有 original 与 result 两种，就切换；否则忽略
        if self.doc.original_image_path and self.doc.result_image_path:
            current = self.doc.image_path
            self.doc.image_path = (
                self.doc.result_image_path if current == self.doc.original_image_path else self.doc.original_image_path
            )
            self.view.load_document(self.doc)
        else:
            QMessageBox.information(self, "提示", "未找到可切换的原图/结果图。")

    def save_json(self):
        if not self.doc:
            return
        try:
            path = save_floorplan_json(self.doc)
            QMessageBox.information(self, "保存", f"已保存: {path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))

    # ---------- 选择 ----------
    def on_room_selected(self, rid: str):
        if not self.doc:
            return
        room = self.doc.get_room(rid)
        self.prop.show_room(room)

    # ---------- BBox 变化 ----------
    def on_room_bbox_changed(self, rid: str, bbox: tuple):
        if not self.doc:
            return
        room = self.doc.get_room(rid)
        old_bbox = room.bbox
        def redo():
            room.update_bbox(bbox, self.doc.img_w, self.doc.img_h)
            self.view.refresh_room(room); self.prop.refresh(room)
            self.doc.ensure_indices()
        def undo():
            room.update_bbox(old_bbox, self.doc.img_w, self.doc.img_h)
            self.view.refresh_room(room); self.prop.refresh(room)
            self.doc.ensure_indices()
        redo(); self.undo.push(UndoCommand(redo, undo, "bbox"))

    # ---------- 字段变化 ----------
    def on_room_field_edited(self, rid: str, field: str, value):
        if not self.doc:
            return
        room = self.doc.get_room(rid)
        if not room:
            return
        old_snapshot = (room.type, room.text_raw, room.label_id)
        def redo():
            if field == "type":
                if value not in self.doc.categories:
                    self.doc.categories.append(value)
                # 若无 label_id，为新类别分配
                room.label_id = self.doc.assign_label_for_category(value)
            room.update_field(field, value, self.doc.img_w, self.doc.img_h)
            self.doc.ensure_indices()
            self.view.refresh_room(room); self.prop.refresh(room)
        def undo():
            room.type, room.text_raw, room.label_id = old_snapshot
            room.recompute(self.doc.img_w, self.doc.img_h)
            self.doc.ensure_indices()
            self.view.refresh_room(room); self.prop.refresh(room)
        redo(); self.undo.push(UndoCommand(redo, undo, field))

    # ---------- 删除/新增 ----------
    def delete_room(self, rid: str):
        if not self.doc: return
        room = self.doc.get_room(rid)
        if not room: return
        idx = self.doc.rooms.index(room)
        def redo():
            self.doc.rooms.remove(room)
            self.view.load_document(self.doc)
        def undo():
            self.doc.rooms.insert(idx, room)
            self.view.load_document(self.doc)
        redo(); self.undo.push(UndoCommand(redo, undo, "delete"))

    def add_room_default(self):
        if not self.doc: return
        w, h = 120, 90
        cx, cy = self.doc.img_w // 2, self.doc.img_h // 2
        x1 = max(0, cx - w // 2)
        y1 = max(0, cy - h // 2)
        x2 = min(self.doc.img_w - 1, x1 + w)
        y2 = min(self.doc.img_h - 1, y1 + h)
        new_type = "新房间"
        label_id = self.doc.assign_label_for_category(new_type)
        new_room = RoomModel(
            id=str(uuid.uuid4()),
            type=new_type,
            label_id=label_id,
            bbox=(x1, y1, x2, y2),
            text_raw="",
            confidence=0.0,
            source="manual",
            edited=True,
        )
        new_room.recompute(self.doc.img_w, self.doc.img_h)
        def redo():
            self.doc.rooms.append(new_room)
            self.doc.ensure_indices()
            # 只刷新新增的房间，不重新加载整个文档
            self.view.add_room_item(new_room)
        def undo():
            self.doc.rooms.remove(new_room)
            self.view.remove_room_item(new_room.id)
        redo(); self.undo.push(UndoCommand(redo, undo, "add"))

    # ---------- 其它 ----------
    def toggle_mask(self, _):
        self.view.toggle_mask(not self.view._show_mask)

    def on_undo(self):
        self.undo.undo();
        if self.doc: self.doc.ensure_indices(); self.view.load_document(self.doc)

    def on_redo(self):
        self.undo.redo();
        if self.doc: self.doc.ensure_indices(); self.view.load_document(self.doc)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", help="直接加载指定 JSON")
    args = parser.parse_args()
    app = QApplication(sys.argv)
    w = MainWindow()
    # 窗口居中并尽可能填满屏幕（使用最大化展示底图全屏）
    w.showMaximized()
    if args.json and os.path.exists(args.json):
        w._load_json_path(args.json)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
