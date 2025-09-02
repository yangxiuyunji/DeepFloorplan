import sys, argparse, os
from pathlib import Path
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt

# 添加上级目录到路径以支持绝对导入
sys.path.insert(0, str(Path(__file__).parent.parent))

from editor.json_io import load_floorplan_json, save_floorplan_json
from editor.models import RoomModel
from editor.scene_view import FloorplanSceneView
from editor.property_panel import PropertyPanel
from editor.undo_stack import UndoStack, UndoCommand

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
        # 连接房间相关信号
        self.view.signals.roomSelected.connect(self.on_room_selected)
        self.view.signals.roomBBoxChanged.connect(self.on_room_bbox_changed)
        self.view.signals.roomDeselected.connect(self.on_room_deselected)  # 新增：取消选择信号
        # 属性面板信号
        self.prop.signals.roomFieldEdited.connect(self.on_room_field_edited)
        self.prop.signals.deleteRoom.connect(self.delete_room)
        self.prop.signals.addRoom.connect(self.add_room_default)
        self.prop.signals.saveRequest.connect(self.save_json)
        self.prop.signals.toggleMask.connect(self.toggle_mask)
        self.prop.signals.globalFieldEdited.connect(self.on_global_field_edited)

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
            # 初始化全局属性显示
            self.prop.set_global_fields(self.doc.house_orientation, self.doc.north_angle)
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
        
        from pathlib import Path
        import os
        
        # 获取当前图像路径的目录
        json_dir = Path(self.doc.json_path).parent if self.doc.json_path else Path.cwd()
        
        # 检查原图和结果图是否存在
        original_exists = self.doc.original_image_path and os.path.exists(self.doc.original_image_path)
        result_exists = self.doc.result_image_path and os.path.exists(self.doc.result_image_path)
        
        # 如果两个图都存在，直接切换
        if original_exists and result_exists:
            current = self.doc.image_path
            self.doc.image_path = (
                self.doc.result_image_path if current == self.doc.original_image_path else self.doc.original_image_path
            )
            self.view.load_document(self.doc)
            return
        
        # 如果当前显示的是原图，尝试切换到结果图
        current_is_original = (self.doc.image_path == self.doc.original_image_path)
        
        if current_is_original:
            # 当前是原图，尝试切换到结果图
            if not result_exists:
                # 结果图不存在，让用户选择
                file_path, _ = QFileDialog.getOpenFileName(
                    self, "选择结果图", 
                    str(json_dir),
                    "图像文件 (*.png *.jpg *.jpeg *.bmp)"
                )
                if file_path:
                    self.doc.result_image_path = file_path
                    self.doc.image_path = file_path
                    self.view.load_document(self.doc)
                    QMessageBox.information(self, "提示", f"已设置结果图: {file_path}")
                return
            else:
                self.doc.image_path = self.doc.result_image_path
                self.view.load_document(self.doc)
        else:
            # 当前是结果图或其他，尝试切换到原图
            if not original_exists:
                # 原图不存在，让用户选择
                file_path, _ = QFileDialog.getOpenFileName(
                    self, "选择原图", 
                    str(json_dir),
                    "图像文件 (*.png *.jpg *.jpeg *.bmp)"
                )
                if file_path:
                    self.doc.original_image_path = file_path
                    self.doc.image_path = file_path
                    self.view.load_document(self.doc)
                    QMessageBox.information(self, "提示", f"已设置原图: {file_path}")
                return
            else:
                self.doc.image_path = self.doc.original_image_path
                self.view.load_document(self.doc)

    def save_json(self):
        if not self.doc:
            return
        try:
            path = save_floorplan_json(self.doc)
            QMessageBox.information(self, "保存", f"已保存: {path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))
        # 保存后刷新全局属性（保持 UI 同步）
        if self.doc:
            self.prop.set_global_fields(self.doc.house_orientation, self.doc.north_angle)

    # ---------- 选择 ----------
    def on_room_selected(self, rid: str):
        if not self.doc:
            return
        room = self.doc.get_room(rid)
        self.prop.show_room(room)

    def on_room_deselected(self):
        """处理取消选择房间"""
        self.prop.show_room(None)  # 清空属性面板

    # ---------- BBox 变化 ----------
    def on_room_bbox_changed(self, rid: str, bbox: tuple):
        if not self.doc:
            return
        room = self.doc.get_room(rid)
        if not room:
            return
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
            # 如果类型改变，需要重新确保统一编号
            if field == "type":
                self.doc.ensure_unified_room_naming()
            else:
                self.doc.ensure_indices()
            self.view.refresh_room(room); self.prop.refresh(room)
        def undo():
            room.type, room.text_raw, room.label_id = old_snapshot
            room.recompute(self.doc.img_w, self.doc.img_h)
            # 恢复类型后也要重新确保统一编号
            self.doc.ensure_unified_room_naming()
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
            # 删除房间后重新确保统一编号
            self.doc.ensure_unified_room_naming()
            self.view.load_document(self.doc)
        def undo():
            self.doc.rooms.insert(idx, room)
            # 恢复后也重新确保统一编号
            self.doc.ensure_unified_room_naming()
            self.view.load_document(self.doc)
        redo(); self.undo.push(UndoCommand(redo, undo, "delete"))

    def add_room_default(self):
        if not self.doc: 
            print("ERROR: No document loaded")
            return
        
        try:
            w, h = 120, 90
            cx, cy = self.doc.img_w // 2, self.doc.img_h // 2
            x1 = max(0, cx - w // 2)
            y1 = max(0, cy - h // 2)
            x2 = min(self.doc.img_w - 1, x1 + w)
            y2 = min(self.doc.img_h - 1, y1 + h)
            
            # 获取用户在属性面板中选择的类型，如果没有选择则使用默认值
            base_type = self.prop.type_combo.currentText() or "新房间"
            # 确保基础类型有效并进行清理
            if not base_type.strip():
                base_type = "新房间"
            
            # 清理和验证房间类型名称
            base_type = base_type.strip()
            # 移除可能导致问题的特殊字符
            import re
            base_type = re.sub(r'[^\w\u4e00-\u9fff]', '', base_type)  # 只保留字母数字和中文
            if not base_type:
                base_type = "新房间"
            
            # 限制房间类型名称长度
            if len(base_type) > 10:
                base_type = base_type[:10]
                
            print(f"Adding room with base_type: '{base_type}'")
                
            label_id = self.doc.assign_label_for_category(base_type)
            
            # 生成序号：找到该基础类型的下一个序号
            existing_indices = []
            for r in self.doc.rooms:
                if r.type == base_type:  # 现在type就是基础类型
                    existing_indices.append(r.index)
            
            next_index = max(existing_indices, default=0) + 1
            # 生成显示名称用于ID
            display_name = f"{base_type}{next_index}"
            stable_id = f"{display_name}_{next_index}"
            
            print(f"Creating room: type={base_type}, index={next_index}, id={stable_id}")
            
            new_room = RoomModel(
                id=stable_id,
                type=base_type,  # 基础类型
                label_id=label_id,
                bbox=(x1, y1, x2, y2),
                text_raw="",
                confidence=0.0,
                source="manual",
                edited=True,
            )
            new_room.index = next_index  # 设置索引
            new_room.recompute(self.doc.img_w, self.doc.img_h)
            
            # 使用列表来存储房间引用，这样可以在redo函数中修改
            current_room = [new_room]  # 用列表包装以便在内部函数中修改
            
            def redo():
                print(f"Starting redo: adding room to document")
                self.doc.rooms.append(current_room[0])
                print(f"Room added to document, total rooms: {len(self.doc.rooms)}")
                
                # 确保新的房间类型被添加到categories中
                if base_type not in self.doc.categories:
                    print(f"Adding new category: {base_type}")
                    self.doc.categories.append(base_type)
                
                print(f"Calling ensure_unified_room_naming...")
                # 使用统一编号方法而不是简单的ensure_indices
                self.doc.ensure_unified_room_naming()
                print(f"Unified room naming ensured")
                
                # 由于ensure_unified_room_naming可能会更改房间ID，我们需要重新找到这个房间
                # 根据位置和类型找到刚添加的房间
                target_room = None
                for room in self.doc.rooms:
                    if (room.type == base_type and 
                        room.bbox == current_room[0].bbox and 
                        room.source == "manual"):
                        target_room = room
                        break
                
                if target_room:
                    current_room[0] = target_room  # 更新引用
                    print(f"Found updated room: {current_room[0].id}")
                else:
                    print(f"Warning: Could not find updated room")
                
                print(f"Calling view.load_document...")
                # 重新加载整个文档以确保句柄正确初始化
                try:
                    self.view.load_document(self.doc)
                    print(f"Document loaded to view successfully")
                except Exception as load_err:
                    print(f"ERROR in view.load_document: {load_err}")
                    import traceback
                    traceback.print_exc()
                    return
                
                # 更新属性面板的categories
                print(f"Updating property panel categories...")
                try:
                    self.prop.load_categories(self.doc.categories)
                    print(f"Property panel updated successfully")
                except Exception as prop_err:
                    print(f"ERROR in prop.load_categories: {prop_err}")
                    import traceback
                    traceback.print_exc()
                    return
                
                # 自动选中新创建的房间
                print(f"Auto-selecting new room: {current_room[0].id}")
                try:
                    self.prop.show_room(current_room[0])
                    print(f"New room selected in property panel successfully")
                except Exception as select_err:
                    print(f"ERROR in prop.show_room: {select_err}")
                    import traceback
                    traceback.print_exc()
                
            def undo():
                print(f"Starting undo: removing room from document")
                self.doc.rooms.remove(current_room[0])
                self.view.load_document(self.doc)
                self.prop.load_categories(self.doc.categories)
                # 清空选择
                self.prop.show_room(None)
                print(f"Room removed and view updated")
                
            redo()
            self.undo.push(UndoCommand(redo, undo, "add"))
            print(f"Room added successfully: {display_name}")
            
        except Exception as e:
            print(f"ERROR in add_room_default: {e}")
            import traceback
            traceback.print_exc()

    # ---------- 全局字段 ----------
    def on_global_field_edited(self, field: str, value):
        if not self.doc:
            return
        old_orientation = self.doc.house_orientation
        old_angle = self.doc.north_angle
        def redo():
            if field == "house_orientation":
                self.doc.house_orientation = str(value)
            elif field == "north_angle":
                try:
                    self.doc.north_angle = int(value) % 360
                except Exception:
                    pass
            self.prop.set_global_fields(self.doc.house_orientation, self.doc.north_angle)
            self.view.viewport().update()  # 刷新指北针
        def undo():
            self.doc.house_orientation = old_orientation
            self.doc.north_angle = old_angle
            self.prop.set_global_fields(self.doc.house_orientation, self.doc.north_angle)
            self.view.viewport().update()
        redo(); self.undo.push(UndoCommand(redo, undo, field))

    # ---------- 其它 ----------
    def toggle_mask(self, _):
        self.view.toggle_mask(not self.view._show_mask)

    def on_undo(self):
        self.undo.undo();
        if self.doc: 
            self.doc.ensure_unified_room_naming(); 
            self.view.load_document(self.doc)

    def on_redo(self):
        self.undo.redo();
        if self.doc: 
            self.doc.ensure_unified_room_naming(); 
            self.view.load_document(self.doc)

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
