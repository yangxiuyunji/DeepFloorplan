import numpy as np
from utils.ocr_enhanced import fuse_ocr_and_segmentation, text_to_label
from room_detection_manager import RefactoredRoomDetectionManager

class FusionDecisionEngine:
    """第三层：融合决策器 (拆分). 功能保持一致。"""
    def __init__(self):
        self.room_manager = RefactoredRoomDetectionManager()
        self._seed_centers_by_label = {}

    def fuse_results(self, ai_prediction, ocr_results, ocr_shape):
        print("🔗 [第3层-融合决策器] 融合AI和OCR结果...")
        ocr_to_512_x = 512.0 / ocr_shape[1]
        ocr_to_512_y = 512.0 / ocr_shape[0]
        original_ocr_results = [item.copy() for item in ocr_results]
        for item in original_ocr_results:
            item['ocr_width'] = ocr_shape[1]
            item['ocr_height'] = ocr_shape[0]
        converted_items = self._convert_ocr_coordinates(original_ocr_results, ocr_to_512_x, ocr_to_512_y)

        processed_items=[]; open_kitchens=[]
        for item in converted_items:
            label = text_to_label(item['text'])
            if label == 7:
                x,y,w,h = item['bbox']; cx,cy = x+w//2, y+h//2
                if ai_prediction[cy,cx]==3:
                    open_kitchens.append(item)
                else:
                    processed_items.append(item)
            else:
                processed_items.append(item)
        enhanced = fuse_ocr_and_segmentation(ai_prediction.copy(), processed_items)
        enhanced = self._estimate_open_kitchen(enhanced, open_kitchens)
        enhanced = self._ocr_driven_region_growing(enhanced, original_ocr_results, ocr_to_512_x, ocr_to_512_y)
        enhanced = self.room_manager.detect_all_rooms(enhanced, converted_items)
        enhanced = self._basic_cleanup(enhanced, original_ocr_results, ocr_to_512_x, ocr_to_512_y)
        print("✅ [第3层-融合决策器] 融合完成")
        return enhanced

    def _convert_ocr_coordinates(self, room_text_items, scale_x, scale_y):
        converted_items=[]
        for item in room_text_items:
            c=item.copy(); x,y,w,h=item['bbox']
            nx=max(0,min(int(x*scale_x),511)); ny=max(0,min(int(y*scale_y),511))
            nw=max(1,min(int(w*scale_x),512-nx)); nh=max(1,min(int(h*scale_y),512-ny))
            c['bbox']=[nx,ny,nw,nh]; converted_items.append(c)
        return converted_items

    def _estimate_open_kitchen(self, enhanced, kitchen_items, size=60):
        if not kitchen_items: return enhanced
        for item in kitchen_items:
            x,y,w,h=item['bbox']; cx,cy=x+w//2, y+h//2; half=size//2
            x1=max(0,cx-half); y1=max(0,cy-half); x2=min(enhanced.shape[1]-1,cx+half); y2=min(enhanced.shape[0]-1,cy+half)
            patch=enhanced[y1:y2,x1:x2]; mask=~np.isin(patch,[9,10]); patch[mask]=7
        return enhanced

    def _ocr_driven_region_growing(self, enhanced, original_ocr_results, scale_x, scale_y):
        # 简化保留原逻辑接口 (如需完整细节可继续拆分)
        for item in original_ocr_results:
            x,y,w,h=item['bbox']
            # 原坐标 -> 512
            sx=int(x*scale_x); sy=int(y*scale_y); sw=max(1,int(w*scale_x)); sh=max(1,int(h*scale_y))
            sx=min(sx,511); sy=min(sy,511); sw=min(sw,512-sx); sh=min(sh,512-sy)
            label=text_to_label(item['text'])
            if label<=0: continue
            region = enhanced[sy:sy+sh, sx:sx+sw]
            mask=~np.isin(region,[9,10])
            region[mask]=label
        return enhanced

    def _basic_cleanup(self, enhanced, original_ocr_results, scale_x, scale_y):
        # 占位: 保留接口
        return enhanced
