import numpy as np
import cv2

class ReasonablenessValidator:
    """第四层：合理性验证器"""
    def __init__(self):
        self.spatial_rules = SpatialRuleEngine()
        self.size_constraints = SizeConstraintEngine()
        self.boundary_detector = BuildingBoundaryDetector()

    def validate_and_correct(self, fused_results, ocr_results, original_size):
        print("🔍 [第4层-合理性验证器] 开始合理性验证...")
        validated_results = self.spatial_rules.validate_spatial_logic(fused_results, ocr_results)
        validated_results = self.size_constraints.validate_size_constraints(validated_results, original_size)
        validated_results = self.boundary_detector.validate_building_boundary(validated_results, original_size)
        validated_results = self._check_geometry_regularization(validated_results)
        print("✅ [第4层-合理性验证器] 合理性验证完成")
        return validated_results

    def _check_geometry_regularization(self, results, ratio_threshold: float = 0.6):
        print("   📐 [几何正则] 检查房间形状...")
        base = results.copy()  # 保留原始标签，防止覆盖其他房间
        corrected = results.copy()
        unique_labels = [l for l in np.unique(results) if l not in {0, 9, 10}]
        for lbl in unique_labels:
            mask = (base == lbl).astype(np.uint8)
            num_c, labeled, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for cid in range(1, num_c):
                region_mask = (labeled == cid).astype(np.uint8)
                orig_area = int(region_mask.sum())
                # 忽略过小组件，避免噪声放大
                if orig_area < 100:  # 面积阈值，可调
                    continue
                contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                cnt = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(cnt)
                w, h = rect[1]
                if w == 0 or h == 0:
                    continue
                compact_ratio = cv2.contourArea(cnt) / (w * h)
                # 形状已经比较紧凑，无需正则
                if compact_ratio >= ratio_threshold:
                    continue
                # 先尝试闭运算
                kernel = np.ones((3, 3), np.uint8)
                grown = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
                contours2, _ = cv2.findContours(grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                new_mask = None
                if contours2:
                    cnt2 = max(contours2, key=cv2.contourArea)
                    rect2 = cv2.minAreaRect(cnt2)
                    w2, h2 = rect2[1]
                    if w2 > 0 and h2 > 0:
                        compact_ratio2 = cv2.contourArea(cnt2) / (w2 * h2)
                        if compact_ratio2 >= ratio_threshold:
                            new_mask = np.zeros_like(region_mask)
                            cv2.drawContours(new_mask, [cnt2], -1, 1, -1)
                if new_mask is None:
                    # 使用凸包，但禁止跨越其它房间：只填充原 label 或背景
                    hull = cv2.convexHull(cnt)
                    hull_mask = np.zeros_like(region_mask)
                    cv2.drawContours(hull_mask, [hull], 0, 1, -1)
                    new_mask = hull_mask
                    method = "凸包填充"
                else:
                    method = "区域生长"
                new_area = int(new_mask.sum())
                # 防止面积暴增（例如客厅吞并其他房间）
                if new_area > orig_area * 2.0:
                    print(f"   ⚠️ [几何正则] 跳过房间{lbl}-{cid} 过度扩张: {orig_area}->{new_area}")
                    continue
                # 仅允许写入原区域或背景(0)，保留其他房间/墙体
                allowed_region = ((base == 0) | (base == lbl))
                write_mask = (new_mask == 1) & allowed_region
                # 先清除原区域（仅该连通域内）
                corrected[region_mask == 1] = 0
                corrected[write_mask] = lbl
                print(f"   🔧 [几何正则] 房间{lbl}-{cid} ({method}) 面积: {orig_area} -> {write_mask.sum()}")
        return corrected

class SpatialRuleEngine:
    """空间逻辑规则引擎"""
    def validate_spatial_logic(self, results, ocr_results):
        print("🧠 [空间规则引擎] 验证空间逻辑...")
        results=self._check_nested_rooms(results, ocr_results)
        results=self._check_room_overlap(results, ocr_results)
        results=self._check_kitchen_position(results, ocr_results)
        return results

    def _check_nested_rooms(self, results, ocr_results):
        print("   🏠 [空间规则引擎] 检查嵌套房间...")
        ocr_room_regions={}
        for item in ocr_results:
            text=item["text"].lower().strip()
            if any(k in text for k in ["卧室","bedroom"]):
                x,y,w,h=item["bbox"]
                sx=512.0/(item.get('ocr_width',1158)); sy=512.0/(item.get('ocr_height',866))
                x512=int(x*sx); y512=int(y*sy); w512=int(w*sx); h512=int(h*sy)
                region={'x1':max(0,x512-w512),'y1':max(0,y512-h512),'x2':min(512,x512+w512+w512),'y2':min(512,y512+h512+h512),'text':text,'center_x':x512+w512//2,'center_y':y512+h512//2}
                ocr_room_regions[text]=region
        bedroom_mask=(results==4).astype(np.uint8)
        num_labels,labels_im,stats,centroids=cv2.connectedComponentsWithStats(bedroom_mask,connectivity=4)
        for room_name,region in ocr_room_regions.items():
            nested=[]
            for comp_id in range(1,num_labels):
                cx,cy=centroids[comp_id]; area=stats[comp_id,cv2.CC_STAT_AREA]
                if region['x1']<=cx<=region['x2'] and region['y1']<=cy<=region['y2']:
                    nested.append(comp_id)
            if len(nested)>1:
                largest=max(nested,key=lambda c:stats[c,cv2.CC_STAT_AREA])
                for comp_id in nested:
                    if comp_id!=largest: results[labels_im==comp_id]=0
        return results

    def _check_room_overlap(self, results, ocr_results):
        print("   🔍 [空间规则引擎] 检查房间重叠冲突...")
        ocr_rooms={}; room_type_map={"厨房":7,"kitchen":7,"卫生间":2,"bathroom":2,"washroom":2,"客厅":3,"living":3,"卧室":4,"bedroom":4,"阳台":6,"balcony":6,"书房":8,"study":8}
        for item in ocr_results:
            text=item['text'].lower().strip(); label=None
            for k,l in room_type_map.items():
                if k in text: label=l; break
            if label:
                x,y,w,h=item['bbox']; sx=512.0/(item.get('ocr_width',1158)); sy=512.0/(item.get('ocr_height',866))
                cx=int((x+w//2)*sx); cy=int((y+h//2)*sy)
                ocr_rooms.setdefault(label,[]).append({'center':(cx,cy),'text':text,'confidence':item.get('confidence',1.0)})
        room_labels=[2,3,4,6,7,8]
        for label in room_labels:
            mask=(results==label)
            if not np.any(mask): continue
            if label in ocr_rooms and ocr_rooms[label]: continue
            area=np.sum(mask); total=results.shape[0]*results.shape[1]; ratio=area/total
            if ratio>0.08: results[mask]=0
        results=self._check_room_overlap_conflicts(results)
        return results

    def _check_kitchen_position(self, results, ocr_results):
        return results

    def _check_room_overlap_conflicts(self, results):
        room_labels=[2,3,4,6,7,8]; names={2:"卫生间",3:"客厅",4:"卧室",6:"阳台",7:"厨房",8:"书房"}
        for label in room_labels:
            mask=(results==label).astype(np.uint8)
            if not np.any(mask): continue
            num,lab,stats,cent=cv2.connectedComponentsWithStats(mask,connectivity=4)
            for comp_id in range(1,num):
                area=stats[comp_id,cv2.CC_STAT_AREA]; total=results.shape[0]*results.shape[1]; ratio=area/total
                if ratio>0.15:
                    component=(lab==comp_id); overlap_count=0
                    for other in room_labels:
                        if other==label: continue
                        other_mask=(results==other)
                        if not np.any(other_mask): continue
                        overlap=np.sum(component & other_mask); overlap_ratio=overlap/area if area>0 else 0
                        if overlap_ratio>0.1: overlap_count+=1
                    if overlap_count>=2 or (overlap_count==1 and ratio>0.25): results[component]=0
        return results

class SizeConstraintEngine:
    """尺寸约束引擎"""
    def validate_size_constraints(self, results, original_size):
        pixel_to_meter=12.0/original_size[0]
        results=self._validate_large_area_rooms(results)
        for room_label in [2,7]:
            room_name={2:"卫生间",7:"厨房"}[room_label]
            results=self._check_room_size(results, room_label, room_name, pixel_to_meter)
        return results

    def _check_room_size(self, results, room_label, room_name, pixel_to_meter):
        mask=(results==room_label).astype(np.uint8); num,lab,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=4)
        if room_label==2: min_a,max_a=2,15
        elif room_label==7: min_a,max_a=3,20
        else: return results
        for cid in range(1,num):
            area_px=stats[cid,cv2.CC_STAT_AREA]; area_m2=area_px*(pixel_to_meter**2)
            if area_m2>max_a: results[lab==cid]=0
        return results

    def _validate_large_area_rooms(self, results):
        total=results.shape[0]*results.shape[1]; names={2:"卫生间",3:"客厅",4:"卧室",6:"阳台",7:"厨房",8:"书房"}
        max_ratios={2:0.10,3:0.40,4:0.30,6:0.15,7:0.15,8:0.30}
        for label,_n in names.items():
            mask=(results==label).astype(np.uint8)
            if not np.any(mask): continue
            num,lab,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=4)
            mr=max_ratios.get(label,0.25)
            for cid in range(1,num):
                area=stats[cid,cv2.CC_STAT_AREA]; ratio=area/total
                if ratio>mr: results[lab==cid]=0
        return results

class BuildingBoundaryDetector:
    """建筑边界检测器"""
    def validate_building_boundary(self, results, original_size):
        return self._remove_edge_misidentifications(results)
    def _remove_edge_misidentifications(self, results):
        h,w=results.shape; edge=20; edges=[(0,edge,0,w),(h-edge,h,0,w),(0,h,0,edge),(0,h,w-edge,w)]
        for y1,y2,x1,x2 in edges:
            region=results[y1:y2,x1:x2]; uniq=np.unique(region)
            for label in uniq:
                if label>1:
                    if np.sum(region==label)>50: results[results==label]=0
        return results
