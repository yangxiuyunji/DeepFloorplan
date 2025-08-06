"""OCR utilities for floorplan processing.

This module extracts textual room labels using Tesseract OCR and
provides helper functions to fuse these labels with semantic
segmentation outputs. The OCR step is optional; if the Tesseract
dependency is missing, the functions gracefully return empty results.
"""
from typing import List, Dict, Tuple, Any
try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    np = None  # type: ignore
    _HAS_NUMPY = False

try:  # pragma: no cover - optional dependency
    import pytesseract
    from pytesseract import Output
    _HAS_TESSERACT = True
except Exception:  # pragma: no cover
    pytesseract = None  # type: ignore
    Output = None  # type: ignore
    _HAS_TESSERACT = False

# Mapping from recognized text to room category ids used by the network.
# Both Chinese and English room names are supported.
TEXT_LABEL_MAP = {
    # bedroom
    '卧室': 4, '主卧': 4, '次卧': 4, 'bedroom': 4, 'br': 4,
    # living / dining / kitchen grouped as class 3
    '客厅': 3, 'living': 3, 'livingroom': 3, 'living room': 3,
    '餐厅': 3, 'dining': 3, 'diningroom': 3, 'dining room': 3,
    '厨房': 3, 'kitchen': 3,
    # bathroom / washroom
    '卫生间': 2, '洗手间': 2, '浴室': 2, 'bathroom': 2,
    'washroom': 2, 'toilet': 2,
    # balcony
    '阳台': 6, 'balcony': 6,
    # hall / lobby
    '玄关': 5, '大厅': 5, 'hall': 5, 'lobby': 5,
    # closet / storage
    '衣柜': 1, 'closet': 1
}


def extract_room_text(image: Any) -> List[Dict[str, Tuple[int, int, int, int]]]:
    """Run OCR on the image and return bounding boxes with text.

    Parameters
    ----------
    image: np.ndarray
        Input RGB image as a NumPy array. The image should be in uint8
        format. If Tesseract is not available the function returns an
        empty list.

    Returns
    -------
    List[Dict]
        Each dict contains keys ``text`` and ``bbox`` where ``bbox`` is
        ``(x, y, w, h)`` in pixel coordinates.
    """
    if not _HAS_TESSERACT or not _HAS_NUMPY:
        return []

    data = pytesseract.image_to_data(image, lang='chi_sim+eng', output_type=Output.DICT)
    results: List[Dict[str, Tuple[int, int, int, int]]] = []
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        text = data['text'][i].strip()
        try:
            conf = float(data['conf'][i])
        except Exception:
            conf = -1
        if text and conf > 0:
            x, y = int(data['left'][i]), int(data['top'][i])
            w, h = int(data['width'][i]), int(data['height'][i])
            results.append({'text': text.lower(), 'bbox': (x, y, w, h)})
    return results


def text_to_label(text: str) -> int:
    """Convert recognized text to a room label.

    Returns -1 if the text does not correspond to any known room type.
    """
    return TEXT_LABEL_MAP.get(text.lower(), -1)


def fuse_ocr_and_segmentation(seg: Any, ocr_results: List[Dict]) -> Any:
    """Fuse OCR results with a segmentation map.

    Recognized textual labels take precedence over the existing
    segmentation. For each OCR bounding box whose text maps to a known
    room class, the corresponding region in ``seg`` is overwritten with
    that class label. Boundary labels (9 for door/window, 10 for wall)
    are preserved.

    Parameters
    ----------
    seg: np.ndarray
        2-D array of segmentation labels.
    ocr_results: List[Dict]
        Output from :func:`extract_room_text`.

    Returns
    -------
    np.ndarray
        The fused segmentation map.
    """
    if not _HAS_NUMPY:
        return seg
        
    if seg.ndim != 2:
        raise ValueError('seg must be a 2-D array')

    fused = seg.copy()
    h, w = fused.shape
    for item in ocr_results:
        label = text_to_label(item['text'])
        if label < 0:
            continue
        x, y, bw, bh = item['bbox']
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + bw), min(h, y + bh)
        region = fused[y0:y1, x0:x1]
        if region.size == 0:
            continue
        mask = np.isin(region, [9, 10])
        region[~mask] = label
        fused[y0:y1, x0:x1] = region
    return fused

__all__ = ['extract_room_text', 'fuse_ocr_and_segmentation', 'text_to_label', 'TEXT_LABEL_MAP']
