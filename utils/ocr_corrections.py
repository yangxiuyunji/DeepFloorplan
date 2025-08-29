"""OCR text correction utilities.

This module provides a simple interface to load correction rules for OCR
results. Rules can be extended or customized per language by placing a JSON
file named ``ocr_corrections_<lang>.json`` in the same directory. If the JSON
file is missing, built-in defaults are used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

# Built-in fallback rules. These mirror the previous hard-coded mappings in the
# demo. The structure allows easy extension to additional languages in the
# future.
_DEFAULT_RULES: Dict[str, Dict[str, str]] = {
    "zh": {
        # 阳台相关修正
        "阳兮": "阳台",
        "阳台": "阳台",
        "陽台": "阳台",
        "阳合": "阳台",
        "阳舍": "阳台",
        "阳古": "阳台",

        # 厨房相关修正
        "厨房": "厨房",
        "廚房": "厨房",
        "厨户": "厨房",
        "厨庐": "厨房",
        "庁房": "厨房",

        # 卫生间相关修正
        "卫生间": "卫生间",
        "衛生間": "卫生间",
        "卫生闬": "卫生间",
        "卫生门": "卫生间",
        "浴室": "卫生间",
        "洗手间": "卫生间",
        "厕所": "卫生间",

        # 客厅相关修正
        "客厅": "客厅",
        "客廳": "客厅",
        "客应": "客厅",
        "客广": "客厅",
        "起居室": "客厅",
        "会客厅": "客厅",

        # 卧室相关修正
        "卧室": "卧室",
        "臥室": "卧室",
        "卧宝": "卧室",
        "卧窒": "卧室",
        "卧空": "卧室",
        "网房": "卧室",
        "主卧": "主卧",
        "次卧": "次卧",

        # 书房相关修正
        "书房": "书房",
        "書房": "书房",
        "书户": "书房",
        "书庐": "书房",
        "学习室": "书房",
        "工作室": "书房",

        # 餐厅相关修正
        "餐厅": "餐厅",
        "餐廳": "餐厅",
        "饭厅": "餐厅",
        "用餐区": "餐厅",

        # 入户相关修正
        "入户": "入户",
        "玄关": "入户",
        "门厅": "入户",

        # 走廊相关修正
        "走廊": "走廊",
        "过道": "走廊",
        "通道": "走廊",

        # 储物相关修正
        "储物间": "储物间",
        "储藏室": "储物间",
        "杂物间": "储物间",
        "衣帽间": "衣帽间",

        # 清理单字符噪音（常见OCR错误识别）
        "门": "",
        "户": "",
        "口": "",
        "人": "",
        "大": "",
        "小": "",
        "中": "",
        "上": "",
        "下": "",
        "左": "",
        "右": "",
        "一": "",
        "二": "",
        "三": "",
        "四": "",
        "五": "",
        "1": "",
        "2": "",
        "3": "",
        "4": "",
        "5": "",
        "6": "",
        "7": "",
        "8": "",
        "9": "",
        "0": "",
        "m": "",
        "M": "",
        "㎡": "",
        "平": "",
        "方": "",
        "米": "",
    }
}


def load_correction_rules(language: str = "zh") -> Dict[str, str]:
    """Load OCR correction rules for the specified language.

    Parameters
    ----------
    language:
        Language code such as ``"zh"`` or ``"en"``.

    Returns
    -------
    Dict[str, str]
        Mapping of original OCR text to the corrected text.
    """
    json_path = Path(__file__).with_name(f"ocr_corrections_{language}.json")
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return _DEFAULT_RULES.get(language, {}).copy()
