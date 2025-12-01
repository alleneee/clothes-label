"""Pants labeling service with brand-specific selection rules."""
import json
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path

from pydantic import BaseModel, model_validator


class LabelResult(BaseModel):
    """Single labeling candidate coming from detection service."""
    brand: str
    pic_id: int
    type: str
    size: int
    confidence: float
    pic_name: str
    type_code: Optional[str] = None
    product_code: str


class LabelingRequest(BaseModel):
    """API request schema, compatible with both list and single payloads."""
    results: List[LabelResult]

    @model_validator(mode='before')
    @classmethod
    def normalize_payload(cls, values):
        # Support {"result": {...}} or {"results": {...}} legacy formats
        if "results" not in values and "result" in values:
            result_value = values.pop("result")
            if isinstance(result_value, list):
                values["results"] = result_value
            else:
                values["results"] = [result_value]
        elif isinstance(values.get("results"), dict):
            values["results"] = [values["results"]]
        return values


class LabeledImage(BaseModel):
    product_code: str
    tag_first_type: str
    tag_second_type: str
    tag_result: Optional[str] = None
    pic_id: int
    type: str
    type_code: Optional[str] = None
    size: int
    confidence: float
    pic_name: str
    new_file_name: str
    return_content_name: Optional[str] = None
    original_file_name: Optional[str] = None
    return_content_type: str = "原图"


class LabelingResponse(BaseModel):
    selected: List[LabeledImage]


pants_type_to_code = {
    "正面模特": "01-model",
    "正面平铺": "01",
    "背面模特": "02-model",
    "背面平铺":"02",
    "全身模特": "01-model-all",
    "口袋特写":"03",
    "裤脚特写":"04",
    "腰部特写":"05",
    "商标特写":"06",
    "其他细节":"07",
}


class PantsLabelingService:
    """Encapsulates brand-specific pants labeling logic."""

    MAX_SELECTION = 6

    def select_images(self, items: List[LabelResult]) -> List[Dict[str, Any]]:
        if not items:
            return []

        brand = self._normalize_text(items[0].brand).upper()
        if any(self._normalize_text(item.brand).upper() != brand for item in items):
            raise ValueError("请求中包含多个品牌，请拆分后再调用接口")

        ordered: List[Dict[str, Any]]
        if brand == "AD":
            ordered = self._select_ad_brand_images(items)
        elif brand == "NK":
            ordered = self._select_nk_brand_images(items)
        elif brand == "LN":
            ordered = self._select_ln_brand_images(items)
        else:
            raise ValueError("不支持当前品牌")

        # 为每个item填充type_code
        self._assign_type_codes(ordered)
        self._assign_new_file_names(brand, ordered)
        return ordered

    def _select_ad_brand_images(self, items: List[LabelResult]) -> List[Dict[str, Any]]:
        """AD品牌选图:
        1. 正面模特 -> 01 (01-model)
        2. 正面平铺 -> 02 (01)
        3. 背面模特 -> 03 (02-model)
        4. 其他细节 -> 04, 05 (口袋/裤脚/腰部/商标特写等)
        5. 全身模特 -> 06 (01-model-all)
        """
        grouped = self._group_by_type(items)
        used_ids: Set[int] = set()
        ordered: List[Dict[str, Any]] = []

        def append_item(candidate: Optional[LabelResult], rule_desc: str):
            if candidate:
                ordered.append({"item": candidate, "rule": rule_desc})

        # 位置1: 正面模特
        append_item(self._pick_best_of_type(grouped, "正面模特", used_ids), "AD规则1: 正面模特")
        
        # 位置2: 正面平铺
        append_item(self._pick_best_of_type(grouped, "正面平铺", used_ids), "AD规则2: 正面平铺")
        
        # 位置3: 背面模特
        append_item(self._pick_best_of_type(grouped, "背面模特", used_ids), "AD规则3: 背面模特")

        # 位置4-5: 其他细节（排除全身模特）
        pool = [item for item in items if item.pic_id not in used_ids]
        detail_candidates = self._pick_from_pool(pool, used_ids, limit=2, exclude_types={"全身模特"})
        for candidate in detail_candidates:
            ordered.append({"item": candidate, "rule": "AD规则4: 其他细节"})
        
        # 位置6: 优先全身模特，如果没有则用其他细节补充
        full_body = self._pick_best_of_type(grouped, "全身模特", used_ids)
        if full_body:
            ordered.append({"item": full_body, "rule": "AD规则5: 全身模特"})
        else:
            # 如果没有全身模特，从剩余图片中选一张
            remaining_pool = [item for item in items if item.pic_id not in used_ids]
            fallback = self._pick_from_pool(remaining_pool, used_ids, limit=1)
            if fallback:
                ordered.append({"item": fallback[0], "rule": "AD规则6: 其他细节补充"})

        self._fill_remaining(items, used_ids, ordered)
        return ordered

    def _select_nk_brand_images(self, items: List[LabelResult]) -> List[Dict[str, Any]]:
        """NK品牌选图:
        1. 正面模特 -> 01 (01-model)
        2. 背面模特 -> 02 (02-model)
        3. 其他细节 -> 03, 04, 05 (口袋/裤脚/腰部/商标特写等)
        4. 全身模特 -> 06 (01-model-all)
        """
        grouped = self._group_by_type(items)
        used_ids: Set[int] = set()
        ordered: List[Dict[str, Any]] = []

        def append_item(candidate: Optional[LabelResult], rule_desc: str):
            if candidate:
                ordered.append({"item": candidate, "rule": rule_desc})

        # 位置1: 正面模特
        append_item(self._pick_best_of_type(grouped, "正面模特", used_ids), "NK规则1: 正面模特")
        
        # 位置2: 背面模特
        append_item(self._pick_best_of_type(grouped, "背面模特", used_ids), "NK规则2: 背面模特")

        # 位置3-5: 其他细节（排除全身模特）
        pool = [item for item in items if item.pic_id not in used_ids]
        detail_candidates = self._pick_from_pool(pool, used_ids, limit=3, exclude_types={"全身模特"})
        for candidate in detail_candidates:
            ordered.append({"item": candidate, "rule": "NK规则3: 其他细节"})
        
        # 位置6: 优先全身模特，如果没有则用其他细节补充
        full_body = self._pick_best_of_type(grouped, "全身模特", used_ids)
        if full_body:
            ordered.append({"item": full_body, "rule": "NK规则4: 全身模特"})
        else:
            # 如果没有全身模特，从剩余图片中选一张
            remaining_pool = [item for item in items if item.pic_id not in used_ids]
            fallback = self._pick_from_pool(remaining_pool, used_ids, limit=1)
            if fallback:
                ordered.append({"item": fallback[0], "rule": "NK规则5: 其他细节补充"})

        self._fill_remaining(items, used_ids, ordered)
        return ordered

    def _select_ln_brand_images(self, items: List[LabelResult]) -> List[Dict[str, Any]]:
        """LN品牌选图：

        只保留文件名中带WHITE的图片，WHITE之后的数字为命名序号，保留序号1-6。
        示例命名: AKLV217-5-MXK-WHITE-1.jpg
        """
        ordered: List[Dict[str, Any]] = []
        
        # 提取WHITE图片及其序号
        white_items: List[Tuple[int, LabelResult]] = []
        for item in items:
            seq = self._extract_ln_white_sequence(item.pic_name)
            if seq is not None and 1 <= seq <= 6:
                white_items.append((seq, item))
        
        # 按序号排序，去重（同一序号取第一个）
        white_items.sort(key=lambda x: x[0])
        seen_seq: Set[int] = set()
        for seq, item in white_items:
            if seq not in seen_seq:
                seen_seq.add(seq)
                ordered.append({"item": item, "rule": f"LN规则: WHITE序号{seq}"})
        
        return ordered

    def _extract_ln_white_sequence(self, pic_name: str) -> Optional[int]:
        """从文件名中提取WHITE后面的序号。
        
        示例: AKLV217-5-MXK-WHITE-1.jpg -> 1
        """
        # 匹配 WHITE 后面跟着的数字（可能有分隔符如 - 或 _）
        match = re.search(r'WHITE[-_]?(\d+)', pic_name, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def _select_default_images(self, items: List[LabelResult]) -> List[Dict[str, Any]]:
        ordered: List[Dict[str, Any]] = []
        for candidate in sorted(items, key=lambda x: x.size, reverse=True)[: self.MAX_SELECTION]:
            ordered.append({"item": candidate, "rule": "默认规则: 按尺寸排序"})
        return ordered

    # --- Shared helpers ------------------------------------------------------

    @staticmethod
    def _normalize_text(value: str) -> str:
        return value.strip() if isinstance(value, str) else value
    
    def _assign_type_codes(self, ordered: List[Dict[str, Any]]) -> None:
        """根据pants_type_to_code映射为每个item填充type_code字段"""
        for record in ordered:
            item = record["item"]
            item_type = self._normalize_text(item.type)
            # 从映射中获取type_code，如果没有则保持为None
            item.type_code = pants_type_to_code.get(item_type)

    def _group_by_type(self, items: List[LabelResult]) -> Dict[str, List[LabelResult]]:
        grouped: Dict[str, List[LabelResult]] = {}
        for item in items:
            item_type = self._normalize_text(item.type)
            grouped.setdefault(item_type, []).append(item)
        for item_type in grouped:
            grouped[item_type].sort(key=lambda x: x.size, reverse=True)
        return grouped

    def _pick_best_of_type(
        self, grouped: Dict[str, List[LabelResult]], target_type: str, used_ids: Set[int]
    ) -> Optional[LabelResult]:
        target_list = grouped.get(target_type)
        if not target_list:
            return None
        for candidate in target_list:
            if candidate.pic_id in used_ids:
                continue
            used_ids.add(candidate.pic_id)
            return candidate
        return None

    def _pick_from_pool(
        self,
        pool: List[LabelResult],
        used_ids: Set[int],
        limit: int,
        exclude_types: Optional[Set[str]] = None,
    ) -> List[LabelResult]:
        exclude_types = {self._normalize_text(t) for t in (exclude_types or set())}
        selected: List[LabelResult] = []
        for candidate in sorted(pool, key=lambda x: x.size, reverse=True):
            normalized_type = self._normalize_text(candidate.type)
            if candidate.pic_id in used_ids or normalized_type in exclude_types:
                continue
            used_ids.add(candidate.pic_id)
            selected.append(candidate)
            if len(selected) >= limit:
                break
        return selected

    def _fill_remaining(
        self,
        items: List[LabelResult],
        used_ids: Set[int],
        ordered: List[Dict[str, Any]],
    ) -> None:
        remaining_needed = self.MAX_SELECTION - len(ordered)
        if remaining_needed <= 0:
            return
        remaining_pool = [item for item in items if item.pic_id not in used_ids]
        fallback_candidates = self._pick_from_pool(remaining_pool, used_ids, limit=remaining_needed)
        for candidate in fallback_candidates:
            ordered.append({"item": candidate, "rule": "规则6: 兜底选择"})

    def _assign_new_file_names(self, brand: str, ordered: List[Dict[str, Any]]) -> None:
        """按照品牌规则，为每条记录分配 brand_XX 的导出文件名。"""
        brand_code = brand.upper()
        if brand_code == "AD":
            self._assign_ad_filenames(brand_code, ordered)
        elif brand_code == "NK":
            self._assign_nk_filenames(brand_code, ordered)
        else:
            self._assign_sequential_filenames(brand_code, ordered)

    def _assign_ad_filenames(self, brand: str, ordered: List[Dict[str, Any]]) -> None:
        """AD品牌文件名：01正面模特、02正面平铺、03背面模特、04-05为其他、06兜底。"""
        front_assigned = False
        flat_assigned = False
        back_assigned = False
        other_codes = iter(["04", "05"])
        for record in ordered:
            item_type = self._normalize_text(record["item"].type)
            code = None

            if not front_assigned and item_type == "正面模特":
                code = "01"
                front_assigned = True
            elif not flat_assigned and item_type == "正面平铺":
                code = "02"
                flat_assigned = True
            elif not back_assigned and item_type == "背面模特":
                code = "03"
                back_assigned = True
            elif item_type != "全身模特":
                code = next(other_codes, None)

            if code is None:
                code = "06"

            record["new_file_name"] = self._compose_new_file_path(record["item"].pic_name, str(record["item"].product_code), code)

    def _assign_nk_filenames(self, brand: str, ordered: List[Dict[str, Any]]) -> None:
        """NK品牌文件名：01正面模特、02背面模特、03-05其他（排除全身）、06兜底。"""
        front_assigned = False
        back_assigned = False
        other_codes = iter(["03", "04", "05"])
        for record in ordered:
            item_type = self._normalize_text(record["item"].type)
            code = None

            if not front_assigned and item_type == "正面模特":
                code = "01"
                front_assigned = True
            elif not back_assigned and item_type == "背面模特":
                code = "02"
                back_assigned = True
            elif item_type != "全身模特":
                code = next(other_codes, None)

            if code is None:
                code = "06"

            record["new_file_name"] = self._compose_new_file_path(record["item"].pic_name, str(record["item"].product_code), code)

    def _assign_sequential_filenames(self, brand: str, ordered: List[Dict[str, Any]]) -> None:
        """默认文件名：brand_01 ~ brand_06 顺序编号。"""
        for idx, record in enumerate(ordered, start=1):
            record["new_file_name"] = self._compose_new_file_path(record["item"].pic_name, str(record["item"].product_code), f"{idx:02d}")

    def _compose_new_file_path(self, original_path: str, product_code: str, code: str) -> str:
        """Switch basename to product_code_code while preserving directory + suffix."""
        try:
            path = Path(original_path)
            suffix = path.suffix or ""
            new_name = f"{product_code}_{code}{suffix}"
            if path.parent == Path('.'):
                return new_name
            return str(path.with_name(new_name))
        except Exception:
            return f"{product_code}_{code}"

if __name__ == "__main__":
    service = PantsLabelingService()
    # 测试NK品牌选图
    print("=" * 60)
    print("测试NK品牌选图")
    print("=" * 60)
    nk_items = [
        LabelResult(brand="NK", pic_id=1, type="正面模特", size=100, confidence=0.9, pic_name="front_model.jpg", product_code="12345"),
        LabelResult(brand="NK", pic_id=2, type="背面模特", size=100, confidence=0.9, pic_name="back_model.jpg", product_code="12345"),
        LabelResult(brand="NK", pic_id=3, type="全身模特", size=100, confidence=0.9, pic_name="full_body.jpg", product_code="12345"),
        LabelResult(brand="NK", pic_id=4, type="口袋特写", size=100, confidence=0.9, pic_name="pocket.jpg", product_code="12345"),
        LabelResult(brand="NK", pic_id=5, type="裤脚特写", size=100, confidence=0.9, pic_name="crotch.jpg", product_code="12345"),
        LabelResult(brand="NK", pic_id=6, type="腰部特写", size=100, confidence=0.9, pic_name="waist.jpg", product_code="12345"),
        LabelResult(brand="NK", pic_id=7, type="商标特写", size=100, confidence=0.9, pic_name="logo.jpg", product_code="12345"),
        LabelResult(brand="NK", pic_id=8, type="其他细节", size=100, confidence=0.9, pic_name="other.jpg", product_code="12345"),
    ]
    nk_result = service.select_images(nk_items)
    # 转换为可JSON序列化的格式
    nk_json = []
    for record in nk_result:
        item = record['item']
        nk_json.append({
            'type': item.type,
            'type_code': item.type_code,
            'product_code': item.product_code,
            'pic_id': item.pic_id,
            'pic_name': item.pic_name,
            'size': item.size,
            'confidence': item.confidence,
            'rule': record['rule'],
            'new_file_name': record['new_file_name']
        })
    print(json.dumps(nk_json, indent=2, ensure_ascii=False))  
    # for idx, record in enumerate(nk_result, 1):
    #     item = record['item']
    #     print(f"{idx}. {item.type} (type_code: {item.type_code}) - {record['rule']} - {record['new_file_name']}")
    
    print("\n" + "=" * 60)
    print("测试AD品牌选图")
    print("=" * 60)
    ad_items = [
        LabelResult(brand="AD", pic_id=1, type="正面模特", size=100, confidence=0.9, pic_name="front_model.jpg", product_code="67890"),
        LabelResult(brand="AD", pic_id=2, type="正面平铺", size=100, confidence=0.9, pic_name="front_flat.jpg", product_code="67890"),
        LabelResult(brand="AD", pic_id=3, type="背面模特", size=100, confidence=0.9, pic_name="back_model.jpg", product_code="67890"),
        LabelResult(brand="AD", pic_id=4, type="全身模特", size=100, confidence=0.9, pic_name="full_body.jpg", product_code="67890"),
        LabelResult(brand="AD", pic_id=5, type="口袋特写", size=100, confidence=0.9, pic_name="pocket.jpg", product_code="67890"),
        LabelResult(brand="AD", pic_id=6, type="裤脚特写", size=100, confidence=0.9, pic_name="crotch.jpg", product_code="67890"),
        LabelResult(brand="AD", pic_id=7, type="腰部特写", size=100, confidence=0.9, pic_name="waist.jpg", product_code="67890"),
        LabelResult(brand="AD", pic_id=8, type="商标特写", size=100, confidence=0.9, pic_name="logo.jpg", product_code="67890"),
    ]
    ad_result = service.select_images(ad_items)
    # 转换为可JSON序列化的格式
    ad_json = []
    for record in ad_result:
        item = record['item']
        ad_json.append({
            'type': item.type,
            'type_code': item.type_code,
            'product_code': item.product_code,
            'pic_id': item.pic_id,
            'pic_name': item.pic_name,
            'size': item.size,
            'confidence': item.confidence,
            'rule': record['rule'],
            'new_file_name': record['new_file_name']
        })
    print(json.dumps(ad_json, indent=2, ensure_ascii=False))
    # for idx, record in enumerate(ad_result, 1):
    #     item = record['item']
    #     print(f"{idx}. {item.type} (type_code: {item.type_code}) - {record['rule']} - {record['new_file_name']}")

    print("\n" + "=" * 60)
    print("测试LN品牌选图 - WHITE文件名过滤")
    print("=" * 60)
    # 使用真实示例文件名
    ln_filenames = [
        "AKLV217-5-MXK-DIAOPAI-1.jpg",
        "AKLV217-5-MXK-DIAOPAI-2.jpg",
        "AKLV217-5-MXK-DIAOPAI-3.jpg",
        "AKLV217-5-MXK-MODEL-1(PNG).jpg",
        "AKLV217-5-MXK-SELL-1.jpg",
        "AKLV217-5-MXK-SELL-2.jpg",
        "AKLV217-5-MXK-SELL-3.jpg",
        "AKLV217-5-MXK-SELL-4.jpg",
        "AKLV217-5-MXK-SHUIXIBIAO-1.jpg",
        "AKLV217-5-MXK-WHITE-1.jpg",
        "AKLV217-5-MXK-WHITE-10.jpg",
        "AKLV217-5-MXK-WHITE-11.jpg",
        "AKLV217-5-MXK-WHITE-14.jpg",
        "AKLV217-5-MXK-WHITE-15.jpg",
        "AKLV217-5-MXK-WHITE-16(800).jpg",
        "AKLV217-5-MXK-WHITE-16(G800).jpg",
        "AKLV217-5-MXK-WHITE-2.jpg",
        "AKLV217-5-MXK-WHITE-3.jpg",
        "AKLV217-5-MXK-WHITE-4.jpg",
        "AKLV217-5-MXK-WHITE-5.jpg",
        "AKLV217-5-MXK-WHITE-6.jpg",
        "AKLV217-5-MXK-WHITE-7.jpg",
        "AKLV217-5-MXK-WHITE-8.jpg",
        "AKLV217-5-MXK-WHITE-9.jpg",
        "AKLV217-5-SIZE.jpg",
    ]
    ln_items = [
        LabelResult(
            brand="LN", 
            pic_id=idx, 
            type="其他", 
            size=100, 
            confidence=0.9, 
            pic_name=fname, 
            product_code="AKLV217-5-MXK"
        )
        for idx, fname in enumerate(ln_filenames, start=1)
    ]
    ln_result = service.select_images(ln_items)
    ln_json = []
    for record in ln_result:
        item = record['item']
        ln_json.append({
            'pic_name': item.pic_name,
            'rule': record['rule'],
            'new_file_name': record['new_file_name']
        })
    print(json.dumps(ln_json, indent=2, ensure_ascii=False))
