"""Pants labeling service with brand-specific selection rules."""
import json
import logging
import re
from typing import List, Dict, Any, Optional, Set, Tuple

from pathlib import Path

logger = logging.getLogger(__name__)

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

# 品牌类型到编号的映射
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
        logger.info(f"选图结果: {ordered}")
        return ordered

    def _select_ad_brand_images(self, items: List[LabelResult]) -> List[Dict[str, Any]]:
        """AD品牌选图：按优先级依次选取图片完成01-06的选图
        
        优先级顺序（从高到低）：
        1. 正面模特
        2. 正面平铺
        3. 背面模特
        4. 背面平铺
        5. 其他细节（不含全身模特）
        6. 全身模特（06位置优先使用）
        
        选图逻辑：
        - 01-05按优先级从高到低选取
        - 06位置优先使用全身模特，没有则用其他图片补充
        - 同类型内按置信度降序选择
        """
        # 过滤掉pic_name中包含"详情图"的图片
        original_count = len(items)
        filtered_items = [item for item in items if "详情图" not in item.pic_name]
        filtered_count = original_count - len(filtered_items)
        if filtered_count > 0:
            logger.info(f"AD品牌选图: 过滤掉{filtered_count}张详情图，剩余{len(filtered_items)}张")
        items = filtered_items
        
        # 定义类型优先级（数字越小优先级越高）
        type_priority_order = ["正面模特", "正面平铺", "背面模特", "背面平铺"]
        
        # 按类型分组，每组按置信度降序排列
        grouped = self._group_by_type(items)
        
        ordered: List[Dict[str, Any]] = []
        used_ids: Set[int] = set()
        backup_pool: List[LabelResult] = []  # 备选池：同类型的其他图片
        
        # 第一轮：按优先级每种类型选一张置信度最高的
        for target_type in type_priority_order:
            if target_type in grouped:
                type_items = grouped[target_type]
                if type_items:
                    best = type_items[0]
                    if best.pic_id not in used_ids:
                        ordered.append({"item": best, "rule": f"AD规则: {target_type}"})
                        used_ids.add(best.pic_id)
                    # 剩余的放入备选池
                    for item in type_items[1:]:
                        if item.pic_id not in used_ids:
                            backup_pool.append(item)
        
        # 第二轮：其他细节类型（不含全身模特，为06预留）
        for item_type, type_items in grouped.items():
            if item_type not in type_priority_order and item_type != "全身模特":
                for item in type_items:
                    if item.pic_id not in used_ids:
                        if len(ordered) < self.MAX_SELECTION - 1:  # 预留一个位置给全身模特
                            ordered.append({"item": item, "rule": "AD规则: 其他细节"})
                            used_ids.add(item.pic_id)
                        else:
                            backup_pool.append(item)
        
        # 第三轮：06位置优先使用全身模特
        if len(ordered) < self.MAX_SELECTION and "全身模特" in grouped:
            full_body_items = grouped["全身模特"]
            for item in full_body_items:
                if item.pic_id not in used_ids:
                    ordered.append({"item": item, "rule": "AD规则: 全身模特(06)"})
                    used_ids.add(item.pic_id)
                    break  # 只选一张全身模特
        
        # 第四轮：如果还不够6张，从备选池按置信度降序补充
        if len(ordered) < self.MAX_SELECTION:
            backup_pool.sort(key=lambda x: (x.confidence, x.size), reverse=True)
            for item in backup_pool:
                if item.pic_id not in used_ids:
                    ordered.append({"item": item, "rule": "AD规则: 备选补充"})
                    used_ids.add(item.pic_id)
                    if len(ordered) >= self.MAX_SELECTION:
                        break
        
        logger.info(f"AD品牌选图: {ordered}")
        return ordered

    def _select_nk_brand_images(self, items: List[LabelResult]) -> List[Dict[str, Any]]:
        """NK品牌选图：按优先级依次选取图片完成01-06的选图
        
        优先级顺序（从高到低）：
        1. 正面模特
        2. 正面平铺
        3. 背面模特
        4. 背面平铺
        5. 其他细节（不含全身模特）
        6. 全身模特（06位置优先使用）
        
        选图逻辑：
        - 01-05按优先级从高到低选取
        - 06位置优先使用全身模特，没有则用其他图片补充
        - 同类型内按置信度降序选择
        """
        # 过滤掉pic_name中包含"详情图"的图片
        original_count = len(items)
        filtered_items = [item for item in items if "详情图" not in item.pic_name]
        filtered_count = original_count - len(filtered_items)
        if filtered_count > 0:
            logger.info(f"NK品牌选图: 过滤掉{filtered_count}张详情图，剩余{len(filtered_items)}张")
        items = filtered_items
        
        # 定义类型优先级
        type_priority_order = ["正面模特", "正面平铺", "背面模特", "背面平铺"]
        
        # 按类型分组，每组按置信度降序排列
        grouped = self._group_by_type(items)
        
        ordered: List[Dict[str, Any]] = []
        used_ids: Set[int] = set()
        backup_pool: List[LabelResult] = []  # 备选池：同类型的其他图片
        
        # 第一轮：按优先级每种类型选一张置信度最高的
        for target_type in type_priority_order:
            if target_type in grouped:
                type_items = grouped[target_type]
                if type_items:
                    best = type_items[0]
                    if best.pic_id not in used_ids:
                        ordered.append({"item": best, "rule": f"NK规则: {target_type}"})
                        used_ids.add(best.pic_id)
                    # 剩余的放入备选池
                    for item in type_items[1:]:
                        if item.pic_id not in used_ids:
                            backup_pool.append(item)
        
        # 第二轮：其他细节类型（不含全身模特，为06预留）
        for item_type, type_items in grouped.items():
            if item_type not in type_priority_order and item_type != "全身模特":
                for item in type_items:
                    if item.pic_id not in used_ids:
                        if len(ordered) < self.MAX_SELECTION - 1:  # 预留一个位置给全身模特
                            ordered.append({"item": item, "rule": "NK规则: 其他细节"})
                            used_ids.add(item.pic_id)
                        else:
                            backup_pool.append(item)
        
        # 第三轮：06位置优先使用全身模特
        if len(ordered) < self.MAX_SELECTION and "全身模特" in grouped:
            full_body_items = grouped["全身模特"]
            for item in full_body_items:
                if item.pic_id not in used_ids:
                    ordered.append({"item": item, "rule": "NK规则: 全身模特(06)"})
                    used_ids.add(item.pic_id)
                    break  # 只选一张全身模特
        
        # 第四轮：如果还不够6张，从备选池按置信度降序补充
        if len(ordered) < self.MAX_SELECTION:
            backup_pool.sort(key=lambda x: (x.confidence, x.size), reverse=True)
            for item in backup_pool:
                if item.pic_id not in used_ids:
                    ordered.append({"item": item, "rule": "NK规则: 备选补充"})
                    used_ids.add(item.pic_id)
                    if len(ordered) >= self.MAX_SELECTION:
                        break
        
        logger.info(f"NK品牌选图: {ordered}")
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
        """根据pants_type_to_code映射为每个item填充type_code字段
        
        Args:
            ordered: 已排序的打标记录列表，每个记录包含item和rule
            
        功能：
        - 遍历所有选中的图片
        - 根据图片类型查找对应的type_code编码
        - 将编码赋值给LabelResult对象的type_code字段
        - 如果类型没有对应编码，则保持为None
        """
        for record in ordered:
            item = record["item"]
            item_type = self._normalize_text(item.type)
            # 从映射中获取type_code，如果没有则保持为None
            item.type_code = pants_type_to_code.get(item_type)

    def _group_by_type(self, items: List[LabelResult]) -> Dict[str, List[LabelResult]]:
        """将图片按类型分组，并按置信度降序排列
        
        Args:
            items: 待分组的图片列表
            
        Returns:
            Dict[str, List[LabelResult]]: 按类型分组的字典，
                                         key为类型名称，value为该类型图片列表
                                         
        功能：
        - 按图片类型分组（如"正面平铺"、"背面平铺"等）
        - 每组内按置信度降序排列（优先选择分类置信度高的图片）
        - 置信度相同时按尺寸降序排列
        - 便于后续按类型选择最优图片
        """
        grouped: Dict[str, List[LabelResult]] = {}
        for item in items:
            item_type = self._normalize_text(item.type)
            grouped.setdefault(item_type, []).append(item)
        for item_type in grouped:
            # 先按置信度降序，置信度相同时按尺寸降序
            grouped[item_type].sort(key=lambda x: (x.confidence, x.size), reverse=True)
        return grouped

    def _pick_best_of_type(
        self, grouped: Dict[str, List[LabelResult]], target_type: str, used_ids: Set[int]
    ) -> Optional[LabelResult]:
        """
        从指定类型的图片中选择最优的一张
        
        Args:
            grouped: 按类型分组的图片字典 {type: [LabelResult, ...]}
            target_type: 目标类型名称（如"正面平铺"、"背面模特"等）
            used_ids: 已使用的图片ID集合，用于去重
            
        Returns:
            LabelResult: 选中的最优图片（按尺寸降序排列的第一个未被使用的）
            None: 如果该类型不存在或所有图片都已被使用
        """
        # 获取指定类型的图片列表
        target_list = grouped.get(target_type)
        if not target_list:
            return None
        
        # 遍历该类型的图片列表（已按尺寸降序排列）
        for candidate in target_list:
            # 跳过已被使用的图片
            if candidate.pic_id in used_ids:
                continue
            
            # 标记为已使用并返回该图片
            used_ids.add(candidate.pic_id)
            return candidate
        
        # 所有图片都已被使用，返回None
        return None

    def _pick_from_pool(
        self,
        pool: List[LabelResult],
        used_ids: Set[int],
        limit: int,
        exclude_types: Optional[Set[str]] = None,
    ) -> List[LabelResult]:
        """从图片池中选择指定数量的图片
        
        Args:
            pool: 候选图片池（未分组的图片列表）
            used_ids: 已使用的图片ID集合
            limit: 需要选择的图片数量限制
            exclude_types: 需要排除的图片类型集合
            
        Returns:
            List[LabelResult]: 选中的图片列表
            
        功能：
        - 按尺寸降序遍历候选图片
        - 跳过已使用和排除类型的图片
        - 最多选择limit张图片
        - 自动标记已使用的pic_id
        """
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
        """按照品牌规则，为每条记录分配 brand_XX 的导出文件名
        
        Args:
            brand: 品牌名称（如"AD"、"NK"）
            ordered: 已排序的打标记录列表
            
        功能：
        - 根据品牌选择对应的文件名分配策略
        - AD品牌：按优先级排序后依次分配01-06
        - NK品牌：按特定规则分配编号
        - 其他品牌：按顺序分配01-06
        """
        brand_code = brand.upper()
        if brand_code == "AD":
            self._assign_ad_filenames(brand_code, ordered)
        elif brand_code == "NK":
            self._assign_nk_filenames(brand_code, ordered)
        else:
            self._assign_sequential_filenames(brand_code, ordered)

    def _assign_ad_filenames(self, brand: str, ordered: List[Dict[str, Any]]) -> None:
        """AD品牌文件名：按已排序的顺序依次分配01-06
        
        注意：ordered列表已在_select_ad_brand_images中按优先级排好序，
              此处直接按顺序分配编号即可
        """
        for idx, record in enumerate(ordered):
            code = f"{idx + 1:02d}" if idx < 6 else "06"
            record["new_file_name"] = self._compose_new_file_path(
                record["item"].pic_name, 
                str(record["item"].product_code), 
                code
            )

    def _assign_nk_filenames(self, brand: str, ordered: List[Dict[str, Any]]) -> None:
        """NK品牌文件名：按已排序的顺序依次分配01-06
        
        注意：ordered列表已在_select_nk_brand_images中按优先级排好序，
              此处直接按顺序分配编号即可
        """
        for idx, record in enumerate(ordered):
            code = f"{idx + 1:02d}" if idx < 6 else "06"
            record["new_file_name"] = self._compose_new_file_path(
                record["item"].pic_name, 
                str(record["item"].product_code), 
                code
            )

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
