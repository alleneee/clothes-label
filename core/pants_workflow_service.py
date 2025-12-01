"""裤子处理全流程服务"""
import json
import logging
import re
from typing import List, Dict, Any, Optional, Callable, Tuple
import requests
from io import BytesIO
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class PantsWorkflowService:
    """裤子处理全流程服务：获取图片 -> 分类 -> 打标"""
    
    def __init__(self, oss_api_url: str = "http://10.125.1.6:8123/oss_pic", labeling_service=None):
        self.oss_api_url = oss_api_url
        self.oss_rename_url = "http://10.125.1.6:8123/oss_change_pic_name"
        self.labeling_service = labeling_service
    
    def fetch_image_from_oss(self, pic_name: str) -> Optional[bytes]:
        """从OSS接口获取图片内容"""
        try:
            response = requests.get(
                self.oss_api_url,
                params={"pic_name": pic_name},
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"获取图片失败 {pic_name}: status_code={response.status_code}")
                return None
            
            # 直接返回响应内容作为图片字节
            return response.content
            
        except Exception as e:
            logger.error(f"获取图片异常 {pic_name}: {e}")
            return None
    
    def parse_pic_list(self, pic_list_str: str) -> List[Dict[str, Any]]:
        """解析图片列表字符串，支持转义和非转义的JSON格式"""
        try:
            # 处理可能的转义字符
            pic_list_str = pic_list_str.strip()
            
            # 先尝试直接解析
            try:
                pic_list = json.loads(pic_list_str)
            except json.JSONDecodeError:
                # 如果失败，尝试先解析外层引号（处理转义的情况）
                # 比如: "\"[{...}]\"" -> "[{...}]"
                if pic_list_str.startswith('"') and pic_list_str.endswith('"'):
                    # 去掉外层引号并解析
                    unquoted = pic_list_str[1:-1]
                    pic_list = json.loads(unquoted)
                else:
                    raise
            
            if not isinstance(pic_list, list):
                raise ValueError("pic_list_str必须是JSON数组格式")
            
            logger.info(f"成功解析图片列表，共{len(pic_list)}张图片")
            return pic_list
        except json.JSONDecodeError as e:
            logger.error(f"解析pic_list_str失败: {e}, 原始内容: {pic_list_str[:200]}")
            raise ValueError(f"pic_list_str格式错误: {str(e)}")
    
    def classify_images(
        self, 
        pic_list: List[Dict[str, Any]], 
        predict_func
    ) -> List[Dict[str, Any]]:
        """
        批量分类图片
        
        Args:
            pic_list: 图片列表，每项包含 pic_id 和 pic_name
            predict_func: 预测函数，接收图片字节，返回 (class_name, confidence)
        
        Returns:
            分类结果列表
        """
        results = []
        
        for pic_info in pic_list:
            pic_id = pic_info.get("pic_id")
            pic_name = pic_info.get("pic_name")
            
            if not pic_name:
                logger.warning(f"图片信息缺少pic_name: {pic_info}")
                continue
            
            # 获取图片
            image_bytes = self.fetch_image_from_oss(pic_name)
            if not image_bytes:
                logger.warning(f"无法获取图片: {pic_name}")
                continue
            
            # 预测分类
            try:
                class_name, confidence = predict_func(image_bytes)
                
                results.append({
                    "pic_id": pic_id,
                    "pic_name": pic_name,
                    "type": class_name,
                    "confidence": confidence,
                    "size": len(image_bytes)
                })
                
                logger.info(f"分类完成: {pic_name} -> {class_name} ({confidence:.3f})")
                
            except Exception as e:
                logger.error(f"分类失败 {pic_name}: {e}")
                continue
        
        return results
    
    def process_complete_workflow(
        self,
        brand: str,
        product_code: str,
        pic_list_str: str,
        predict_func: Callable[[bytes], tuple],
        rename_in_oss: bool = True,
        picture_type: str = "pants"
    ) -> List[Dict[str, Any]]:
        """
        完整的裤子处理流程
        
        Args:
            brand: 品牌
            product_code: 产品编码
            pic_list_str: 图片列表JSON字符串
            predict_func: 预测函数，接收图片字节，返回 (class_name, confidence)
            rename_in_oss: 是否在OSS中重命名文件
            picture_type: 图片类型
        
        Returns:
            最终的打标结果列表
        """
        if brand == "LN":
            return self._process_ln_brand(
                product_code=product_code,
                pic_list_str=pic_list_str,
                picture_type=picture_type
            )
        else:    
            # 1. 解析图片列表
            logger.info(f"开始处理品牌{brand}, 货号{product_code}的图片列表")
            pic_list = self.parse_pic_list(pic_list_str)
            
            if not pic_list:
                raise ValueError("图片列表为空")
            
            # 2. 批量分类图片
            logger.info(f"品牌{brand}, 货号{product_code}, 开始批量分类{len(pic_list)}张图片")
            classification_results = self.classify_images(pic_list, predict_func)
            logger.info(f"品牌{brand}, 货号{product_code}, 分类完成: {classification_results}")
            
            if not classification_results:
                raise ValueError("没有成功分类的图片")
            
            # 3. 构建LabelResult列表
            logger.info(f"品牌{brand}, 货号{product_code}, 构建打标结果列表，共{len(classification_results)}个结果")
            from core.pants_labeling_service import LabelResult
            
            label_results = [
                LabelResult(
                    brand=brand,
                    pic_id=item["pic_id"],
                    type=item["type"],
                    size=item["size"],
                    confidence=item["confidence"],
                    pic_name=item["pic_name"],
                    product_code=product_code
                )
                for item in classification_results
            ]
            
            # 4. 调用打标服务
            logger.info(f"品牌{brand}, 货号{product_code}, 开始调用打标服务")
            if not self.labeling_service:
                raise ValueError("打标服务未初始化")
            
            ordered = self.labeling_service.select_images(label_results)
            logger.info(f"品牌{brand}, 货号{product_code}, 打标完成: {ordered}")
            
            # 5. 处理文件名，使用原始product_code
            logger.info(f"品牌{brand}, 货号{product_code}, 开始处理文件名和OSS重命名")
            
            result_list = []
            for idx, record in enumerate(ordered[:6]):
                # 替换文件名中的product_code为原始字符串
                original_filename = record['new_file_name']
                file_path = Path(original_filename)
                suffix = file_path.suffix
                
                # 从文件名中提取code部分（例如：12345_01.jpg -> 01）
                filename_parts = file_path.stem.split('_')
                if len(filename_parts) >= 2:
                    code = filename_parts[-1]
                    new_filename = f"{product_code}_{code}{suffix}"
                else:
                    new_filename = original_filename
                
                # 6. 如果需要，在OSS中重命名文件
                # 生成新的文件路径
                new_path = self.generate_new_file_path(
                    new_file_name=new_filename,
                    brand=brand,
                    product_code=product_code,
                    picture_type=picture_type
                )
                
                # 调用OSS重命名接口
                renamed_file = self.rename_file_in_oss(
                    old_name=record['item'].pic_name,
                    new_name=new_path
                )
                
                # 如果重命名成功，记录日志
                if renamed_file:
                    logger.info(f"品牌{brand}, 货号{product_code}, OSS重命名成功: {record['item'].pic_name} -> {renamed_file}")
                
                result_list.append({
                    "product_code": product_code,
                    "tag_first_type": "1",
                    "tag_second_type": "11",
                    "tag_result": record['item'].type_code,
                    "pic_id": record['item'].pic_id,
                    "type": record['item'].type,
                    "type_code": record['item'].type_code,
                    "size": record['item'].size,
                    "confidence": record['item'].confidence,
                    "pic_name": record['item'].pic_name,  # 保持原始文件名
                    "new_file_name": new_filename,
                    "return_content_name": renamed_file,
                    "original_file_name": record['item'].pic_name,
                    "return_content_type": picture_type  # 使用传入的picture_type
                })
            
            logger.info(f"品牌{brand}, 货号{product_code}, 完整流程处理完成，返回{len(result_list)}个结果")
            return result_list
    
    def _process_ln_brand(
        self,
        product_code: str,
        pic_list_str: str,
        picture_type: str = "pants"
    ) -> List[Dict[str, Any]]:
        """
        处理李宁(LN)品牌的图片选择逻辑
        
        只保留文件名中带WHITE的图片，WHITE之后的数字为序号，保留序号1-6。
        新文件名格式: product_code_序号.后缀
        示例: AKLV217-5-MXK-WHITE-1.jpg -> product_code_01.jpg
        """
        logger.info(f"开始处理品牌-李宁, 货号{product_code}的图片列表")
        
        # 1. 解析图片列表
        pic_list = self.parse_pic_list(pic_list_str)
        if not pic_list:
            raise ValueError("图片列表为空")
        
        logger.info(f"品牌LN, 货号{product_code}, 共{len(pic_list)}张图片")
        
        # 2. 提取WHITE图片及其序号
        white_items: List[Tuple[int, Dict[str, Any]]] = []
        for pic_info in pic_list:
            pic_name = pic_info.get("pic_name", "")
            seq = self._extract_ln_white_sequence(pic_name)
            if seq is not None and 1 <= seq <= 6:
                white_items.append((seq, pic_info))
        
        if not white_items:
            logger.warning(f"品牌LN, 货号{product_code}, 没有找到符合条件的WHITE图片")
            return []
        
        # 3. 按序号排序，去重（同一序号取第一个）
        white_items.sort(key=lambda x: x[0])
        seen_seq: set = set()
        selected_items: List[Tuple[int, Dict[str, Any]]] = []
        for seq, pic_info in white_items:
            if seq not in seen_seq:
                seen_seq.add(seq)
                selected_items.append((seq, pic_info))
        
        logger.info(f"品牌LN, 货号{product_code}, 筛选出{len(selected_items)}张WHITE图片")
        
        # 4. 生成结果列表
        result_list = []
        for seq, pic_info in selected_items:
            pic_name = pic_info.get("pic_name", "")
            pic_id = pic_info.get("pic_id")
            
            # 获取原始文件后缀
            suffix = Path(pic_name).suffix or ".jpg"
            
            # 生成新文件名: product_code_序号.后缀
            new_filename = f"{product_code}_{seq:02d}{suffix}"
            
            # 生成OSS新路径
            new_path = self.generate_new_file_path(
                new_file_name=new_filename,
                brand="LN",
                product_code=product_code,
                picture_type=picture_type
            )
            
            # 调用OSS重命名接口
            renamed_file = self.rename_file_in_oss(
                old_name=pic_name,
                new_name=new_path
            )
            
            if renamed_file:
                logger.info(f"品牌LN, 货号{product_code}, OSS重命名成功: {pic_name} -> {renamed_file}")
            
            result_list.append({
                "product_code": product_code,
                "tag_first_type": "1",
                "tag_second_type": "11",
                "tag_result": None,
                "pic_id": pic_id,
                "type": seq,
                "type_code": seq,
                "size": 0,
                "confidence": 1.0,
                "pic_name": pic_name,
                "new_file_name": new_filename,
                "return_content_name": renamed_file,
                "original_file_name": pic_name,
                "return_content_type": picture_type
            })
        
        logger.info(f"品牌LN, 货号{product_code}, 完整流程处理完成，返回{len(result_list)}个结果")
        return result_list
    
    def _extract_ln_white_sequence(self, pic_name: str) -> Optional[int]:
        """从文件名中提取WHITE后面的序号。
        
        示例: AKLV217-5-MXK-WHITE-1.jpg -> 1
        """
        match = re.search(r'WHITE[-_]?(\d+)', pic_name, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None
    
    def generate_new_file_path(
        self,
        new_file_name: str,
        brand: str,
        product_code: str,
        picture_type: str = "pants"
    ) -> str:
        """
        生成新的文件路径
        
        Args:
            new_file_name: 新文件名
            brand: 品牌
            product_code: 产品编码
            picture_type: 图片类型，默认为"pants"
        
        Returns:
            新的文件路径
        """
        # 获取当前时间的年/月/日
        today = datetime.today()
        year = today.strftime("%Y")
        month = today.strftime("%m")
        day = today.strftime("%d")
        
        # 拼接成目标路径
        new_path = f"temp3/{year}/{month}/{month}-{day}/{picture_type}/{brand}/{product_code}/{new_file_name}"
        
        return new_path
    
    def rename_file_in_oss(
        self,
        old_name: str,
        new_name: str,
        name_type: str = "default"
    ) -> Optional[str]:
        """
        调用OSS接口重命名文件
        
        Args:
            old_name: 原文件名
            new_name: 新文件名（完整路径）
            name_type: 名称类型
        
        Returns:
            重命名后的文件名，失败返回None
        """
        try:
            response = requests.get(
                self.oss_rename_url,
                params={
                    "old_name": old_name,
                    "name_type": name_type,
                    "new_name": new_name
                },
                timeout=30
            )

            if response.status_code != 200:
                logger.error(f"重命名文件失败 {old_name}: status_code={response.status_code}")
                return None
            
            # 重命名成功，返回新路径
            logger.info(f"文件重命名成功: {old_name} -> {new_name}")
            return new_name
                
        except Exception as e:
            logger.error(f"重命名文件异常 {old_name}: {e}")
            return None
