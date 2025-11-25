#!/usr/bin/env python3
"""
HF Datasets数据模块 - 支持增量数据扩展
替代原有复杂数据处理，提供简洁高效的数据加载方案
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Generator, Tuple
from datetime import datetime
import logging

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict, Features, Value, Image, load_dataset, concatenate_datasets
import datasets
from torchvision import transforms
import numpy as np
from PIL import Image as PILImage

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 静默PIL警告
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='PIL')


class ClothesDatasetBuilder:
    """衣服数据集构建器 - 支持增量数据添加"""
    
    def __init__(
        self,
        data_dir: str = "datasets/main/train/clothes",
        cache_dir: str = "datasets/hf_cache",
        dataset_name: str = "clothes_classification_v1",
        class_names: Optional[List[str]] = None,
        auto_detect: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.dataset_name = dataset_name
        self.auto_detect = auto_detect
        
        if class_names:
            self.class_names = list(class_names)
        elif auto_detect:
            self.class_names = self._discover_classes()
        else:
            raise ValueError("When auto_detect is False, class_names must be provided.")
        
        # 创建类别到ID的映射
        self.class_to_id = {name: idx for idx, name in enumerate(self.class_names)}
        self.id_to_class = {idx: name for idx, name in enumerate(self.class_names)}
        
        # 确保缓存目录存在
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _discover_classes(self) -> List[str]:
        """自动发现数据目录中的类别名称"""
        if not self.data_dir.exists():
            logger.warning(f"⚠️ 数据目录不存在: {self.data_dir}, 使用默认空类别列表")
            return []

        class_names = sorted([
            item.name for item in self.data_dir.iterdir() if item.is_dir()
        ])

        if not class_names:
            logger.warning(f"⚠️ 在 {self.data_dir} 中未发现类别目录")
        return class_names
        
    def _scan_images(self, scan_dir: Optional[Path] = None) -> Generator[Dict[str, Any], None, None]:
        """扫描图片文件，生成数据记录"""
        scan_dir = scan_dir or self.data_dir
        
        logger.info(f"🔍 扫描图片目录: {scan_dir}")
        
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        for class_dir in scan_dir.iterdir():
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            if class_name not in self.class_to_id:
                logger.warning(f"⚠️  未知类别: {class_name}")
                continue
                
            class_id = self.class_to_id[class_name]
            
            image_files = []
            for ext in supported_formats:
                image_files.extend(class_dir.glob(f"*{ext}"))
            
            logger.info(f"📁 {class_name}: 发现 {len(image_files)} 张图片")
            
            for img_path in image_files:
                try:
                    # 验证图片可以打开，宽容处理格式不匹配的文件
                    with PILImage.open(img_path) as img:
                        width, height = img.size
                        # 记录实际格式
                        actual_format = img.format
                        
                    # 生成唯一ID
                    relative_path = str(img_path.relative_to(scan_dir))
                    unique_id = hashlib.md5(relative_path.encode()).hexdigest()
                    
                    yield {
                        'id': unique_id,
                        'image': str(img_path),  # HF Datasets会自动处理图片加载
                        'class_name': class_name,
                        'label': class_id,
                        'width': width,
                        'height': height,
                        'file_size': img_path.stat().st_size,
                        'relative_path': relative_path,
                        'scan_time': datetime.now().isoformat(),
                        'actual_format': actual_format,  # 添加实际格式信息
                    }
                    
                except Exception as e:
                    logger.warning(f"⚠️  无法处理图片 {img_path}: {e} - 跳过")
                    continue
    
    def build_dataset(self, force_rebuild: bool = False) -> Dataset:
        """构建HF Dataset"""
        dataset_path = self.cache_dir / f"{self.dataset_name}.hf"
        
        # 检查缓存
        if dataset_path.exists() and not force_rebuild:
            logger.info(f"📦 加载缓存数据集: {dataset_path}")
            try:
                return Dataset.load_from_disk(str(dataset_path))
            except Exception as e:
                logger.warning(f"⚠️  缓存加载失败: {e}，重新构建")
        
        logger.info("🏗️  构建新数据集...")
        
        # 定义数据集特征
        features = Features({
            'id': Value('string'),
            'image': Image(),  # HF Datasets的Image特征
            'class_name': Value('string'),
            'label': datasets.ClassLabel(names=self.class_names),  # 使用ClassLabel以支持分层采样
            'width': Value('int32'),
            'height': Value('int32'),
            'file_size': Value('int64'),
            'relative_path': Value('string'),
            'scan_time': Value('string'),
        })
        
        # 从生成器创建数据集
        dataset = Dataset.from_generator(
            self._scan_images,
            features=features
        )
        
        # 保存到缓存
        logger.info(f"💾 保存数据集到: {dataset_path}")
        dataset.save_to_disk(str(dataset_path))
        
        # 保存元数据
        metadata = {
            'dataset_name': self.dataset_name,
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'class_to_id': self.class_to_id,
            'total_samples': len(dataset),
            'created_time': datetime.now().isoformat(),
            'data_dir': str(self.data_dir),
        }
        
        with open(self.cache_dir / f"{self.dataset_name}_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 数据集构建完成: {len(dataset)} 个样本")
        return dataset
    
    def add_new_data(self, new_data_dir: str) -> Dataset:
        """增量添加新数据"""
        logger.info(f"📈 增量添加数据: {new_data_dir}")
        
        # 加载现有数据集
        existing_dataset = self.build_dataset(force_rebuild=False)
        existing_ids = set(existing_dataset['id'])
        
        # 扫描新数据
        new_data_path = Path(new_data_dir)
        new_records = []
        
        for record in self._scan_images(new_data_path):
            # 为新数据生成不同的ID前缀，确保不重复
            record['id'] = f"new_{record['id']}"
            if record['id'] not in existing_ids:
                new_records.append(record)
        
        if not new_records:
            logger.info("📭 没有发现新数据")
            return existing_dataset
        
        logger.info(f"📥 发现 {len(new_records)} 条新数据")
        
        # 创建新数据的Dataset
        features = existing_dataset.features
        new_dataset = Dataset.from_list(new_records, features=features)
        
        # 合并数据集
        combined_dataset = concatenate_datasets([existing_dataset, new_dataset])
        
        # 更新版本号
        version_num = int(self.dataset_name.split('_v')[-1]) + 1
        new_dataset_name = f"clothes_classification_v{version_num}"
        
        # 保存新版本
        new_dataset_path = self.cache_dir / f"{new_dataset_name}.hf"
        combined_dataset.save_to_disk(str(new_dataset_path))
        
        # 更新元数据
        metadata = {
            'dataset_name': new_dataset_name,
            'previous_version': self.dataset_name,
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'total_samples': len(combined_dataset),
            'new_samples': len(new_records),
            'updated_time': datetime.now().isoformat(),
        }
        
        with open(self.cache_dir / f"{new_dataset_name}_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 数据集更新完成: 新增 {len(new_records)} 样本，总计 {len(combined_dataset)} 样本")
        
        # 更新当前数据集名称
        self.dataset_name = new_dataset_name
        
        return combined_dataset


class HFDatasetsModule(pl.LightningDataModule):
    """HF Datasets数据模块 - PyTorch Lightning兼容"""
    
    def __init__(
        self,
        data_dir: str = "datasets/main/train/clothes",
        cache_dir: str = "datasets/hf_cache", 
        dataset_name: str = "clothes_classification_v1",
        batch_size: int = 32,
        image_size: int = 384,
        num_workers: int = 8,
        train_split: float = 0.8,
        val_split: float = 0.2,
        pin_memory: bool = True,
        augmentation_enabled: bool = True,
        class_names: Optional[List[str]] = None,
        auto_detect_classes: bool = True,
        **kwargs
    ):
        super().__init__()
        
        # 基础参数
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.pin_memory = pin_memory
        self.augmentation_enabled = augmentation_enabled
        
        # 数据集构建器
        self.builder = ClothesDatasetBuilder(
            data_dir=data_dir,
            cache_dir=cache_dir,
            dataset_name=dataset_name,
            class_names=class_names,
            auto_detect=auto_detect_classes
        )
        
        # 数据变换
        self.setup_transforms()
        
        # 数据集变量
        self.dataset: Optional[Dataset] = None
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        
        # 类别权重（延迟计算）
        self._class_weights: Optional[torch.Tensor] = None
        
    def setup_transforms(self):
        """设置数据变换"""
        # 训练时的数据增强
        self.train_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) if self.augmentation_enabled else transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 验证时的变换（无增强）
        self.val_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def setup(self, stage: str = None):
        """设置数据集"""
        if self.dataset is None:
            # 构建或加载数据集
            self.dataset = self.builder.build_dataset()
        
        # 分割数据集
        if stage == 'fit' or stage is None:
            # 按类别分层分割，保证每个类别的比例
            split_dataset = self.dataset.train_test_split(
                test_size=self.val_split,
                seed=42,
                stratify_by_column='label'  # 按标签分层
            )
            
            self.train_dataset = split_dataset['train']
            self.val_dataset = split_dataset['test']
            
            # 设置数据格式和变换
            self.train_dataset = self.train_dataset.with_transform(self._train_transform)
            self.val_dataset = self.val_dataset.with_transform(self._val_transform)
            
            logger.info(f"📊 数据集分割完成:")
            logger.info(f"   训练集: {len(self.train_dataset)} 样本")
            logger.info(f"   验证集: {len(self.val_dataset)} 样本")
    
    def _train_transform(self, examples):
        """训练数据变换函数"""
        images = []
        for img in examples['image']:
            # 处理调色板图像和透明度问题
            if img.mode == 'P':
                img = img.convert('RGBA')
            img = img.convert('RGB')
            images.append(self.train_transform(img))
        return {
            'image': images,
            'label': examples['label']
        }
    
    def _val_transform(self, examples):
        """验证数据变换函数"""
        images = []
        for img in examples['image']:
            # 处理调色板图像和透明度问题
            if img.mode == 'P':
                img = img.convert('RGBA')
            img = img.convert('RGB')
            images.append(self.val_transform(img))
        return {
            'image': images,
            'label': examples['label']
        }
    
    def train_dataloader(self) -> DataLoader:
        """训练数据加载器"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        """验证数据加载器"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )
    
    def test_dataloader(self) -> DataLoader:
        """测试数据加载器（使用验证集）"""
        return self.val_dataloader()
    
    def _collate_fn(self, batch):
        """批处理函数"""
        # 检查数据类型
        if isinstance(batch[0], dict):
            # 正常的dict类型
            images = torch.stack([item['image'] for item in batch])
            labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        else:
            # 如果不是dict，尝试直接解析
            images = torch.stack([item[0] if isinstance(item, (list, tuple)) else item for item in batch])
            labels = torch.tensor([item[1] if isinstance(item, (list, tuple)) else 0 for item in batch], dtype=torch.long)
        
        return {
            'image': images,
            'label': labels
        }
    
    def add_new_data(self, new_data_dir: str):
        """增量添加新数据"""
        logger.info(f"📈 添加新数据到数据模块: {new_data_dir}")
        
        # 使用构建器添加新数据
        self.dataset = self.builder.add_new_data(new_data_dir)
        
        # 重新设置数据集分割
        self.setup()
        
        logger.info("✅ 新数据添加完成，数据集已更新")
    
    def get_class_distribution(self) -> Dict[str, int]:
        """获取类别分布"""
        if self.dataset is None:
            self.dataset = self.builder.build_dataset()
        
        class_counts = {}
        for class_name in self.builder.class_names:
            class_id = self.builder.class_to_id[class_name]
            count = sum(1 for label in self.dataset['label'] if label == class_id)
            class_counts[class_name] = count
        
        return class_counts
    
    def get_class_weights(self) -> Optional[torch.Tensor]:
        """计算类别权重，用于解决类别不平衡问题"""
        if self.dataset is None:
            self.dataset = self.builder.build_dataset()
        
        class_counts = list(self.get_class_distribution().values())
        total_samples = sum(class_counts)
        
        if total_samples == 0:
            return None
        
        # 计算权重：类别权重 = 总样本数 / (类别数 * 类别样本数)
        class_weights = []
        for count in class_counts:
            if count > 0:
                weight = total_samples / (len(class_counts) * count)
                class_weights.append(weight)
            else:
                class_weights.append(0.0)
        
        return torch.tensor(class_weights, dtype=torch.float32)
    
    @property
    def class_weights(self) -> Optional[torch.Tensor]:
        """类别权重属性"""
        if self._class_weights is None:
            self._class_weights = self.get_class_weights()
        return self._class_weights
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """获取数据集信息"""
        if self.dataset is None:
            self.dataset = self.builder.build_dataset()
        
        return {
            'total_samples': len(self.dataset),
            'num_classes': len(self.builder.class_names),
            'class_names': self.builder.class_names,
            'class_distribution': self.get_class_distribution(),
            'dataset_name': self.dataset_name,
            'cache_dir': str(self.cache_dir),
        }


# 兼容性函数，方便从现有代码迁移
def create_hf_datamodule(config: dict) -> HFDatasetsModule:
    """从配置创建HF Datasets数据模块"""
    data_config = config.get('data', {})
    
    return HFDatasetsModule(
        data_dir=data_config.get('data_dir', 'datasets/main/train/clothes'),
        batch_size=data_config.get('batch_size', 32),
        image_size=data_config.get('image_size', 384),
        num_workers=data_config.get('num_workers', 8),
        augmentation_enabled=data_config.get('augmentation', {}).get('enabled', True),
    )


if __name__ == "__main__":
    # 演示用法
    print("🚀 HF Datasets模块演示")
    
    # 创建数据模块
    dm = HFDatasetsModule()
    
    # 构建数据集
    dm.setup()
    
    # 显示数据集信息
    info = dm.get_dataset_info()
    print(f"📊 数据集信息: {info}")
    
    # 测试数据加载器
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    print(f"📦 批次形状: {batch['image'].shape}, {batch['label'].shape}") 