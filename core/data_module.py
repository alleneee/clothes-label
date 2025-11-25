#!/usr/bin/env python3
"""
简化数据模块 - 仅使用HuggingFace Datasets方式
所有自定义数据集处理代码已移除，统一使用HFDatasetsModule
"""

from .hf_datasets_module import HFDatasetsModule, create_hf_datamodule

# 为了兼容性，重新导出主要类和函数
ClothesDataModule = HFDatasetsModule


def create_datamodule(config: dict) -> HFDatasetsModule:
    """从配置创建数据模块"""
    return create_hf_datamodule(config)


# 兼容性别名
EnhancedClothesDataModule = HFDatasetsModule


def get_datamodule_class():
    """获取数据模块类"""
    return HFDatasetsModule


# 所有自定义数据集处理代码已移除，包括：
# - EnhancedImageDataset
# - 自定义数据加载和预处理
# - 复杂的增强管道
# - 手动数据分割
# - 类别权重计算
# 
# 现在统一使用 HFDatasetsModule 处理所有数据相关功能
