"""
核心训练模块
包含主要的训练、数据处理和预测功能
"""

# 导入现有模块（避免循环导入）
from .data_module import EnhancedClothesDataModule
from .hardware_optimizer import HardwareDetector, ConfigOptimizer

__all__ = [
    'EnhancedClothesDataModule',
    'HardwareDetector',
    'ConfigOptimizer'
]