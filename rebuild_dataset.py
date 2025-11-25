#!/usr/bin/env python
"""重建数据集脚本，测试WebP支持"""

import sys
import os
sys.path.append('/data/hx/model-train')

from core.hf_datasets_module import HFDatasetsModule
import yaml

def main():
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("🔄 开始重建数据集...")
    
    # 创建数据模块
    data_module = HFDatasetsModule(
        data_dir=config['data']['data_dir'],
        batch_size=32,
        image_size=config['data']['image_size'],
        num_workers=1,
        augmentation_enabled=False
    )
    
    # 强制重建数据集
    dataset = data_module.builder.build_dataset(force_rebuild=True)
    print(f'✅ 数据集重建完成！总样本数: {len(dataset)}')

if __name__ == "__main__":
    main()
