#!/usr/bin/env python3
"""
增强版衣服分类训练脚本
目标：将准确率提升到80%+
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor,
    TQDMProgressBar, DeviceStatsMonitor, ModelSummary
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.train import ProductClassifier
from core.data_module import HFDatasetsModule


def setup_callbacks(config):
    """设置回调函数"""
    callbacks = []
    
    # 检查点回调
    checkpoint_config = config['training']['checkpoint']
    
    # 生成包含日期的文件名
    current_date = datetime.now().strftime("%Y%m%d")
    base_filename = config['checkpointing']['filename']
    # 在基础文件名前添加日期
    filename_with_date = f"{current_date}-{base_filename}"
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['checkpointing']['dirpath'],
        filename=filename_with_date,
        monitor=checkpoint_config['monitor'],
        mode=checkpoint_config['mode'],
        save_top_k=checkpoint_config['save_top_k'],
        save_last=checkpoint_config['save_last'],
        every_n_epochs=checkpoint_config.get('every_n_epochs', 1),
        save_weights_only=config['checkpointing'].get('save_weights_only', False),
        auto_insert_metric_name=False
    )
    callbacks.append(checkpoint_callback)
    
    # 早停回调
    early_stopping_config = config['training']['early_stopping']
    early_stopping = EarlyStopping(
        monitor=early_stopping_config['monitor'],
        patience=early_stopping_config['patience'],
        mode=early_stopping_config['mode'],
        min_delta=early_stopping_config['min_delta'],
        verbose=True
    )
    callbacks.append(early_stopping)
    
    # 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # 进度条
    progress_bar = TQDMProgressBar()
    callbacks.append(progress_bar)
    
    # 设备状态监控
    if config['logging'].get('log_gpu_memory', False):
        device_stats = DeviceStatsMonitor()
        callbacks.append(device_stats)
    
    # 模型摘要
    model_summary = ModelSummary(max_depth=2)
    callbacks.append(model_summary)
    
    return callbacks


def setup_logger(config):
    """设置日志记录器"""
    log_config = config['logging']
    
    logger = TensorBoardLogger(
        save_dir=log_config['log_dir'],
        name=log_config['experiment_name'],
        log_graph=log_config.get('tensorboard', {}).get('log_graph', False),
        default_hp_metric=False
    )
    
    return logger


def setup_trainer(config, callbacks, logger):
    """设置训练器"""
    training_config = config['training']
    multi_gpu_config = config['multi_gpu']
    
    # 分布式策略
    strategy = 'auto'  # 默认为auto，让PyTorch Lightning自动选择
    if torch.cuda.device_count() > 1:
        strategy = DDPStrategy(
            find_unused_parameters=multi_gpu_config.get('find_unused_parameters', False)
        )
    
    # 训练器参数
    trainer_kwargs = {
        'max_epochs': training_config['max_epochs'],
        'callbacks': callbacks,
        'logger': logger,
        'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
        'devices': 'auto',
        'strategy': strategy,
        'precision': multi_gpu_config.get('precision', 32),
        'gradient_clip_val': training_config.get('gradient_clip_val', 0),
        'accumulate_grad_batches': training_config.get('accumulate_grad_batches', 1),
        'deterministic': config.get('performance', {}).get('deterministic', False),
        'benchmark': config.get('performance', {}).get('benchmark', True),
        'enable_progress_bar': True,
        'enable_model_summary': True,
    }
    
    # 验证配置
    validation_config = config.get('validation', {})
    trainer_kwargs.update({
        'check_val_every_n_epoch': validation_config.get('check_val_every_n_epoch', 1),
        'val_check_interval': validation_config.get('val_check_interval', 1.0),
        'limit_val_batches': validation_config.get('limit_val_batches', 1.0),
        'num_sanity_val_steps': validation_config.get('num_sanity_val_steps', 2)
    })
    
    # 高级训练配置
    advanced_config = config.get('advanced_training', {})
    adaptive_config = advanced_config.get('adaptive', {})
    
    # 注意：auto_lr_find 和 auto_scale_batch_size 在新版本中需要单独调用
    # 这里先创建trainer，然后在后面使用tuner来调用
    
    trainer = pl.Trainer(**trainer_kwargs)
    
    return trainer


def main():
    parser = argparse.ArgumentParser(description='增强版衣服分类训练')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练')
    parser.add_argument('--test-only', action='store_true',
                        help='只进行测试')
    
    args = parser.parse_args()
    
    # 加载配置
    print(f"📖 加载配置文件: {args.config}")
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置环境变量
    env_vars = config.get('environment_variables', {})
    for key, value in env_vars.items():
        os.environ[key] = str(value)
    
    # 创建输出目录
    os.makedirs(config['checkpointing']['dirpath'], exist_ok=True)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    
    # 设置随机种子
    pl.seed_everything(42, workers=True)
    
    print("🚀 开始训练准备...")
    
    # 1. 数据模块
    print("📊 初始化数据模块...")
    data_module = HFDatasetsModule(
        data_dir=config['data']['data_dir'],
        cache_dir=config['data'].get('cache_dir', 'datasets/hf_cache'),
        dataset_name=config['data'].get('dataset_name', 'clothes_classification_v1'),
        batch_size=config['data']['batch_size'],
        image_size=config['data']['image_size'],
        num_workers=config['data']['num_workers'],
        augmentation_enabled=config['data']['augmentation']['enabled'],
        class_names=None if config['classes'].get('auto_detect', True) else config['classes'].get('names'),
        auto_detect_classes=config['classes'].get('auto_detect', True)
    )
    data_module.prepare_data()
    data_module.setup('fit')
    
    # 2. 模型
    print("🤖 初始化模型...")
    if args.resume:
        print(f"📥 从检查点恢复: {args.resume}")
        model = ProductClassifier.load_from_checkpoint(args.resume, config=config)
    else:
        # 获取数据集信息
        dataset_info = data_module.get_dataset_info()
        
        # 更新配置中的类别信息
        config['model']['num_classes'] = dataset_info['num_classes']
        config['classes']['names'] = dataset_info['class_names']
        config['classes']['num_classes'] = dataset_info['num_classes']
        
        model = ProductClassifier(config)
        
        # 设置类别权重
        if data_module.class_weights is not None:
            model.class_weights = data_module.class_weights
    
    # 3. 回调和日志
    print("📝 设置回调和日志...")
    callbacks = setup_callbacks(config)
    logger = setup_logger(config)
    
    # 4. 训练器
    print("⚙️ 设置训练器...")
    trainer = setup_trainer(config, callbacks, logger)
    
    # 5. 模型信息
    print("\n" + "="*60)
    print("📊 模型信息:")
    print(f"   - 模型名称: {config['model']['name']}")
    print(f"   - 类别数量: {dataset_info['num_classes']}")
    print(f"   - 图片尺寸: {config['data']['image_size']}x{config['data']['image_size']}")
    print(f"   - 批次大小: {config['data']['batch_size']}")
    print(f"   - 学习率: {config['model']['learning_rate']}")
    print(f"   - 最大轮数: {config['training']['max_epochs']}")
    
    print("\n📊 数据信息:")
    print(f"   - 训练样本: {len(data_module.train_dataset)}")
    if data_module.val_dataset:
        print(f"   - 验证样本: {len(data_module.val_dataset)}")
    
    print("\n📊 增强设置:")
    aug_config = config['data'].get('augmentation', {})
    if aug_config.get('enabled', False):
        print(f"   - 数据增强: 启用")
        print(f"   - Mixup Alpha: {aug_config.get('mixup_alpha', 0)}")
        print(f"   - CutMix Alpha: {aug_config.get('cutmix_alpha', 0)}")
        print(f"   - 旋转角度: {aug_config.get('rotation_degrees', 0)}°")
        print(f"   - 水平翻转: {aug_config.get('horizontal_flip', 0)}")
    else:
        print(f"   - 数据增强: 禁用")
    
    print("\n📊 设备信息:")
    print(f"   - 可用GPU: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"   - GPU {i}: {gpu_name}")
    print("="*60 + "\n")
    
    # 6. 自动调优（如果启用）
    advanced_config = config.get('advanced_training', {})
    adaptive_config = advanced_config.get('adaptive', {})
    
    # 暂时禁用自动调优功能，避免版本兼容性问题
    # if adaptive_config.get('lr_finder', False):
    #     print("🔍 自动寻找最佳学习率...")
    #     lr_finder = trainer.tuner.lr_find(model, data_module)
    #     new_lr = lr_finder.suggestion()
    #     print(f"💡 建议学习率: {new_lr}")
    #     model.learning_rate = new_lr
    # 
    # if adaptive_config.get('auto_batch_size', False):
    #     print("🔍 自动调整批次大小...")
    #     trainer.tuner.scale_batch_size(model, data_module, mode='power')
    #     print(f"💡 调整后批次大小: {data_module.batch_size}")
    
    try:
        if args.test_only:
            # 只进行测试
            print("🧪 开始测试...")
            trainer.test(model, data_module)
        else:
            # 开始训练
            print("🚀 开始训练...")
            trainer.fit(model, data_module, ckpt_path=args.resume)
            
            # 训练完成后进行测试（使用验证集作为测试集）
            if hasattr(data_module, 'val_dataset') and data_module.val_dataset:
                print("🧪 开始测试...")
                trainer.test(model, data_module, ckpt_path='best')
    
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🏁 训练结束")


if __name__ == "__main__":
    main() 