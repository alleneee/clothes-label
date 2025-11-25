"""
PyTorch Lightning 商品分类训练脚本
优化版本，支持Python 3.11特性和CUDA 12.4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import timm
import argparse
import yaml
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 配置中文字体和静默警告
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=UserWarning, module='seaborn')
from typing import Dict, Any, Optional, Union, Tuple
import sys
import os
from datetime import datetime

# Python 3.11性能优化
if sys.version_info >= (3, 11):
    # 启用Python 3.11的性能优化
    import gc
    gc.set_threshold(700, 10, 10)  # 优化垃圾回收

try:
    # 尝试相对导入（当作为包的一部分运行时）
    from .hardware_optimizer import HardwareDetector, ConfigOptimizer, DynamicConfigAdjuster
except ImportError:
    # 尝试绝对导入（当直接运行时）
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.hardware_optimizer import HardwareDetector, ConfigOptimizer, DynamicConfigAdjuster


class ProductClassifier(pl.LightningModule):
    """商品分类器 Lightning 模块"""

    def __init__(self, config, enable_dynamic_adjustment=False):
        super().__init__()
        self.save_hyperparameters()

        # 从配置加载参数
        model_config = config['model']
        self.num_classes = model_config['num_classes']
        self.model_name = model_config['name']
        self.learning_rate = model_config['learning_rate']
        self.weight_decay = model_config['weight_decay']
        # 获取类别名称，支持从checkpoint加载时没有类别名称的情况
        self.class_names = config.get('classes', {}).get('names', [])

        # 动态配置调整
        self.enable_dynamic_adjustment = enable_dynamic_adjustment
        if enable_dynamic_adjustment:
            self.dynamic_adjuster = DynamicConfigAdjuster(config)
            self.batch_times = []
        
        # 创建模型
        pretrained = config.get('model', {}).get('pretrained', True)  # 从配置文件读取pretrained设置
        use_timm_pretrained = pretrained and not model_config.get('local_pretrained_path')
        self.model = timm.create_model(
            self.model_name,
            pretrained=use_timm_pretrained,
            num_classes=self.num_classes,
            drop_rate=model_config.get('drop_rate', 0.2),
            drop_path_rate=model_config.get('drop_path_rate', 0.2)
        )

        local_pretrained_path = model_config.get('local_pretrained_path')
        if pretrained and local_pretrained_path:
            try:
                if os.path.exists(local_pretrained_path):
                    print(f"📂 从本地加载预训练权重: {local_pretrained_path}")
                    checkpoint = torch.load(local_pretrained_path, map_location='cpu')
                    state_dict = checkpoint.get('state_dict', checkpoint)
                    self.model.load_state_dict(state_dict, strict=False)
                    print("✅ 本地预训练权重加载完成")
                else:
                    print(f"⚠️ 本地预训练权重不存在: {local_pretrained_path}")
            except Exception as e:
                print(f"⚠️ 本地预训练权重加载失败: {e}")
        
        # torch.compile()优化 - PyTorch 2.0性能提升
        performance_config = config.get('performance', {})
        if performance_config.get('enable_torch_compile', True):
            try:
                # 检查PyTorch版本和torch.compile支持
                if hasattr(torch, 'compile') and hasattr(torch, '__version__') and torch.__version__ >= '2.0.0':
                    compile_mode = performance_config.get('torch_compile_mode', 'reduce-overhead')
                    
                    # 支持的编译模式
                    valid_modes = ['default', 'reduce-overhead', 'max-autotune']
                    if compile_mode not in valid_modes:
                        compile_mode = 'reduce-overhead'
                    
                    print(f"🚀 启用torch.compile()优化，模式: {compile_mode}")
                    self.model = torch.compile(self.model, mode=compile_mode)
                    print("✅ torch.compile()优化已启用")
                else:
                    if hasattr(torch, '__version__'):
                        print(f"⚠️ PyTorch版本 {torch.__version__} 不支持torch.compile()，需要2.0.0+")
                    else:
                        print("⚠️ 当前PyTorch版本不支持torch.compile()")
            except Exception as e:
                print(f"⚠️ torch.compile()启用失败: {e}")
                print("📝 继续使用未编译的模型")
        else:
            print("📝 torch.compile()优化已禁用")
        
        # 损失函数 (将在训练时根据数据不均衡情况动态设置)
        self.criterion = nn.CrossEntropyLoss()
        self.use_balanced_loss = False
        
        # 用于收集预测结果
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def set_balanced_loss(self, balanced_loss):
        """设置平衡的损失函数"""
        if balanced_loss is not None:
            self.criterion = balanced_loss
            self.use_balanced_loss = True
            print("✅ 已启用平衡损失函数")
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        import time
        start_time = time.time()

        # 支持字典和元组两种格式
        if isinstance(batch, dict):
            # 字典格式: {'image': x, 'label': y}
            x, y = batch['image'], batch['label']
            logits = self(x)
            loss = self.criterion(logits, y)
            
            # 计算准确率
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(preds == y).float() / len(y)
        elif len(batch) == 4:
            # Mixup/CutMix: (x, y_a, y_b, lam)
            x, y_a, y_b, lam = batch
            logits = self(x)
            
            # 计算混合损失
            loss_a = self.criterion(logits, y_a)
            loss_b = self.criterion(logits, y_b)
            loss = lam * loss_a + (1 - lam) * loss_b
            
            # 计算准确率（使用主要标签）
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(preds == y_a).float() / len(y_a)
        else:
            # 正常批次: (x, y)
            x, y = batch
            logits = self(x)
            loss = self.criterion(logits, y)
            
            # 计算准确率
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(preds == y).float() / len(y)

        # 记录指标
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)

        # 性能监控和动态调整
        if self.enable_dynamic_adjustment:
            batch_time = time.time() - start_time
            self.batch_times.append(batch_time)

            # 每100个batch检查一次
            if batch_idx % 100 == 0 and len(self.batch_times) >= 10:
                avg_batch_time = sum(self.batch_times[-10:]) / 10

                # 获取内存使用情况
                if torch.cuda.is_available():
                    memory_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                    gpu_utilization = None  # 需要nvidia-ml-py来获取
                else:
                    import psutil
                    memory_usage = psutil.virtual_memory().percent / 100
                    gpu_utilization = None

                # 检查是否需要调整
                if self.dynamic_adjuster.monitor_training_performance(
                    self.current_epoch, avg_batch_time, memory_usage, gpu_utilization
                ):
                    print(f"\n🔧 检测到性能问题，正在调整配置...")
                    new_config = self.dynamic_adjuster.adjust_config()
                    # 注意：实际的批次大小调整需要重启训练，这里只是记录

        return loss
    
    def validation_step(self, batch, batch_idx):
        # 支持字典和元组两种格式
        if isinstance(batch, dict):
            x, y = batch['image'], batch['label']
        else:
            x, y = batch
        
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == y).float() / len(y)
        
        # 记录指标
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        
        # 收集预测结果
        self.validation_step_outputs.append({
            'preds': preds.cpu(),
            'targets': y.cpu(),
            'loss': loss.cpu()
        })
        
        return {'val_loss': loss, 'val_acc': acc}
    
    def on_validation_epoch_end(self):
        """验证轮次结束时计算详细指标"""
        if not self.validation_step_outputs:
            return
        
        # 合并所有预测结果
        all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs])
        all_targets = torch.cat([x['targets'] for x in self.validation_step_outputs])
        
        # 计算每个类别的准确率
        for i, class_name in enumerate(self.class_names):
            class_mask = all_targets == i
            if class_mask.sum() > 0:
                class_acc = (all_preds[class_mask] == all_targets[class_mask]).float().mean()
                self.log(f'val_acc_{class_name}', class_acc, on_epoch=True)
        
        # 清空输出
        self.validation_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        # 支持字典和元组两种格式
        if isinstance(batch, dict):
            x, y = batch['image'], batch['label']
        else:
            x, y = batch
        
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == y).float() / len(y)
        
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', acc, on_epoch=True)
        
        # 收集预测结果
        self.test_step_outputs.append({
            'preds': preds.cpu(),
            'targets': y.cpu(),
            'probs': F.softmax(logits, dim=1).cpu()
        })
        
        return {'test_loss': loss, 'test_acc': acc}
    
    def on_test_epoch_end(self):
        """测试结束时生成详细报告"""
        if not self.test_step_outputs:
            return
        
        # 合并所有预测结果
        all_preds = torch.cat([x['preds'] for x in self.test_step_outputs])
        all_targets = torch.cat([x['targets'] for x in self.test_step_outputs])
        
        # 生成分类报告
        print("\n" + "="*50)
        print("测试集分类报告")
        print("="*50)
        print(classification_report(
            all_targets.numpy(),
            all_preds.numpy(),
            target_names=self.class_names
        ))
        
        # 生成混淆矩阵
        cm = confusion_matrix(all_targets.numpy(), all_preds.numpy())
        self._plot_confusion_matrix(cm)
        
    def _plot_confusion_matrix(self, cm):
        """绘制混淆矩阵"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names,
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
    
    def predict_single(self, image_path: str):
        """单张图片预测"""
        from PIL import Image
        from torchvision import transforms
        
        # 数据预处理
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 加载和预处理图片
        image = Image.open(image_path)
        # 修复PIL透明度警告：如果是调色板图像且有透明度，先转换为RGBA再转RGB
        if image.mode == 'P' and 'transparency' in image.info:
            image = image.convert('RGBA')
        image = image.convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        
        # 预测
        self.eval()
        with torch.no_grad():
            if torch.cuda.is_available():
                image_tensor = image_tensor.cuda()
            
            logits = self(image_tensor)
            probs = F.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()
        
        return {
            'predicted_class': self.class_names[pred_class],
            'predicted_index': pred_class,
            'confidence': confidence,
            'probabilities': {
                self.class_names[i]: probs[0][i].item()
                for i in range(len(self.class_names))
            }
        }


def load_config(config_path: str = 'configs/config.yaml'):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def train_model(config, auto_optimize=True, multi_gpu=True, force_strategy=None, ckpt_path=None):
    """
    训练模型

    Args:
        config: 训练配置
        auto_optimize: 是否启用硬件自动优化
        multi_gpu: 是否启用多GPU训练
        force_strategy: 强制使用的多GPU策略 ('ddp', 'dp', 'deepspeed')
        ckpt_path: checkpoint文件路径，用于恢复训练
    """

    # 硬件检测和配置优化
    if auto_optimize:
        print("🔍 检测硬件配置并优化参数...")
        hardware_detector = HardwareDetector()
        hardware_detector.print_hardware_info()

        optimizer = ConfigOptimizer(hardware_detector)
        original_config = config.copy()
        config = optimizer.optimize_training_config(config)

        # 打印优化报告
        report = optimizer.generate_optimization_report(original_config, config)
        print(f"\n{report}")

        # 保存优化后的配置
        optimized_config_path = f"config_optimized_{hardware_detector.get_hardware_tier()}.yaml"
        with open(optimized_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
        print(f"\n💾 优化后配置已保存: {optimized_config_path}")

    # 多GPU配置
    multi_gpu_trainer = None
    if multi_gpu:
        print("🚀 初始化多GPU训练配置...")
        multi_gpu_trainer = MultiGPUTrainer(config, force_strategy=force_strategy)
        # 更新数据模块配置以适应多GPU
        if 'data' not in config:
            config['data'] = {}
        config['data'] = multi_gpu_trainer.update_dataloader_config(config.get('data', {}))
        config['data']['batch_size'] = multi_gpu_trainer.gpu_config['batch_size_per_gpu']

        print(f"✅ 多GPU配置完成:")
        print(f"   - GPU数量: {multi_gpu_trainer.gpu_config['devices']}")
        print(f"   - 训练策略: {multi_gpu_trainer.gpu_config['strategy']}")
        print(f"   - 每GPU批次大小: {multi_gpu_trainer.gpu_config['batch_size_per_gpu']}")
        print(f"   - 总批次大小: {multi_gpu_trainer.gpu_config['total_batch_size']}")
        print(f"   - 混合精度: {'启用' if multi_gpu_trainer.gpu_config['precision'] == 16 else '禁用'}")
    else:
        print("📱 使用单GPU/CPU训练模式")

    # 创建数据模块
    data_module = ProductDataModule(
        data_dir=config['data']['data_dir'],
        batch_size=config['data']['batch_size'],
        image_size=config['data']['image_size'],
        num_workers=config['data']['num_workers'],
        auto_split=config['data']['auto_split'],
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split'],
        test_split=config['data']['test_split'],
        classification_mode=config['data'].get('classification_mode', 'main_category'),
        nested_structure=config['data'].get('nested_structure', False),
        imbalance_config=config.get('imbalance', {})
    )
    
    # 准备数据
    data_module.prepare_data()
    data_module.setup()
    data_module.print_data_info()
    
    # 更新配置中的类别信息
    config['model']['num_classes'] = data_module.num_classes
    if 'classes' not in config:
        config['classes'] = {}
    config['classes']['names'] = data_module.class_names
    
    # 创建模型
    enable_dynamic = auto_optimize and config.get('advanced', {}).get('enable_dynamic_adjustment', False)
    model = ProductClassifier(config, enable_dynamic_adjustment=enable_dynamic)

    # 设置平衡的损失函数
    balanced_loss = data_module.get_balanced_loss()
    if balanced_loss is not None:
        model.set_balanced_loss(balanced_loss)
    
    # 设置回调函数
    checkpoint_config = config.get('checkpointing', {})
    
    # 生成包含日期的文件名
    current_date = datetime.now().strftime("%Y%m%d")
    base_filename = checkpoint_config.get('filename', 'best-{epoch:02d}-{val_acc:.3f}')
    filename_with_date = f"{current_date}-{base_filename}"
    
    callbacks = [
        ModelCheckpoint(
            monitor=checkpoint_config.get('monitor', 'val_acc'),
            mode=checkpoint_config.get('mode', 'max'),
            save_top_k=checkpoint_config.get('save_top_k', 1),
            filename=filename_with_date,
            save_last=checkpoint_config.get('save_last', True),
            dirpath=checkpoint_config.get('dirpath', None),  # 使用配置中的路径
            save_weights_only=checkpoint_config.get('save_weights_only', False),
            auto_insert_metric_name=False
        ),
        EarlyStopping(
            monitor='val_acc',
            mode='max',
            patience=config['training']['patience'],
            verbose=True
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    # 设置日志记录器
    logger = TensorBoardLogger(
        config['logging']['log_dir'], 
        name=config['logging']['experiment_name']
    )
    
    # 创建训练器 - 支持多GPU
    if multi_gpu_trainer:
        trainer = multi_gpu_trainer.create_trainer(callbacks=callbacks, logger=logger)
    else:
        # 单GPU/CPU训练器
        trainer = pl.Trainer(
            max_epochs=config['training']['max_epochs'],
            callbacks=callbacks,
            logger=logger,
            accelerator='auto',
            devices='auto',
            precision=16 if config['training']['mixed_precision'] else 32,
            log_every_n_steps=10,
            val_check_interval=0.5,
            gradient_clip_val=config['training']['gradient_clip_val']
        )
    
    # 开始训练
    if ckpt_path:
        print(f"从checkpoint恢复训练: {ckpt_path}")
        trainer.fit(model, data_module, ckpt_path=ckpt_path)
    else:
        print("开始新的训练...")
        trainer.fit(model, data_module)
    
    # 找到并重命名最佳模型（支持分布式训练）
    print("正在查找最佳模型...")
    best_model_path = find_and_rename_best_model()

    # 测试最佳模型（只在主进程中执行测试）
    should_test = True
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            should_test = (dist.get_rank() == 0)
    except ImportError:
        pass
    
    if should_test:
        print("测试最佳模型...")
        try:
            if best_model_path and os.path.exists(best_model_path):
                trainer.test(model, data_module, ckpt_path=best_model_path)
                print(f"✅ 最佳模型测试完成: {best_model_path}")
            else:
                print("⚠️ 未找到最佳模型checkpoint，跳过测试")
        except Exception as e:
            print(f"⚠️ 模型测试失败: {e}")
            print("训练已完成，但测试阶段出现问题")
    else:
        print(f"⏳ [Rank {dist.get_rank()}] 跳过测试阶段（仅在主进程中执行）")

    print("训练完成!")
    if best_model_path:
        print(f"最佳模型保存在: {best_model_path}")
    else:
        print("最佳模型路径未找到")
    
    return model, trainer


def find_and_rename_best_model():
    """
    找到准确率最高的checkpoint文件并重命名为best模型
    支持分布式训练，只在主进程（rank 0）中执行模型查找和重命名

    Returns:
        str: 最佳模型的路径，如果未找到则返回None
    """
    import glob
    import shutil
    import time
    
    # 检查是否在分布式训练中
    is_distributed = False
    is_main_process = True
    
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            is_distributed = True
            is_main_process = (dist.get_rank() == 0)
    except ImportError:
        pass
    
    checkpoint_dir = "model/checkpoints"
    best_model_path = None
    
    if is_main_process:
        # 只在主进程中执行模型查找和重命名
        print("🔍 [主进程] 查找最佳模型...")
        
        # 查找所有checkpoint文件（支持多种文件名格式）
        ckpt_patterns = [
            "a10-v2-*.ckpt",
            "best-*.ckpt", 
            "enhanced-clothes-*.ckpt",
            "*.ckpt"
        ]
        
        ckpt_files = []
        for pattern in ckpt_patterns:
            files = glob.glob(os.path.join(checkpoint_dir, pattern))
            ckpt_files.extend(files)
        
        # 去除重复文件
        ckpt_files = list(set(ckpt_files))
        
        if not ckpt_files:
            print("❌ [主进程] 未找到任何checkpoint文件")
            best_model_path = None
        else:
            print(f"📁 [主进程] 找到 {len(ckpt_files)} 个checkpoint文件")

            best_acc = 0.0
            best_file = None
            best_epoch = 0

            for ckpt_file in ckpt_files:
                # 从文件名提取准确率
                filename = os.path.basename(ckpt_file)
                try:
                    # 支持多种文件名格式
                    acc = None
                    epoch = None
                    
                    # 格式1: a10-v2-{epoch}-{acc}.ckpt
                    if 'a10-v2-' in filename:
                        parts = filename.replace('.ckpt', '').split('-')
                        if len(parts) >= 4:
                            acc = float(parts[-1])
                            epoch = int(parts[-2])
                    
                    # 格式2: best-{epoch}-{acc}.ckpt 或 enhanced-clothes-{epoch}-{acc}.ckpt
                    elif '-' in filename and filename.count('-') >= 2:
                        parts = filename.replace('.ckpt', '').split('-')
                        if len(parts) >= 3:
                            try:
                                acc = float(parts[-1])
                                epoch = int(parts[-2])
                            except ValueError:
                                # 如果最后两部分不是数字，尝试其他组合
                                for i in range(len(parts)-1, 0, -1):
                                    try:
                                        acc = float(parts[i])
                                        epoch = int(parts[i-1])
                                        break
                                    except ValueError:
                                        continue
                    
                    if acc is not None and epoch is not None:
                        print(f"  📄 [主进程] Epoch {epoch}: 准确率 {acc:.3f}")
                        
                        if acc > best_acc:
                            best_acc = acc
                            best_file = ckpt_file
                            best_epoch = epoch
                    else:
                        print(f"  ⚠️  [主进程] 无法解析文件名: {filename}")
                        
                except (ValueError, IndexError) as e:
                    print(f"  ⚠️  [主进程] 无法解析文件名: {filename}, 错误: {e}")

            if best_file:
                # 创建最佳模型的新路径
                current_date = datetime.now().strftime("%Y%m%d")
                best_model_name = f"best_model_epoch_{best_epoch}_acc_{best_acc:.3f}_{current_date}.ckpt"
                best_model_path = os.path.join(checkpoint_dir, best_model_name)

                try:
                    # 确保checkpoint目录存在
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    # 复制最佳模型
                    shutil.copy2(best_file, best_model_path)
                    print(f"🏆 [主进程] 最佳模型已保存: {best_model_path}")
                    print(f"   原文件: {best_file}")
                    print(f"   Epoch: {best_epoch}, 准确率: {best_acc:.3f}")

                    # 同时创建一个简单的best.ckpt链接
                    simple_best_path = os.path.join(checkpoint_dir, "best.ckpt")
                    if os.path.exists(simple_best_path):
                        os.remove(simple_best_path)
                    shutil.copy2(best_file, simple_best_path)
                    print(f"🔗 [主进程] 同时创建简化链接: {simple_best_path}")

                except Exception as e:
                    print(f"❌ [主进程] 复制最佳模型失败: {e}")
                    best_model_path = best_file
            else:
                print("❌ [主进程] 无法确定最佳模型")
                best_model_path = None
        
        # 创建状态文件，告知其他进程结果
        if is_distributed:
            status_file = os.path.join(checkpoint_dir, "best_model_status.txt")
            try:
                with open(status_file, 'w') as f:
                    f.write(best_model_path if best_model_path else "None")
                print(f"📝 [主进程] 已写入状态文件: {status_file}")
            except Exception as e:
                print(f"⚠️ [主进程] 无法写入状态文件: {e}")
    
    else:
        # 非主进程等待主进程完成
        print(f"⏳ [Rank {dist.get_rank()}] 等待主进程完成模型查找...")
        
        status_file = os.path.join(checkpoint_dir, "best_model_status.txt")
        max_wait_time = 60  # 最多等待60秒
        wait_time = 0
        
        while wait_time < max_wait_time:
            if os.path.exists(status_file):
                try:
                    with open(status_file, 'r') as f:
                        result = f.read().strip()
                    best_model_path = result if result != "None" else None
                    print(f"📖 [Rank {dist.get_rank()}] 从状态文件读取结果: {best_model_path}")
                    break
                except Exception as e:
                    print(f"⚠️ [Rank {dist.get_rank()}] 读取状态文件失败: {e}")
            
            time.sleep(1)
            wait_time += 1
        
        if wait_time >= max_wait_time:
            print(f"⏰ [Rank {dist.get_rank()}] 等待超时，使用默认路径")
            best_model_path = None
    
    # 分布式同步屏障
    if is_distributed:
        try:
            dist.barrier()
            print(f"🔄 [Rank {dist.get_rank() if dist.is_initialized() else 'Single'}] 同步完成")
        except Exception as e:
            print(f"⚠️ 分布式同步失败: {e}")
    
    return best_model_path


def inference_example(model_path: str, image_path: str, config_path: str = 'config.yaml'):
    """推理示例"""
    
    # 加载配置
    config = load_config(config_path)
    
    # 加载模型
    model = ProductClassifier.load_from_checkpoint(model_path, config=config)
    
    # 预测
    result = model.predict_single(image_path)
    
    print("\n" + "="*50)
    print("预测结果")
    print("="*50)
    print(f"图片路径: {image_path}")
    print(f"预测类别: {result['predicted_class']}")
    print(f"置信度: {result['confidence']:.4f}")
    print("\n各类别概率:")
    for class_name, prob in result['probabilities'].items():
        print(f"  {class_name}: {prob:.4f}")
    print("="*50)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PyTorch Lightning 商品分类')

    # 基本参数
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='配置文件路径')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'inference'], help='运行模式')

    # 硬件优化参数
    parser.add_argument('--auto-optimize', action='store_true', default=True,
                       help='自动检测硬件并优化配置')
    parser.add_argument('--no-auto-optimize', dest='auto_optimize', action='store_false',
                       help='禁用自动硬件优化')
    parser.add_argument('--dynamic-adjustment', action='store_true',
                       help='启用训练过程中的动态配置调整')
    parser.add_argument('--hardware-tier', type=str,
                       choices=['high_end', 'mid_high', 'mid_range', 'low_end', 'cpu_only'],
                       help='手动指定硬件等级')

    # 多GPU参数
    parser.add_argument('--multi-gpu', action='store_true', default=True,
                       help='启用多GPU训练（默认启用）')
    parser.add_argument('--no-multi-gpu', dest='multi_gpu', action='store_false',
                       help='禁用多GPU训练')
    parser.add_argument('--strategy', type=str, default=None,
                       choices=['ddp', 'dp', 'deepspeed'],
                       help='强制使用的多GPU策略')
    parser.add_argument('--gpus', type=int, default=None,
                       help='使用的GPU数量（默认使用所有可用GPU）')

    # 推理参数
    parser.add_argument('--model_path', type=str, help='模型路径（推理模式）')
    parser.add_argument('--image_path', type=str, help='图片路径（推理模式）')
    
    # 训练恢复参数
    parser.add_argument('--ckpt_path', type=str, help='checkpoint文件路径，用于恢复训练')
    
    # 分布式训练参数（由torchrun自动添加）
    parser.add_argument('--local-rank', type=int, default=0, help='分布式训练的本地rank')
    parser.add_argument('--local_rank', type=int, default=0, help='分布式训练的本地rank (torchrun兼容)')
    
    args = parser.parse_args()
    
    # 检查配置文件
    if not Path(args.config).exists():
        print(f"错误: 配置文件不存在: {args.config}")
        return
    
    # 加载配置
    config = load_config(args.config)
    
    if args.mode == 'train':
        # 训练模式

        # 设置GPU使用数量
        if args.gpus:
            import os
            if args.gpus > torch.cuda.device_count():
                print(f"⚠️  指定的GPU数量 {args.gpus} 超过可用数量 {torch.cuda.device_count()}")
                args.gpus = torch.cuda.device_count()
            # 限制可见GPU
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(args.gpus)))

        # 如果指定了硬件等级，使用对应的配置模板
        if args.hardware_tier:
            from ..configs.config_templates import ConfigTemplateGenerator
            generator = ConfigTemplateGenerator()

            template_methods = {
                'high_end': generator.generate_high_end_config,
                'mid_high': generator.generate_mid_high_config,
                'mid_range': generator.generate_mid_range_config,
                'low_end': generator.generate_low_end_config,
                'cpu_only': generator.generate_cpu_only_config
            }

            if args.hardware_tier in template_methods:
                print(f"使用 {args.hardware_tier} 硬件等级的配置模板")
                template_config = template_methods[args.hardware_tier]()

                # 合并用户配置和模板配置
                for key, value in template_config.items():
                    if key not in config:
                        config[key] = value
                    elif isinstance(value, dict) and isinstance(config[key], dict):
                        config[key].update(value)

        # 开始训练，传递多GPU参数
        model, trainer = train_model(
            config,
            auto_optimize=args.auto_optimize,
            multi_gpu=args.multi_gpu,
            force_strategy=args.strategy,
            ckpt_path=args.ckpt_path
        )
        
    elif args.mode == 'inference':
        # 推理模式
        if not args.model_path or not args.image_path:
            print("推理模式需要指定 --model_path 和 --image_path")
            return
        
        inference_example(args.model_path, args.image_path, args.config)


if __name__ == '__main__':
    main()
