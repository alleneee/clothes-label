#!/usr/bin/env python3
"""
集成HuggingFace Evaluate的训练模块
优化评估方式，提供更专业的指标计算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging

from .hf_evaluate_module import HFEvaluateModule
from .train import ProductClassifier  # 继承原有的训练类

logger = logging.getLogger(__name__)


class ProductClassifierWithHFEvaluate(ProductClassifier):
    """集成HuggingFace Evaluate的商品分类器"""
    
    def __init__(self, config, enable_dynamic_adjustment=False):
        super().__init__(config, enable_dynamic_adjustment)
        
        # 初始化HF Evaluate模块
        self.hf_evaluator = HFEvaluateModule(
            class_names=self.class_names,
            output_dir="evaluation_results",
            save_results=True
        )
        
        # 存储预测结果用于详细评估
        self.validation_predictions = []
        self.validation_references = []
        self.validation_probabilities = []
        
        self.test_predictions = []
        self.test_references = []
        self.test_probabilities = []
        
        logger.info("✅ 已集成HuggingFace Evaluate评估模块")
    
    def validation_step(self, batch, batch_idx):
        """验证步骤 - 集成HF Evaluate"""
        # 处理不同的batch格式
        if isinstance(batch, dict):
            x, y = batch['image'], batch['label']
        else:
            x, y = batch
            
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # 计算预测结果和概率
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == y).float() / len(y)
        
        # 记录基础指标
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        
        # 收集预测结果用于详细评估
        self.validation_predictions.extend(preds.cpu().numpy())
        self.validation_references.extend(y.cpu().numpy())
        self.validation_probabilities.extend(probs.cpu().numpy())
        
        return {'val_loss': loss, 'val_acc': acc}
    
    def on_validation_epoch_end(self):
        """验证轮次结束时使用HF Evaluate计算详细指标"""
        if not self.validation_predictions:
            return
        
        try:
            # 转换为numpy数组
            predictions = np.array(self.validation_predictions)
            references = np.array(self.validation_references)
            probabilities = np.array(self.validation_probabilities)
            
            # 使用HF Evaluate计算指标
            metrics = self.hf_evaluator.compute_metrics(
                predictions=predictions,
                references=references,
                probabilities=probabilities
            )
            
            # 记录主要指标到Lightning
            main_metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
            for metric_name in main_metrics:
                if metric_name in metrics:
                    self.log(f'val_{metric_name}', metrics[metric_name], on_epoch=True)
            
            # 记录每个类别的F1得分
            for class_name in self.class_names:
                f1_key = f'f1_{class_name}'
                if f1_key in metrics:
                    self.log(f'val_f1_{class_name}', metrics[f1_key], on_epoch=True)
            
            # 记录身体分类特殊指标
            body_metrics = [k for k in metrics.keys() if 'body' in k and 'error_rate' in k]
            for metric_name in body_metrics:
                self.log(f'val_{metric_name}', metrics[metric_name], on_epoch=True)
                
        except Exception as e:
            logger.error(f"❌ 验证指标计算失败: {e}")
        
        # 清空累积的预测结果
        self.validation_predictions.clear()
        self.validation_references.clear()
        self.validation_probabilities.clear()
    
    def test_step(self, batch, batch_idx):
        """测试步骤 - 集成HF Evaluate"""
        # 处理不同的batch格式
        if isinstance(batch, dict):
            x, y = batch['image'], batch['label']
        else:
            x, y = batch
            
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # 计算预测结果和概率
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == y).float() / len(y)
        
        # 记录基础指标
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', acc, on_epoch=True)
        
        # 收集预测结果用于详细评估
        self.test_predictions.extend(preds.cpu().numpy())
        self.test_references.extend(y.cpu().numpy())
        self.test_probabilities.extend(probs.cpu().numpy())
        
        return {'test_loss': loss, 'test_acc': acc}
    
    def on_test_epoch_end(self):
        """测试结束时使用HF Evaluate生成详细报告"""
        if not self.test_predictions:
            return
        
        try:
            # 转换为numpy数组
            predictions = np.array(self.test_predictions)
            references = np.array(self.test_references)
            probabilities = np.array(self.test_probabilities)
            
            # 使用HF Evaluate生成完整报告
            report = self.hf_evaluator.generate_evaluation_report(
                predictions=predictions,
                references=references,
                probabilities=probabilities,
                dataset_name="test"
            )
            
            # 打印评估摘要
            self.hf_evaluator.print_evaluation_summary(report)
            
            # 记录主要指标到Lightning
            metrics = report['metrics']
            main_metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'matthews_correlation']
            for metric_name in main_metrics:
                if metric_name in metrics:
                    self.log(f'test_{metric_name}', metrics[metric_name], on_epoch=True)
            
            # 记录每个类别的指标
            for class_name in self.class_names:
                for metric_type in ['f1', 'precision', 'recall', 'accuracy']:
                    metric_key = f'{metric_type}_{class_name}'
                    if metric_key in metrics:
                        self.log(f'test_{metric_key}', metrics[metric_key], on_epoch=True)
            
            # 记录身体分类特殊指标
            body_metrics = [k for k in metrics.keys() if 'body' in k and 'error_rate' in k]
            for metric_name in body_metrics:
                self.log(f'test_{metric_name}', metrics[metric_name], on_epoch=True)
                
        except Exception as e:
            logger.error(f"❌ 测试指标计算失败: {e}")
        
        # 清空累积的预测结果
        self.test_predictions.clear()
        self.test_references.clear()
        self.test_probabilities.clear()
    
    def get_current_metrics(self, dataset_type: str = "validation") -> Dict[str, float]:
        """获取当前指标（用于外部调用）"""
        if dataset_type == "validation":
            if not self.validation_predictions:
                return {}
            
            predictions = np.array(self.validation_predictions)
            references = np.array(self.validation_references)
            probabilities = np.array(self.validation_probabilities)
            
        elif dataset_type == "test":
            if not self.test_predictions:
                return {}
                
            predictions = np.array(self.test_predictions)
            references = np.array(self.test_references)
            probabilities = np.array(self.test_probabilities)
        else:
            return {}
        
        try:
            metrics = self.hf_evaluator.compute_metrics(
                predictions=predictions,
                references=references,
                probabilities=probabilities
            )
            return metrics
        except Exception as e:
            logger.error(f"❌ 获取当前指标失败: {e}")
            return {}
    
    def evaluate_predictions(
        self,
        predictions: np.ndarray,
        references: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        dataset_name: str = "custom"
    ) -> Dict[str, Any]:
        """评估自定义预测结果"""
        return self.hf_evaluator.generate_evaluation_report(
            predictions=predictions,
            references=references,
            probabilities=probabilities,
            dataset_name=dataset_name
        )


def create_hf_evaluate_classifier(config: Dict[str, Any]) -> ProductClassifierWithHFEvaluate:
    """创建集成HF Evaluate的分类器"""
    return ProductClassifierWithHFEvaluate(config)


# 兼容性函数
def get_classifier_class():
    """获取分类器类"""
    return ProductClassifierWithHFEvaluate


if __name__ == "__main__":
    # 测试用法
    import yaml
    
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建分类器
    classifier = create_hf_evaluate_classifier(config)
    
    print("✅ 集成HF Evaluate的分类器创建成功")
    print(f"📊 支持的评估指标: {classifier.hf_evaluator.metrics_to_compute}")
    print(f"🎯 类别数量: {len(classifier.class_names)}")
    print(f"📁 评估结果保存目录: {classifier.hf_evaluator.output_dir}")
