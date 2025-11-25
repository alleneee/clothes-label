#!/usr/bin/env python3
"""
HuggingFace Evaluate 评估模块
使用标准化指标评估衣服分类模型性能
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
import json
from datetime import datetime

try:
    import evaluate
    HF_EVALUATE_AVAILABLE = True
except ImportError:
    HF_EVALUATE_AVAILABLE = False
    logging.warning("HuggingFace Evaluate not available. Install with: pip install evaluate")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HFEvaluateModule:
    """HuggingFace Evaluate 评估模块"""
    
    def __init__(
        self,
        class_names: List[str],
        metrics_to_compute: Optional[List[str]] = None,
        output_dir: str = "evaluation_results",
        save_results: bool = True
    ):
        """
        初始化评估模块
        
        Args:
            class_names: 类别名称列表
            metrics_to_compute: 要计算的指标列表
            output_dir: 结果输出目录
            save_results: 是否保存结果
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.output_dir = Path(output_dir)
        self.save_results = save_results
        
        if self.save_results:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 默认指标
        if metrics_to_compute is None:
            metrics_to_compute = [
                "accuracy",
                "precision", 
                "recall", 
                "f1",
                "matthews_correlation"
            ]
        
        self.metrics_to_compute = metrics_to_compute
        
        # 初始化HF Evaluate指标
        self.hf_metrics = {}
        if HF_EVALUATE_AVAILABLE:
            self._initialize_hf_metrics()
        else:
            logger.warning("使用fallback评估方法")
    
    def _initialize_hf_metrics(self):
        """初始化HuggingFace Evaluate指标"""
        try:
            for metric_name in self.metrics_to_compute:
                if metric_name in ["precision", "recall", "f1"]:
                    # 这些指标需要指定平均方式
                    self.hf_metrics[metric_name] = evaluate.load(metric_name)
                else:
                    self.hf_metrics[metric_name] = evaluate.load(metric_name)
            
            logger.info(f"✅ 成功初始化HF Evaluate指标: {list(self.hf_metrics.keys())}")
        except Exception as e:
            logger.error(f"❌ 初始化HF Evaluate指标失败: {e}")
            HF_EVALUATE_AVAILABLE = False
    
    def compute_metrics(
        self,
        predictions: np.ndarray,
        references: np.ndarray,
        probabilities: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        计算评估指标
        
        Args:
            predictions: 预测结果
            references: 真实标签
            probabilities: 预测概率（可选）
            
        Returns:
            计算得到的指标字典
        """
        if HF_EVALUATE_AVAILABLE:
            return self._compute_hf_metrics(predictions, references, probabilities)
        else:
            return self._compute_fallback_metrics(predictions, references, probabilities)
    
    def _compute_hf_metrics(
        self,
        predictions: np.ndarray,
        references: np.ndarray,
        probabilities: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """使用HuggingFace Evaluate计算指标"""
        results = {}
        
        for metric_name, metric in self.hf_metrics.items():
            try:
                if metric_name == "accuracy":
                    result = metric.compute(predictions=predictions, references=references)
                    results[metric_name] = result['accuracy']
                
                elif metric_name in ["precision", "recall", "f1"]:
                    # 计算macro和micro平均
                    for avg_type in ["macro", "micro"]:
                        result = metric.compute(
                            predictions=predictions, 
                            references=references,
                            average=avg_type
                        )
                        results[f"{metric_name}_{avg_type}"] = result[metric_name]
                    
                    # 计算每个类别的指标
                    result_per_class = metric.compute(
                        predictions=predictions, 
                        references=references,
                        average=None
                    )
                    for i, class_name in enumerate(self.class_names):
                        if i < len(result_per_class[metric_name]):
                            results[f"{metric_name}_{class_name}"] = result_per_class[metric_name][i]
                
                elif metric_name == "matthews_correlation":
                    result = metric.compute(predictions=predictions, references=references)
                    results[metric_name] = result['matthews_correlation']
                
                else:
                    # 通用指标计算
                    result = metric.compute(predictions=predictions, references=references)
                    results[metric_name] = result.get(metric_name, result)
                    
            except Exception as e:
                logger.error(f"❌ 计算指标 {metric_name} 失败: {e}")
                continue
        
        # 添加额外的自定义指标
        results.update(self._compute_custom_metrics(predictions, references, probabilities))
        
        return results
    
    def _compute_fallback_metrics(
        self,
        predictions: np.ndarray,
        references: np.ndarray,
        probabilities: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """使用sklearn作为fallback计算指标"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            classification_report, matthews_corrcoef
        )
        
        results = {}
        
        try:
            # 基础指标
            results['accuracy'] = accuracy_score(references, predictions)
            results['precision_macro'] = precision_score(references, predictions, average='macro', zero_division=0)
            results['precision_micro'] = precision_score(references, predictions, average='micro', zero_division=0)
            results['recall_macro'] = recall_score(references, predictions, average='macro', zero_division=0)
            results['recall_micro'] = recall_score(references, predictions, average='micro', zero_division=0)
            results['f1_macro'] = f1_score(references, predictions, average='macro', zero_division=0)
            results['f1_micro'] = f1_score(references, predictions, average='micro', zero_division=0)
            results['matthews_correlation'] = matthews_corrcoef(references, predictions)
            
            # 每个类别的指标
            precision_per_class = precision_score(references, predictions, average=None, zero_division=0)
            recall_per_class = recall_score(references, predictions, average=None, zero_division=0)
            f1_per_class = f1_score(references, predictions, average=None, zero_division=0)
            
            for i, class_name in enumerate(self.class_names):
                if i < len(precision_per_class):
                    results[f'precision_{class_name}'] = precision_per_class[i]
                    results[f'recall_{class_name}'] = recall_per_class[i]
                    results[f'f1_{class_name}'] = f1_per_class[i]
            
            # 添加额外的自定义指标
            results.update(self._compute_custom_metrics(predictions, references, probabilities))
            
        except Exception as e:
            logger.error(f"❌ Fallback指标计算失败: {e}")
            results['accuracy'] = accuracy_score(references, predictions)
        
        return results
    
    def _compute_custom_metrics(
        self,
        predictions: np.ndarray,
        references: np.ndarray,
        probabilities: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """计算自定义指标"""
        custom_metrics = {}
        
        # 每个类别的准确率
        for i, class_name in enumerate(self.class_names):
            class_mask = references == i
            if np.sum(class_mask) > 0:
                class_acc = np.sum(predictions[class_mask] == references[class_mask]) / np.sum(class_mask)
                custom_metrics[f'accuracy_{class_name}'] = class_acc
        
        # 针对身体分类的特殊指标
        if any('全身' in name for name in self.class_names) and any('半身' in name for name in self.class_names):
            custom_metrics.update(self._compute_body_classification_metrics(predictions, references))
        
        return custom_metrics
    
    def _compute_body_classification_metrics(
        self,
        predictions: np.ndarray,
        references: np.ndarray
    ) -> Dict[str, Any]:
        """计算身体分类特殊指标"""
        metrics = {}
        
        # 找到全身和半身类别的索引
        full_body_idx = None
        half_body_idx = None
        
        for i, class_name in enumerate(self.class_names):
            if '全身' in class_name:
                full_body_idx = i
            elif '半身' in class_name:
                half_body_idx = i
        
        if full_body_idx is not None and half_body_idx is not None:
            # 全身模特误分类为半身的比例
            full_body_mask = references == full_body_idx
            if np.sum(full_body_mask) > 0:
                full_to_half_errors = np.sum(predictions[full_body_mask] == half_body_idx)
                metrics['full_body_to_half_body_error_rate'] = full_to_half_errors / np.sum(full_body_mask)
            
            # 半身模特误分类为全身的比例
            half_body_mask = references == half_body_idx
            if np.sum(half_body_mask) > 0:
                half_to_full_errors = np.sum(predictions[half_body_mask] == full_body_idx)
                metrics['half_body_to_full_body_error_rate'] = half_to_full_errors / np.sum(half_body_mask)
        
        return metrics
    
    def plot_confusion_matrix(
        self,
        predictions: np.ndarray,
        references: np.ndarray,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """绘制混淆矩阵"""
        cm = confusion_matrix(references, predictions)
        
        # 创建图像
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 绘制混淆矩阵
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax
        )
        
        ax.set_title('Confusion Matrix', fontsize=16, pad=20)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        # 旋转标签以避免重叠
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_evaluation_report(
        self,
        predictions: np.ndarray,
        references: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        dataset_name: str = "test"
    ) -> Dict[str, Any]:
        """生成完整的评估报告"""
        logger.info(f"📊 生成 {dataset_name} 集评估报告...")
        
        # 计算指标
        metrics = self.compute_metrics(predictions, references, probabilities)
        
        # 创建报告
        report = {
            'dataset': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(predictions),
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'metrics': metrics
        }
        
        # 保存结果
        if self.save_results:
            # 保存指标
            metrics_file = self.output_dir / f"{dataset_name}_metrics.json"
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            # 保存混淆矩阵
            cm_file = self.output_dir / f"{dataset_name}_confusion_matrix.png"
            self.plot_confusion_matrix(predictions, references, str(cm_file))
            
            logger.info(f"✅ 评估结果已保存到 {self.output_dir}")
        
        return report
    
    def print_evaluation_summary(self, report: Dict[str, Any]):
        """打印评估摘要"""
        print("\n" + "="*60)
        print(f"📊 {report['dataset'].upper()} 集评估报告")
        print("="*60)
        print(f"样本数量: {report['num_samples']}")
        print(f"类别数量: {report['num_classes']}")
        print("-"*60)
        
        metrics = report['metrics']
        
        # 主要指标
        print("📈 主要指标:")
        main_metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'matthews_correlation']
        for metric in main_metrics:
            if metric in metrics:
                print(f"  {metric:20}: {metrics[metric]:.4f}")
        
        # 身体分类特殊指标
        body_metrics = [k for k in metrics.keys() if 'body' in k and 'error_rate' in k]
        if body_metrics:
            print("\n🎯 身体分类错误率:")
            for metric in body_metrics:
                print(f"  {metric:35}: {metrics[metric]:.4f}")
        
        # 每个类别的F1得分
        print("\n📊 各类别F1得分:")
        for class_name in self.class_names:
            f1_key = f'f1_{class_name}'
            if f1_key in metrics:
                print(f"  {class_name:20}: {metrics[f1_key]:.4f}")
        
        print("="*60)


def create_hf_evaluator(class_names: List[str], **kwargs) -> HFEvaluateModule:
    """创建HF评估器的工厂函数"""
    return HFEvaluateModule(class_names, **kwargs)


if __name__ == "__main__":
    # 示例用法
    class_names = [
        "logo", "下摆", "侧面", "其他", "口袋", "正面", 
        "正面全身模特", "正面半身模特", "背面", "背面模特", "袖口", "领口"
    ]
    
    # 创建评估器
    evaluator = HFEvaluateModule(class_names)
    
    # 模拟预测结果
    np.random.seed(42)
    predictions = np.random.randint(0, len(class_names), 100)
    references = np.random.randint(0, len(class_names), 100)
    
    # 生成评估报告
    report = evaluator.generate_evaluation_report(predictions, references)
    evaluator.print_evaluation_summary(report)
