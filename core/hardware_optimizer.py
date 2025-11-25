#!/usr/bin/env python3
"""
硬件配置优化器
根据机器硬件配置自动调整训练参数
"""

import psutil
import platform
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import yaml


class HardwareDetector:
    """硬件检测器"""
    
    def __init__(self):
        self.hardware_info = {}
        self.detect_hardware()
    
    def detect_hardware(self):
        """检测硬件配置"""
        print("检测硬件配置...")
        
        # 基本系统信息
        self.hardware_info['system'] = {
            'platform': platform.system(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version()
        }
        
        # CPU信息
        self.hardware_info['cpu'] = {
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'max_frequency': psutil.cpu_freq().max if psutil.cpu_freq() else None,
            'current_frequency': psutil.cpu_freq().current if psutil.cpu_freq() else None
        }
        
        # 内存信息
        memory = psutil.virtual_memory()
        self.hardware_info['memory'] = {
            'total_gb': round(memory.total / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'percent_used': memory.percent
        }
        
        # GPU信息
        self.hardware_info['gpu'] = self.detect_gpu()
        
        # 存储信息
        self.hardware_info['storage'] = self.detect_storage()
    
    def detect_gpu(self) -> Dict[str, Any]:
        """检测GPU信息"""
        gpu_info = {
            'has_cuda': False,
            'has_mps': False,
            'gpu_count': 0,
            'gpu_memory_gb': 0,
            'gpu_names': [],
            'cuda_version': None
        }
        
        try:
            import torch
            
            # 检测CUDA
            if torch.cuda.is_available():
                gpu_info['has_cuda'] = True
                gpu_info['gpu_count'] = torch.cuda.device_count()
                gpu_info['cuda_version'] = torch.version.cuda
                
                for i in range(gpu_info['gpu_count']):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    gpu_info['gpu_names'].append(gpu_name)
                    gpu_info['gpu_memory_gb'] += gpu_memory
                
                gpu_info['gpu_memory_gb'] = round(gpu_info['gpu_memory_gb'], 2)
            
            # 检测MPS (Apple Silicon)
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                gpu_info['has_mps'] = True
                gpu_info['gpu_count'] = 1
                gpu_info['gpu_names'] = ['Apple Silicon GPU']
                # MPS内存通常与系统内存共享
                gpu_info['gpu_memory_gb'] = self.hardware_info['memory']['total_gb']
        
        except ImportError:
            print("⚠️  PyTorch未安装，无法检测GPU信息")
        
        return gpu_info
    
    def detect_storage(self) -> Dict[str, Any]:
        """检测存储信息"""
        try:
            disk_usage = psutil.disk_usage('/')
            return {
                'total_gb': round(disk_usage.total / (1024**3), 2),
                'free_gb': round(disk_usage.free / (1024**3), 2),
                'used_percent': round((disk_usage.used / disk_usage.total) * 100, 2)
            }
        except:
            return {'total_gb': 0, 'free_gb': 0, 'used_percent': 0}
    
    def get_hardware_tier(self) -> str:
        """获取硬件等级"""
        gpu_memory = self.hardware_info['gpu']['gpu_memory_gb']
        system_memory = self.hardware_info['memory']['total_gb']
        cpu_cores = self.hardware_info['cpu']['logical_cores']
        
        # 高端配置
        if gpu_memory >= 16 and system_memory >= 32 and cpu_cores >= 16:
            return 'high_end'
        # 中高端配置
        elif gpu_memory >= 8 and system_memory >= 16 and cpu_cores >= 8:
            return 'mid_high'
        # 中端配置
        elif gpu_memory >= 4 and system_memory >= 8 and cpu_cores >= 4:
            return 'mid_range'
        # 低端配置
        else:
            return 'low_end'
    
    def print_hardware_info(self):
        """打印硬件信息"""
        print("="*60)
        print("硬件配置信息")
        print("="*60)
        
        # 系统信息
        sys_info = self.hardware_info['system']
        print(f"操作系统: {sys_info['platform']}")
        print(f"架构: {sys_info['architecture']}")
        print(f"Python版本: {sys_info['python_version']}")
        
        # CPU信息
        cpu_info = self.hardware_info['cpu']
        print(f"\nCPU:")
        print(f"  物理核心: {cpu_info['physical_cores']}")
        print(f"  逻辑核心: {cpu_info['logical_cores']}")
        if cpu_info['max_frequency']:
            print(f"  最大频率: {cpu_info['max_frequency']:.0f} MHz")
        
        # 内存信息
        mem_info = self.hardware_info['memory']
        print(f"\n内存:")
        print(f"  总内存: {mem_info['total_gb']} GB")
        print(f"  可用内存: {mem_info['available_gb']} GB")
        print(f"  使用率: {mem_info['percent_used']:.1f}%")
        
        # GPU信息
        gpu_info = self.hardware_info['gpu']
        print(f"\nGPU:")
        if gpu_info['has_cuda']:
            print(f"  CUDA可用: ✅")
            print(f"  CUDA版本: {gpu_info['cuda_version']}")
            print(f"  GPU数量: {gpu_info['gpu_count']}")
            print(f"  GPU内存: {gpu_info['gpu_memory_gb']} GB")
            for i, name in enumerate(gpu_info['gpu_names']):
                print(f"  GPU {i}: {name}")
        elif gpu_info['has_mps']:
            print(f"  MPS可用: ✅ (Apple Silicon)")
            print(f"  GPU内存: 与系统内存共享")
        else:
            print(f"  GPU加速: ❌ (仅CPU)")
        
        # 存储信息
        storage_info = self.hardware_info['storage']
        print(f"\n存储:")
        print(f"  总容量: {storage_info['total_gb']} GB")
        print(f"  可用空间: {storage_info['free_gb']} GB")
        print(f"  使用率: {storage_info['used_percent']:.1f}%")
        
        # 硬件等级
        tier = self.get_hardware_tier()
        tier_names = {
            'high_end': '高端',
            'mid_high': '中高端', 
            'mid_range': '中端',
            'low_end': '低端'
        }
        print(f"\n硬件等级: {tier_names.get(tier, tier)}")


class ConfigOptimizer:
    """配置优化器"""
    
    def __init__(self, hardware_detector: HardwareDetector):
        self.hardware = hardware_detector
        self.tier = hardware_detector.get_hardware_tier()
    
    def optimize_training_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """优化训练配置"""
        optimized_config = base_config.copy()
        
        # 根据硬件等级调整参数
        if self.tier == 'high_end':
            optimized_config = self._optimize_high_end(optimized_config)
        elif self.tier == 'mid_high':
            optimized_config = self._optimize_mid_high(optimized_config)
        elif self.tier == 'mid_range':
            optimized_config = self._optimize_mid_range(optimized_config)
        else:  # low_end
            optimized_config = self._optimize_low_end(optimized_config)
        
        # 通用优化
        optimized_config = self._apply_common_optimizations(optimized_config)
        
        return optimized_config
    
    def _optimize_high_end(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """高端硬件优化"""
        # 数据配置
        config['data']['batch_size'] = min(64, config['data'].get('batch_size', 32) * 2)
        config['data']['num_workers'] = min(16, self.hardware.hardware_info['cpu']['logical_cores'])
        
        # 训练配置
        config['training']['mixed_precision'] = True
        config['training']['gradient_clip_val'] = 1.0
        
        # Fine-tune配置
        if 'finetune' in config:
            config['finetune']['training']['max_epochs'] = min(30, 
                config['finetune']['training'].get('max_epochs', 20))
        
        return config
    
    def _optimize_mid_high(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """中高端硬件优化"""
        # 数据配置
        config['data']['batch_size'] = min(48, config['data'].get('batch_size', 32) + 16)
        config['data']['num_workers'] = min(12, self.hardware.hardware_info['cpu']['logical_cores'])
        
        # 训练配置
        config['training']['mixed_precision'] = True
        config['training']['gradient_clip_val'] = 1.0
        
        return config
    
    def _optimize_mid_range(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """中端硬件优化"""
        # 数据配置
        config['data']['batch_size'] = min(32, config['data'].get('batch_size', 32))
        config['data']['num_workers'] = min(8, self.hardware.hardware_info['cpu']['logical_cores'])
        
        # 训练配置
        config['training']['mixed_precision'] = True
        config['training']['gradient_clip_val'] = 0.5
        
        return config
    
    def _optimize_low_end(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """低端硬件优化"""
        # 数据配置
        config['data']['batch_size'] = min(16, config['data'].get('batch_size', 32))
        config['data']['num_workers'] = min(4, max(1, self.hardware.hardware_info['cpu']['logical_cores'] // 2))
        
        # 训练配置
        config['training']['mixed_precision'] = False  # 可能不支持
        config['training']['gradient_clip_val'] = 0.5
        config['training']['max_epochs'] = min(50, config['training'].get('max_epochs', 100))
        
        # Fine-tune配置
        if 'finetune' in config:
            config['finetune']['training']['max_epochs'] = min(15, 
                config['finetune']['training'].get('max_epochs', 20))
            config['finetune']['training']['learning_rate'] *= 0.5  # 降低学习率
        
        return config
    
    def _apply_common_optimizations(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """应用通用优化"""
        # 根据GPU内存调整批次大小
        gpu_memory = self.hardware.hardware_info['gpu']['gpu_memory_gb']
        
        if gpu_memory > 0:
            # 根据GPU内存进一步调整批次大小
            if gpu_memory < 4:
                config['data']['batch_size'] = min(config['data']['batch_size'], 8)
            elif gpu_memory < 8:
                config['data']['batch_size'] = min(config['data']['batch_size'], 16)
            elif gpu_memory < 12:
                config['data']['batch_size'] = min(config['data']['batch_size'], 32)
        else:
            # 仅CPU，大幅降低批次大小
            config['data']['batch_size'] = min(config['data']['batch_size'], 4)
            config['training']['mixed_precision'] = False
        
        # 确保num_workers不超过CPU核心数
        max_workers = max(1, self.hardware.hardware_info['cpu']['logical_cores'] - 1)
        config['data']['num_workers'] = min(config['data']['num_workers'], max_workers)
        
        return config
    
    def generate_optimization_report(self, original_config: Dict[str, Any], 
                                   optimized_config: Dict[str, Any]) -> str:
        """生成优化报告"""
        report = []
        report.append("="*60)
        report.append("配置优化报告")
        report.append("="*60)
        
        report.append(f"硬件等级: {self.tier}")
        report.append(f"GPU内存: {self.hardware.hardware_info['gpu']['gpu_memory_gb']} GB")
        report.append(f"系统内存: {self.hardware.hardware_info['memory']['total_gb']} GB")
        report.append(f"CPU核心: {self.hardware.hardware_info['cpu']['logical_cores']}")
        
        report.append("\n参数调整:")
        
        # 比较关键参数
        key_params = [
            ('data.batch_size', '批次大小'),
            ('data.num_workers', '数据加载线程'),
            ('training.mixed_precision', '混合精度'),
            ('training.max_epochs', '最大轮数'),
            ('training.gradient_clip_val', '梯度裁剪')
        ]
        
        for param_path, param_name in key_params:
            original_val = self._get_nested_value(original_config, param_path)
            optimized_val = self._get_nested_value(optimized_config, param_path)
            
            if original_val != optimized_val:
                report.append(f"  {param_name}: {original_val} → {optimized_val}")
        
        # Fine-tune参数
        if 'finetune' in optimized_config:
            ft_params = [
                ('finetune.training.max_epochs', 'Fine-tune轮数'),
                ('finetune.training.learning_rate', 'Fine-tune学习率')
            ]
            
            for param_path, param_name in ft_params:
                original_val = self._get_nested_value(original_config, param_path)
                optimized_val = self._get_nested_value(optimized_config, param_path)
                
                if original_val != optimized_val:
                    report.append(f"  {param_name}: {original_val} → {optimized_val}")
        
        return "\n".join(report)
    
    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """获取嵌套字典的值"""
        keys = path.split('.')
        value = config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return None


def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], config_path: str):
    """保存配置文件"""
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='硬件配置优化器')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='配置文件路径')
    parser.add_argument('--output', type=str, 
                       help='优化后配置文件输出路径')
    parser.add_argument('--detect-only', action='store_true', 
                       help='仅检测硬件，不优化配置')
    parser.add_argument('--save-hardware-info', action='store_true', 
                       help='保存硬件信息到文件')
    
    args = parser.parse_args()
    
    # 检测硬件
    print("硬件配置优化器")
    detector = HardwareDetector()
    detector.print_hardware_info()
    
    # 保存硬件信息
    if args.save_hardware_info:
        hardware_file = 'hardware_info.json'
        with open(hardware_file, 'w', encoding='utf-8') as f:
            json.dump(detector.hardware_info, f, ensure_ascii=False, indent=2)
        print(f"\n💾 硬件信息已保存: {hardware_file}")
    
    if args.detect_only:
        return
    
    # 加载和优化配置
    try:
        original_config = load_config(args.config)
        print(f"\n📄 加载配置文件: {args.config}")
        
        optimizer = ConfigOptimizer(detector)
        optimized_config = optimizer.optimize_training_config(original_config)
        
        # 生成报告
        report = optimizer.generate_optimization_report(original_config, optimized_config)
        print(f"\n{report}")
        
        # 保存优化后的配置
        output_path = args.output or f"config_optimized_{detector.get_hardware_tier()}.yaml"
        save_config(optimized_config, output_path)
        print(f"\n💾 优化后配置已保存: {output_path}")
        
        print(f"\n🚀 使用优化后的配置:")
        print(f"   python train.py --config {output_path}")
        
    except FileNotFoundError:
        print(f"\n❌ 配置文件不存在: {args.config}")
    except Exception as e:
        print(f"\n❌ 配置优化失败: {e}")


class DynamicConfigAdjuster:
    """动态配置调整器"""

    def __init__(self, initial_config: Dict[str, Any]):
        self.config = initial_config.copy()
        self.performance_history = []
        self.adjustment_count = 0
        self.max_adjustments = 3

    def monitor_training_performance(self, epoch: int, batch_time: float,
                                   memory_usage: float, gpu_utilization: float = None) -> bool:
        """
        监控训练性能并决定是否需要调整

        Args:
            epoch: 当前轮数
            batch_time: 批次处理时间（秒）
            memory_usage: 内存使用率（0-1）
            gpu_utilization: GPU使用率（0-1）

        Returns:
            是否需要调整配置
        """
        performance = {
            'epoch': epoch,
            'batch_time': batch_time,
            'memory_usage': memory_usage,
            'gpu_utilization': gpu_utilization,
            'timestamp': time.time()
        }

        self.performance_history.append(performance)

        # 保留最近10个记录
        if len(self.performance_history) > 10:
            self.performance_history.pop(0)

        # 检查是否需要调整
        if len(self.performance_history) >= 3 and self.adjustment_count < self.max_adjustments:
            return self._should_adjust()

        return False

    def _should_adjust(self) -> bool:
        """判断是否应该调整配置"""
        recent_performance = self.performance_history[-3:]

        # 检查内存使用率
        avg_memory = sum(p['memory_usage'] for p in recent_performance) / len(recent_performance)
        if avg_memory > 0.9:  # 内存使用率超过90%
            return True

        # 检查批次处理时间
        avg_batch_time = sum(p['batch_time'] for p in recent_performance) / len(recent_performance)
        if avg_batch_time > 10:  # 批次处理时间超过10秒
            return True

        # 检查GPU使用率（如果有）
        gpu_utils = [p['gpu_utilization'] for p in recent_performance if p['gpu_utilization'] is not None]
        if gpu_utils:
            avg_gpu_util = sum(gpu_utils) / len(gpu_utils)
            if avg_gpu_util < 0.3:  # GPU使用率低于30%
                return True

        return False

    def adjust_config(self) -> Dict[str, Any]:
        """调整配置参数"""
        if self.adjustment_count >= self.max_adjustments:
            return self.config

        recent_performance = self.performance_history[-3:]
        avg_memory = sum(p['memory_usage'] for p in recent_performance) / len(recent_performance)
        avg_batch_time = sum(p['batch_time'] for p in recent_performance) / len(recent_performance)

        adjustments = []

        # 内存压力大，减少批次大小
        if avg_memory > 0.9:
            old_batch_size = self.config['data']['batch_size']
            new_batch_size = max(1, old_batch_size // 2)
            self.config['data']['batch_size'] = new_batch_size
            adjustments.append(f"批次大小: {old_batch_size} → {new_batch_size} (内存压力)")

        # 处理时间长，减少数据加载线程
        if avg_batch_time > 10:
            old_workers = self.config['data']['num_workers']
            new_workers = max(1, old_workers - 2)
            self.config['data']['num_workers'] = new_workers
            adjustments.append(f"数据线程: {old_workers} → {new_workers} (处理时间长)")

        # GPU使用率低，可能可以增加批次大小
        gpu_utils = [p['gpu_utilization'] for p in recent_performance if p['gpu_utilization'] is not None]
        if gpu_utils and avg_memory < 0.7:
            avg_gpu_util = sum(gpu_utils) / len(gpu_utils)
            if avg_gpu_util < 0.3:
                old_batch_size = self.config['data']['batch_size']
                new_batch_size = min(64, old_batch_size + 4)
                self.config['data']['batch_size'] = new_batch_size
                adjustments.append(f"批次大小: {old_batch_size} → {new_batch_size} (GPU利用率低)")

        if adjustments:
            self.adjustment_count += 1
            print(f"\n🔧 动态调整配置 (第{self.adjustment_count}次):")
            for adj in adjustments:
                print(f"  {adj}")

        return self.config


import time


if __name__ == '__main__':
    main()
