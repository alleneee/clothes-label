# PyTorch Lightning 商品分类系统

基于 PyTorch Lightning 的极简商品图片分类系统，支持单GPU/多GPU训练，功能完整。

## 🆕 新功能：OSS图片预测接口

### 简化的OSS预测
新增了专为OSS图片设计的简化预测接口，使用两步流程：

```python
# 步骤1: 从OSS获取图片
image_data = get_oss_image_base64("products/shirt_001.jpg")

# 步骤2: 调用简化预测接口
response = requests.post('http://localhost:8000/predict/simple', json={
    'image_data': image_data,
    'pic_name': 'products/shirt_001.jpg'
})
```

**主要特性：**
- ✅ **简化输出**：仅返回置信度最高的预测结果
- ✅ **两步流程**：先获取OSS图片，再进行预测
- ✅ **格式支持**：支持AVIF、JPEG、PNG等格式
- ✅ **灵活集成**：可与现有OSS服务无缝集成

**响应示例：**
```json
{
    "success": true,
    "message": "图片预测成功",
    "data": {
        "predicted_class": "shirt",
        "predicted_index": 0,
        "confidence": 0.9876,
        "pic_name": "products/shirt_001.jpg",
        "processing_time": 0.234
    }
}
```

📖 **详细文档**：[OSS预测API文档](oss_prediction_api.md)

## ✨ 特点

- 🚀 **极简易用**: 一行命令开始训练，自动检测硬件配置
- ⚡ **多GPU加速**: 支持DDP分布式训练，显著提升训练速度
- 📊 **智能监控**: TensorBoard集成，训练过程可视化
- 🔄 **灵活数据**: 支持多种数据格式，自动拆分和预处理
- 🎯 **高精度**: 基于EfficientNet v2，达到工业级精度
- 🛠️ **自动优化**: 硬件自适应，批次大小和学习率自动调整

## 🚀 快速开始

### 1. 一键安装（Linux服务器 - Python 3.11 + CUDA 12.4）
```bash
# 克隆项目
git clone <your-repo-url>
cd model-train

# Python 3.11专用安装（推荐）
bash install_py311.sh

# 检查环境兼容性
python scripts/check_py311_compatibility.py

# 使用生成的启动脚本
source activate.sh        # 激活环境
bash start_train.sh       # 开始训练
bash start_api.sh         # 启动API服务
```

### 2. 准备数据集
将图片按类别放入 `datasets/main/` 目录：
```
datasets/main/
├── 类别1/
├── 类别2/
└── 类别3/
```

### 3. 开始训练
```bash
# 单GPU训练
python train.py --config configs/config.yaml

# 多GPU训练（自动检测）
python train.py --config configs/config.yaml --gpus 2

# 查看训练进度
tensorboard --logdir lightning_logs
```

### 4. 模型推理
```bash
# 单张图片预测
python core/predict.py --model_path model/best_model.ckpt --image_path test.jpg

# 启动API服务
cd api && python main.py
```

## 📖 详细文档

**完整的训练指南请查看：[TRAINING_GUIDE.md](TRAINING_GUIDE.md)**

该指南包含：
- 详细的数据准备方法
- 多GPU训练配置
- 高级功能使用
- 问题排查指南
- 性能优化建议

## 📁 项目结构

```
model-train/
├── train.py              # 主训练脚本
├── requirements.txt      # 依赖包
├── configs/              # 配置文件
├── core/                 # 核心功能模块
├── datasets/             # 数据集目录
├── docs/                 # 文档
├── tools/                # 工具脚本
├── scripts/              # 启动脚本
├── api/                  # API接口
└── model/                # 模型文件
```

## 🎯 主要特性

- ✅ **极简易用**: 一行命令开始训练
- ✅ **多GPU加速**: 自动检测GPU，支持DDP分布式训练
- ✅ **智能优化**: 自动调整批次大小、学习率等参数
- ✅ **数据灵活**: 支持多种数据格式和目录结构
- ✅ **高精度**: 基于EfficientNet v2，工业级精度
- ✅ **完整功能**: 训练、微调、推理、API服务一体化

## � 常见问题

### 内存不足
```yaml
# 在 configs/config.yaml 中调整
training:
  batch_size: 16
data:
  num_workers: 2
```

### 训练速度慢
```yaml
# 启用优化选项
training:
  mixed_precision: true
  batch_size: 64
data:
  num_workers: 8
```

## � 获取帮助

- � **详细指南**: [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- 🐛 **问题排查**: 查看训练日志和配置文件
- 💡 **性能优化**: 根据硬件调整批次大小和学习率

立即开始您的商品分类项目吧！🚀
