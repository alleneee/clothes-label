# 衣服分类模型项目

基于EfficientNetV2的衣服11分类模型，支持训练、预测和API服务。

## 项目结构

```
├── simple_api.py          # 🚀 主要API服务入口
├── train_launcher.py     # 📚 训练启动脚本
├── test_model.py         # 🧪 模型测试脚本
├── analyze_dataset.py    # 📊 数据集分析工具
├── config.yaml           # ⚙️ 主配置文件
├── requirements.txt      # 📦 依赖包
├── core/                 # 核心模块
│   ├── train.py         # 训练核心代码
│   ├── data_module.py   # 数据处理模块
│   ├── predict.py       # 预测功能
│   └── hardware_optimizer.py  # 硬件优化
├── model/               # 模型文件
│   └── checkpoints_enhanced/  # 训练好的模型
├── configs/             # 配置文件 (简化版)
├── datasets/            # 数据集
└── docs/               # 文档
```

## 核心功能

### 1. 🚀 API服务（主要入口）
```bash
# 启动API服务
python simple_api.py

# 服务地址: http://localhost:8000
# API文档: http://localhost:8000/docs
```

### 2. 📚 模型训练
```bash
# 模型训练
python train_launcher.py --config config.yaml

# 恢复训练
python train_launcher.py --config config.yaml --resume model/checkpoints_enhanced/latest.ckpt
```

#### 🚀 性能优化
本项目包含多项性能优化功能：
- **torch.compile()**: PyTorch 2.0编译优化，显著提升训练速度
- **混合精度训练**: 16位混合精度，减少显存占用
- **优化的数据加载**: 多进程数据加载，提升IO效率
- **梯度累积**: 支持大批次等效训练

### 3. 🧪 模型测试
```bash
# 测试单张图片
python test_model.py --image path/to/image.jpg

# 指定模型和配置
python test_model.py --image path/to/image.jpg --checkpoint model/checkpoints_enhanced/best.ckpt
```

### 4. 📊 数据集分析
```bash
# 分析数据集
python analyze_dataset.py --dataset datasets/main

# 生成分布图
python analyze_dataset.py --dataset datasets/main --plot

# 检查数据完整性
python analyze_dataset.py --dataset datasets/main --check-integrity
```

## 分类类别

模型支持11个衣服部位分类：
- logo - 标志/商标
- 下摆 - 下摆部位
- 侧面 - 侧面视角
- 其他 - 其他部位
- 口袋 - 口袋部位
- 正面 - 正面视角
- 正面模特 - 正面模特图
- 背面 - 背面视角
- 背面模特 - 背面模特图
- 袖口 - 袖口部位
- 领口 - 领口部位

## 快速开始

### 1. 环境准备
```bash
# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备
```bash
# 分析数据集
python analyze_dataset.py --dataset datasets/main --check-integrity
```

### 3. 启动服务
```bash
# 启动API服务（推荐）
python simple_api.py
```

### 4. 测试预测
```bash
# 测试图片预测
python test_model.py --image path/to/test/image.jpg
```

## API使用

### 上传图片预测
```bash
curl -X POST "http://localhost:8000/predict/upload" \
  -F "file=@your_image.jpg" \
  -F "top_k=3"
```

### Base64图片预测
```bash
curl -X POST "http://localhost:8000/predict/base64" \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "data:image/jpeg;base64,/9j/4AAQ...", "return_top_k": 3}'
```

### 健康检查
```bash
curl http://localhost:8000/health
```

## 配置说明

主要配置文件：`config.yaml`
- `model`: 模型配置（网络架构、参数等）
- `data`: 数据配置（路径、预处理等）
- `training`: 训练配置（学习率、批次大小等）
- `checkpointing`: 检查点配置

## 硬件要求

- **推荐**: NVIDIA GPU（如A10、V100等）
- **最低**: CPU（速度较慢）
- **内存**: 至少8GB RAM
- **存储**: 至少5GB可用空间

## 性能指标

- **准确率**: 85.1%（基于enhanced-clothes-08-0.851.ckpt）
- **推理速度**: GPU下约50ms/张，CPU下约200ms/张
- **模型大小**: 约100MB

## 维护说明

### 磁盘空间管理
```bash
# 清理日志
rm -rf logs/*.log

# 清理缓存
find . -name "__pycache__" -type d -exec rm -rf {} +

# 清理旧检查点（保留最好的几个）
```

### 模型更新
1. 训练新模型后，检查点会自动保存到 `model/checkpoints_enhanced/`
2. API服务会自动加载最高准确率的模型
3. 可以通过修改 `simple_api.py` 中的路径来指定特定模型

## 故障排除

### 常见问题
1. **CUDA内存不足**: 减少batch_size或image_size
2. **模型加载失败**: 检查检查点路径和配置文件
3. **数据集路径错误**: 确认datasets目录结构正确
4. **端口占用**: 修改simple_api.py中的端口号

### 日志查看
```bash
# 查看训练日志
tail -f logs/training.log

# 查看API日志
tail -f logs/api.log
```

## 联系信息

如有问题，请检查：
1. 配置文件是否正确
2. 依赖是否完整安装
3. 数据集路径是否正确
4. 模型文件是否存在 # clothes-label
