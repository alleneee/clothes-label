# 模型训练完整指南

本指南涵盖了从数据准备到模型训练的完整流程，包括单GPU、多GPU训练和各种高级功能。

## 🚀 快速开始

### 1. 环境准备

#### 一键安装（推荐）
```bash
# 一键创建conda环境并安装所有依赖
bash install.sh
```

安装完成后会自动创建以下启动脚本：
- `activate.sh` - 激活环境
- `start_train.sh` - 开始训练
- `start_api.sh` - 启动API服务
- `start_jupyter.sh` - 启动Jupyter

#### 手动安装（可选）
```bash
# 创建conda环境
conda create -n model-train python=3.11 -y

# 激活环境
conda activate model-train

# 安装PyTorch (自动检测GPU/CPU)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 安装项目依赖
pip install -r requirements.txt

# 安装开发工具
pip install jupyter ipykernel tensorboard
```

### 2. 数据准备

#### 方法A: 标准目录结构（推荐）
```
datasets/
├── main/
│   ├── train/
│   │   ├── 类别1/
│   │   └── 类别2/
│   ├── val/
│   │   ├── 类别1/
│   │   └── 类别2/
│   └── test/
│       ├── 类别1/
│       └── 类别2/
```

#### 方法B: 自动拆分
```
datasets/
├── main/
│   ├── 类别1/    # 所有类别1的图片
│   └── 类别2/    # 所有类别2的图片
```

#### 方法C: 嵌套结构（支持细粒度分类）
```
datasets/
├── main/
│   ├── 衣服/
│   │   ├── 正面图/
│   │   ├── 背面图/
│   │   └── 侧面图/
│   └── 裤子/
│       ├── 正面图/
│       └── 背面图/
```

#### 数据准备工具
```bash
# 自动解析和拆分数据集
python tools/auto_parse_dataset.py your_dataset

# 手动设置数据集
python tools/setup_dataset.py --mode create_structure
```

### 3. 配置文件

编辑 `configs/config.yaml`：

```yaml
# 数据配置
data:
  data_dir: "datasets/main"
  batch_size: 32
  image_size: 224
  num_workers: 4
  auto_split: true
  train_split: 0.7
  val_split: 0.2
  test_split: 0.1
  nested_structure: false        # 是否使用嵌套结构
  classification_mode: "main_category"  # 分类模式

# 模型配置
model:
  name: "efficientnetv2_s"       # 模型类型
  learning_rate: 1e-4
  weight_decay: 1e-4

# 训练配置
training:
  max_epochs: 100
  patience: 10
  mixed_precision: true          # 混合精度训练
  gradient_clip_val: 1.0

# 日志配置
logging:
  log_dir: "lightning_logs"
  experiment_name: "product_classification"
```

## 🎯 开始训练

### 单GPU/CPU训练
```bash
# 基础训练
python train.py --config configs/config.yaml

# 启用硬件自动优化
python train.py --config configs/config.yaml --auto-optimize

# 禁用多GPU（强制单GPU）
python train.py --config configs/config.yaml --no-multi-gpu
```

### 多GPU训练（推荐）
```bash
# 自动多GPU训练（推荐）
python train.py --config configs/config.yaml

# 指定GPU数量
python train.py --config configs/config.yaml --gpus 2

# 强制使用DDP策略
python train.py --config configs/config.yaml --strategy ddp

# 使用专门的多GPU脚本
python scripts/train_multi_gpu.py --config configs/config.yaml
```

### 微调训练
```bash
# 对错误样本进行微调
python finetune/corrected_fine_tune.py --config configs/config.yaml

# 快速微调
python finetune/quick_corrected_fine_tune.py --config configs/config.yaml
```

## ⚙️ 高级功能

### 1. 硬件自动优化
系统会自动检测硬件配置并优化训练参数：
- 自动调整批次大小
- 优化数据加载进程数
- 启用混合精度训练
- 选择最优的训练策略

### 2. 多GPU训练策略

| 策略 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| DDP | 2+GPU，推荐 | 最高效，支持多机 | 内存占用稍高 |
| DataParallel | 单机多卡 | 实现简单 | 存在GPU0瓶颈 |
| DeepSpeed | 大模型 | 内存优化 | 需要额外安装 |

### 3. 数据不均衡处理
```yaml
# 在配置文件中启用
imbalance:
  enabled: true
  strategy: "weighted_loss"      # 或 "oversample", "undersample"
  auto_detect: true
```

### 4. 分类模式选择

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `main_category` | 按主类别分类 | 商品大类分类 |
| `sub_category` | 按子类别分类 | 细粒度分类 |
| `image_type` | 按图像类型分类 | 角度/视图分类 |

## 📊 监控和调试

### TensorBoard监控
```bash
# 启动TensorBoard
tensorboard --logdir lightning_logs

# 浏览器访问
http://localhost:6006
```

### GPU监控
```bash
# 实时监控GPU使用
watch -n 1 nvidia-smi

# 测试多GPU功能
python tests/test_multi_gpu.py
```

## 🔧 常见问题解决

### 内存不足
```yaml
# 减小批次大小
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

data:
  pin_memory: true
  num_workers: 8
```

### 精度不够高
```yaml
# 使用更大的模型
model:
  name: "efficientnetv2_l"
  learning_rate: 5e-5

training:
  max_epochs: 200
```

### AVIF格式支持
```bash
# 安装AVIF支持
python tools/install_avif_support.py

# 转换AVIF到JPEG
python tools/avif_converter.py --input_dir datasets/main --output_dir datasets/converted
```

## 🎯 推理预测

### 命令行预测
```bash
# 单张图片预测
python core/predict.py --model_path model/best_model.ckpt --mode single --image_path test.jpg

# 批量预测
python core/predict.py --model_path model/best_model.ckpt --mode batch --image_folder test_images/

# Web界面
python core/predict.py --model_path model/best_model.ckpt --mode web
```

### FastAPI服务（推荐）
```bash
# 启动API服务
python scripts/start_api.py

# 开发模式（自动重载）
python scripts/start_api.py --dev

# 生产模式（多进程）
python scripts/start_api.py --prod

# 自定义配置
python scripts/start_api.py --host 0.0.0.0 --port 8000 --model-path model/best_model.ckpt
```

### API使用示例
```python
import requests
import base64

# 读取图像文件
with open('test.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

# 发送预测请求
response = requests.post('http://localhost:8000/predict/single',
    json={
        'image_data': image_data,
        'return_probabilities': True
    }
)

result = response.json()
print(f"预测类别: {result['data']['predicted_class']}")
print(f"置信度: {result['data']['confidence']}")
```

### API功能
- **单张预测**: `POST /predict/single` - Base64图像数据
- **批量预测**: `POST /predict/batch` - 最多50张图像
- **文件上传**: `POST /predict/upload` - 直接上传图像文件
- **健康检查**: `GET /system/health` - 服务状态检查
- **模型信息**: `GET /system/model-info` - 获取模型详情
- **API文档**: `http://localhost:8000/docs` - Swagger UI

## 📈 性能优化建议

### 1. 数据加载优化
- 使用SSD存储数据集
- 适当增加 `num_workers`
- 启用 `pin_memory`

### 2. 训练优化
- 启用混合精度训练
- 使用合适的批次大小
- 多GPU训练使用DDP策略

### 3. 模型选择
- 快速原型：`efficientnetv2_s`
- 平衡性能：`efficientnetv2_m`
- 追求精度：`efficientnetv2_l`

## 🛠️ 工具脚本

```bash
# 数据集分析
python tools/quick_dataset_analysis.py

# 清理损坏图片
python tools/clean_corrupted_images.py

# 修复数据集格式
python tools/fix_dataset_format.py

# 快速设置
python scripts/quick_setup.py
```

## 💡 最佳实践

1. **数据准备**：确保数据集结构正确，图片质量良好
2. **配置调优**：根据硬件配置调整批次大小和学习率
3. **监控训练**：使用TensorBoard监控训练过程
4. **多GPU训练**：优先使用DDP策略
5. **模型保存**：定期保存检查点，防止训练中断
6. **测试验证**：训练完成后在测试集上验证性能

---

这个指南涵盖了模型训练的所有重要方面。如果遇到问题，请检查配置文件设置或查看训练日志获取更多信息。
