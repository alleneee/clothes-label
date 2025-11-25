# OSS图片预测API文档

## 概述

简化的图片预测接口，专为OSS图片预测场景设计。使用两步流程：
1. 先调用OSS接口获取图片数据
2. 再调用简化预测接口，只返回置信度最高的结果

## 接口详情

### 端点
```
POST /predict/simple
```

### 请求参数
| 参数名 | 类型 | 必填 | 描述 |
|--------|------|------|------|
| image_data | string | 是 | Base64编码的图像数据 |
| pic_name | string | 否 | 图片名称（用于标识） |

### 响应格式

#### 成功响应
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

#### 错误响应
```json
{
    "detail": {
        "error": "无效的Base64图像数据",
        "error_code": "INVALID_IMAGE"
    }
}
```

## 使用示例

### 完整流程示例（Python）
```python
import requests
import base64

# 步骤1: 从OSS获取图片
def get_oss_image_base64(pic_name):
    response = requests.get(
        'http://localhost:8124/oss_pic',
        params={'pic_name': pic_name}
    )
    if response.status_code == 200:
        return base64.b64encode(response.content).decode('utf-8')
    else:
        raise Exception(f"OSS图片获取失败: {response.status_code}")

# 步骤2: 调用简化预测接口
def predict_image(image_data, pic_name):
    response = requests.post(
        'http://localhost:8000/predict/simple',
        json={
            'image_data': image_data,
            'pic_name': pic_name
        }
    )
    return response.json()

# 完整使用流程
pic_name = "products/shirt_001.jpg"
try:
    # 获取图片数据
    image_data = get_oss_image_base64(pic_name)

    # 进行预测
    result = predict_image(image_data, pic_name)

    if result['success']:
        data = result['data']
        print(f"预测类别: {data['predicted_class']}")
        print(f"置信度: {data['confidence']:.4f}")
        print(f"处理时间: {data['processing_time']:.3f}秒")
except Exception as e:
    print(f"预测失败: {e}")
```

### curl命令示例
```bash
# 步骤1: 获取OSS图片并转换为base64
IMAGE_DATA=$(curl -s "http://localhost:8124/oss_pic?pic_name=products/shirt_001.jpg" | base64 -w 0)

# 步骤2: 调用预测接口
curl -X POST "http://localhost:8000/predict/simple" \
  -H "Content-Type: application/json" \
  -d "{
    \"image_data\": \"$IMAGE_DATA\",
    \"pic_name\": \"products/shirt_001.jpg\"
  }"
```

### JavaScript fetch
```javascript
const response = await fetch(
    'http://localhost:8000/predict/oss?pic_name=products/shirt_001.jpg'
);
const result = await response.json();

if (result.success) {
    console.log('预测类别:', result.data.predicted_class);
    console.log('置信度:', result.data.confidence);
}
```

## 特性

### 1. 自动格式处理
- 支持JPEG、PNG、AVIF等多种图片格式
- 自动处理透明通道，转换为RGB格式
- AVIF格式会自动转换为JPEG进行处理

### 2. 简化输出
- 只返回置信度最高的预测结果
- 不包含所有类别的概率分布
- 减少响应数据量，提高传输效率

### 3. 错误处理
- 图片不存在时返回404错误
- 图片格式无效时返回400错误
- OSS服务不可用时返回相应错误信息

## 与现有接口的对比

| 特性 | 原有接口 (/predict/single) | 新OSS接口 (/predict/oss) |
|------|---------------------------|--------------------------|
| 输入方式 | Base64编码的图片数据 | OSS图片名称 |
| 输出内容 | 完整预测结果 + 概率分布 | 仅最高置信度结果 |
| 图片获取 | 需要客户端处理 | 服务端自动获取 |
| AVIF支持 | 需要客户端转换 | 服务端自动转换 |
| 响应大小 | 较大（包含所有概率） | 较小（仅核心信息） |

## 依赖服务

### OSS图片服务
- 端口: 8124
- 接口: `/oss_pic?pic_name={图片名称}`
- 功能: 从阿里云OSS获取图片并返回JPEG格式

### 模型推理服务
- 端口: 8000
- 功能: 图像分类预测

## 错误代码

| 错误代码 | 描述 | 解决方案 |
|----------|------|----------|
| INVALID_IMAGE | 图片不存在或无法获取 | 检查图片路径是否正确 |
| PROCESSING_ERROR | 预测处理失败 | 检查模型服务状态 |
| VALIDATION_ERROR | 参数验证失败 | 检查请求参数格式 |

## 性能考虑

1. **网络延迟**: 需要从OSS获取图片，可能增加响应时间
2. **并发限制**: OSS服务的并发处理能力
3. **图片大小**: 大图片会增加传输和处理时间
4. **缓存策略**: 考虑对常用图片进行缓存

## 最佳实践

1. **图片命名**: 使用有意义的图片路径和名称
2. **错误处理**: 始终检查响应的success字段
3. **超时设置**: 设置合适的请求超时时间（建议30秒）
4. **重试机制**: 对网络错误实施重试策略
5. **日志记录**: 记录预测结果用于后续分析

## 测试工具

使用提供的测试脚本验证接口功能：
```bash
python examples/test_oss_prediction.py
```

## 注意事项

1. 确保OSS服务 (端口8124) 正常运行
2. 确保模型推理服务 (端口8000) 正常运行
3. 图片路径区分大小写
4. 支持的图片格式: JPEG, PNG, AVIF, BMP等
5. 建议图片大小不超过10MB
