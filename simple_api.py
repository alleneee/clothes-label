#!/usr/bin/env python3
"""
简化衣服12分类模型的FastAPI服务
"""

import os
import sys
import yaml
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import time
import uuid

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
from io import BytesIO
import base64

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.train import ProductClassifier
from core.pants_labeling_service import (
    LabelResult,
    LabelingRequest,
    LabeledImage,
    LabelingResponse,
    PantsLabelingService
)
from core.pants_workflow_service import PantsWorkflowService

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量 - 衣服模型
model = None
class_names = None
config = None
device = None
transform = None

# 全局变量 - 裤子模型
pants_labeling_service = PantsLabelingService()
pants_workflow_service = PantsWorkflowService(labeling_service=pants_labeling_service)


class PredictionRequest(BaseModel):
    """预测请求模型"""
    image_base64: str
    return_top_k: int = 3


class PredictionItem(BaseModel):
    """单个预测结果"""
    class_name: str
    confidence: float
    probability: float


class PredictionResponse(BaseModel):
    """预测响应模型 - 裤子分类专用"""
    class_name: str
    confidence: float


class ClothesClassificationResponse(BaseModel):
    """衣服分类响应模型"""
    success: bool
    message: str
    predictions: List[PredictionItem]
    processing_time: float
    model_info: Dict[str, Any]


class ModelInfo(BaseModel):
    """模型信息"""
    model_name: str
    num_classes: int
    class_names: List[str]
    device: str


class PantsWorkflowRequest(BaseModel):
    """裤子全流程请求模型"""
    brand: str
    product_code: str
    pic_list_str: Union[str, List[Dict[str, Any]]]  # 支持字符串或直接的JSON数组
    rename_in_oss: bool = True
    picture_type: str = "pants"


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    model_loaded: bool
    device: str
    uptime: float


# FastAPI应用
app = FastAPI(
    title="衣服12分类API",
    description="基于EfficientNetV2的衣服部位分类服务 - 12个分类",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 启动时间
start_time = time.time()


def load_clothes_model():
    """加载衣服分类模型"""
    global model, class_names, config, device, transform
    
    try:
        logger.info("🚀 开始加载衣服分类模型...")
        
        # 配置
        best_checkpoint = "model/best.ckpt"
        config_file = "config.yaml"
        
        # 1. 检查最佳检查点是否存在
        if not os.path.exists(best_checkpoint):
            raise FileNotFoundError(f"最佳模型不存在: {best_checkpoint}")
        
        logger.info(f"✅ 使用最佳模型: {best_checkpoint}")
        
        # 2. 加载配置
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 3. 临时禁用预训练权重下载（因为我们要加载已训练的权重）
        original_pretrained = config.get('model', {}).get('pretrained', True)
        config['model']['pretrained'] = False
        
        # 4. 加载模型
        model = ProductClassifier.load_from_checkpoint(best_checkpoint, config=config)
        
        # 5. 恢复原始配置
        config['model']['pretrained'] = original_pretrained
        model.eval()
        
        # 6. 设置设备
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        logger.info(f"✅ 模型已加载到设备: {device}")
        
        # 7. 设置类别名称 - 改为动态获取，避免与数据集不一致
        # 优先从模型本身读取类别名称（训练时已保存）
        if hasattr(model, 'class_names') and model.class_names:
            class_names = model.class_names
        # 其次尝试从配置文件中读取
        elif config.get('classes', {}).get('names'):
            class_names = config['classes']['names']
        else:
            raise ValueError("无法获取类别名称，请检查checkpoint或配置文件中的 classes.names")
        
        # 验证类别名称是否正确
        logger.info(f"✅ 类别名称验证:")
        for i, name in enumerate(class_names):
            logger.info(f"   {i}: {name}")
        
        # 特别验证关键分类
        if len(class_names) >= 8:
            logger.info(f"🎯 关键验证:")
            logger.info(f"   索引6: {class_names[6]}")
            logger.info(f"   索引7: {class_names[7]}")
            
            # 确认是否包含期望的分类
            if "正面全身模特" in class_names and "正面半身模特" in class_names:
                logger.info("✅ 类别标签验证通过：包含正面全身模特和正面半身模特")
            else:
                logger.warning("⚠️ 类别标签可能不正确，请检查数据集")
 
        # 8. 设置图片变换
        image_size = config['data']['image_size']
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        logger.info(f"✅ 衣服分类模型加载完成!")
        logger.info(f"   - 类别数量: {len(class_names)}")
        logger.info(f"   - 图片尺寸: {image_size}x{image_size}")
        logger.info(f"   - 设备: {device}")

        
        return True
        
    except Exception as e:
        logger.error(f"❌ 衣服模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_pants_model():
    """加载裤子分类模型"""
    global pants_model, pants_class_names, pants_config, device, pants_transform
    
    try:
        logger.info("🚀 开始加载裤子分类模型...")
        
        # 配置
        best_checkpoint = "model/checkpoints_pants/20251124-pants-05-0.899.ckpt"
        config_file = "config-pants.yaml"
        
        # 1. 检查检查点是否存在
        if not os.path.exists(best_checkpoint):
            logger.warning(f"裤子模型不存在: {best_checkpoint}，跳过加载")
            return False
        
        logger.info(f"✅ 使用裤子模型: {best_checkpoint}")
        
        # 2. 加载配置
        with open(config_file, 'r', encoding='utf-8') as f:
            pants_config = yaml.safe_load(f)
        
        # 3. 临时禁用预训练权重下载
        original_pretrained = pants_config.get('model', {}).get('pretrained', True)
        pants_config['model']['pretrained'] = False
        
        # 4. 加载模型
        pants_model = ProductClassifier.load_from_checkpoint(best_checkpoint, config=pants_config)
        
        # 5. 恢复原始配置
        pants_config['model']['pretrained'] = original_pretrained
        pants_model.eval()
        
        # 6. 设置设备（使用与衣服模型相同的设备）
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pants_model = pants_model.to(device)
        logger.info(f"✅ 裤子模型已加载到设备: {device}")
        
        # 7. 设置类别名称
        if hasattr(pants_model, 'class_names') and pants_model.class_names:
            pants_class_names = pants_model.class_names
        elif pants_config.get('classes', {}).get('names'):
            pants_class_names = pants_config['classes']['names']
        else:
            raise ValueError("无法获取裤子类别名称")
        
        # 验证类别名称
        logger.info(f"✅ 裤子类别名称验证:")
        for i, name in enumerate(pants_class_names):
            logger.info(f"   {i}: {name}")
        
        # 8. 设置图片变换
        image_size = pants_config['data']['image_size']
        pants_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        logger.info(f"✅ 裤子分类模型加载完成!")
        logger.info(f"   - 类别数量: {len(pants_class_names)}")
        logger.info(f"   - 图片尺寸: {image_size}x{image_size}")
        logger.info(f"   - 设备: {device}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 裤子模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def preprocess_image_from_bytes(image_bytes: bytes):
    """从字节数据预处理图片"""
    try:
        # 打开图片
        image = Image.open(io.BytesIO(image_bytes))
        
        # 处理透明度
        if image.mode == 'P' and 'transparency' in image.info:
            image = image.convert('RGBA')
        image = image.convert('RGB')
        
        # 应用变换
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # 添加批次维度
        
        return image_tensor
        
    except Exception as e:
        raise ValueError(f"图片预处理失败: {e}")


def predict_image_tensor(image_tensor: torch.Tensor, top_k: int = 3):
    """对图片张量进行预测"""
    try:
        image_tensor = image_tensor.to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # 获取top-k结果
            top_probs, top_indices = torch.topk(probabilities[0], top_k)
            
            results = []
            for i in range(top_k):
                class_idx = top_indices[i].item()
                prob = top_probs[i].item()
                class_name = class_names[class_idx]
                
                results.append(PredictionItem(
                    class_name=class_name,
                    confidence=prob,
                    probability=prob
                ))
            
            return results
            
    except Exception as e:
        raise RuntimeError(f"模型预测失败: {e}")


def predict_pants_from_bytes(image_bytes: bytes) -> tuple:
    """
    从字节数据预测裤子类型（通用方法）
    
    Args:
        image_bytes: 图片字节数据
    
    Returns:
        tuple: (class_name, confidence)
    """
    if pants_model is None:
        raise RuntimeError("裤子模型未加载")
    
    try:
        # 预处理图片
        image = Image.open(BytesIO(image_bytes))
        if image.mode == 'P' and 'transparency' in image.info:
            image = image.convert('RGBA')
        image = image.convert('RGB')
        image_tensor = pants_transform(image).unsqueeze(0).to(device)
        
        # 进行预测
        with torch.no_grad():
            outputs = pants_model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            top_prob, top_idx = torch.max(probabilities[0], dim=0)
            
            class_name = pants_class_names[top_idx.item()]
            confidence = top_prob.item()
        
        return class_name, confidence
        
    except Exception as e:
        raise RuntimeError(f"裤子预测失败: {e}")


@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    # 加载衣服模型
    clothes_success = load_clothes_model()
    if not clothes_success:
        logger.error("衣服模型加载失败，相关服务可能无法正常工作")
    
    # 加载裤子模型
    pants_success = load_pants_model()
    if not pants_success:
        logger.warning("裤子模型加载失败，裤子分类服务不可用")


@app.get("/", response_model=Dict[str, str])
async def root():
    """根路径"""
    return {
        "message": "衣服12分类API服务",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=device if device else "unknown",
        uptime=time.time() - start_time
    )


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """获取模型信息"""
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    return ModelInfo(
        model_name="enhanced-clothes-best",
        num_classes=len(class_names),
        class_names=class_names,
        device=device
    )


@app.post("/predict/upload", response_model=ClothesClassificationResponse)
async def predict_upload(
    file: UploadFile = File(...),
    top_k: int = 3
):
    """通过文件上传进行预测"""
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    # 检查文件类型
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="文件必须是图片格式")
    
    start_time_pred = time.time()
    
    try:
        # 读取文件内容
        contents = await file.read()
        
        # 预处理图片
        image_tensor = preprocess_image_from_bytes(contents)
        
        # 进行预测
        predictions = predict_image_tensor(image_tensor, top_k)
        
        processing_time = time.time() - start_time_pred
        
        return ClothesClassificationResponse(
            success=True,
            message="预测成功",
            predictions=predictions,
            processing_time=processing_time,
            model_info={
                "model_name": "enhanced-clothes-best",
                "device": device
            }
        )
        
    except Exception as e:
        logger.error(f"预测失败: {e}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@app.post("/predict/base64", response_model=ClothesClassificationResponse)
async def predict_base64(request: PredictionRequest):
    """通过base64编码的图片进行预测"""
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    start_time_pred = time.time()
    
    try:
        # 解码base64图片
        try:
            # 移除可能的数据URL前缀
            if ',' in request.image_base64:
                request.image_base64 = request.image_base64.split(',')[1]
            
            image_bytes = base64.b64decode(request.image_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"base64解码失败: {str(e)}")
        
        # 预处理图片
        image_tensor = preprocess_image_from_bytes(image_bytes)
        
        # 进行预测
        predictions = predict_image_tensor(image_tensor, request.return_top_k)
        
        processing_time = time.time() - start_time_pred
        
        return ClothesClassificationResponse(
            success=True,
            message="预测成功",
            predictions=predictions,
            processing_time=processing_time,
            model_info={
                "model_name": "enhanced-clothes-best",
                "device": device
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"预测失败: {e}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@app.get("/classes", response_model=List[str])
async def get_classes():
    """获取衣服分类类别名称"""
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    return class_names


@app.get("/classes/pants", response_model=List[str])
async def get_pants_classes():
    """获取裤子分类类别名称"""
    if pants_model is None:
        raise HTTPException(status_code=503, detail="裤子模型未加载")
    
    return pants_class_names


@app.get("/model/pants/info", response_model=ModelInfo)
async def get_pants_model_info():
    """获取裤子模型信息"""
    if pants_model is None:
        raise HTTPException(status_code=503, detail="裤子模型未加载")
    
    return ModelInfo(
        model_name="pants-classification-best",
        num_classes=len(pants_class_names),
        class_names=pants_class_names,
        device=device
    )


@app.post("/predict/pants/upload", response_model=PredictionResponse)
async def predict_pants_upload(file: UploadFile = File(...)):
    """裤子分类 - 通过文件上传进行预测，返回置信度最高的类别"""
    if pants_model is None:
        raise HTTPException(status_code=503, detail="裤子模型未加载")
    
    # 检查文件类型
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="文件必须是图片格式")
    
    try:
        # 读取文件内容
        contents = await file.read()
        
        # 调用通用预测函数
        class_name, confidence = predict_pants_from_bytes(contents)
        
        return PredictionResponse(
            class_name=class_name,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"裤子分类预测失败: {e}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@app.post("/label/pants", response_model=LabelingResponse)
async def label_pants_images(payload: Union[List[LabelResult], LabelingRequest]):
    """裤子打标接口 - 根据品牌规则挑选6张图片"""
    if isinstance(payload, list):
        results = payload
    else:
        results = payload.results

    if not results:
        raise HTTPException(status_code=400, detail="results 不能为空")

    try:
        ordered = pants_labeling_service.select_images(results)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 转换为LabeledImage列表
    selected_images = [
        LabeledImage(
            product_code=record['item'].product_code,
            **{k: getattr(record['item'], k) for k in ['pic_id', 'type', 'type_code', 'size', 'confidence', 'pic_name']},
            new_file_name=record['new_file_name']
        )
        for record in ordered[:6]
    ]

    return LabelingResponse(selected=selected_images)


@app.post("/workflow/pants", response_model=LabelingResponse)
async def pants_workflow(request: PantsWorkflowRequest):
    """裤子全流程接口：获取图片 -> 分类 -> 打标"""
    if pants_model is None:
        raise HTTPException(status_code=503, detail="裤子模型未加载")
    
    try:
        # 处理pic_list_str：支持直接传入List或字符串
        import json
        if isinstance(request.pic_list_str, list):
            # 如果是List，转换为JSON字符串
            pic_list_str = json.dumps(request.pic_list_str, ensure_ascii=False)
            logger.info(f"品牌{request.brand}, 货号{request.product_code}, 接收到List格式的pic_list，已转换为JSON字符串")
        else:
            pic_list_str = request.pic_list_str
            logger.info(f"品牌{request.brand}, 货号{request.product_code}, 接收到字符串格式的pic_list")
        
        # 调用完整流程，使用通用预测函数
        result_list = pants_workflow_service.process_complete_workflow(
            brand=request.brand,
            product_code=request.product_code,
            pic_list_str=pic_list_str,
            predict_func=predict_pants_from_bytes,
            rename_in_oss=request.rename_in_oss,
            picture_type=request.picture_type
        )
        
        # 转换为LabeledImage列表
        selected_images = [
            LabeledImage(**item)
            for item in result_list
        ]
        
        return LabelingResponse(selected=selected_images)
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"裤子全流程处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@app.post("/predict/pants/base64", response_model=PredictionResponse)
async def predict_pants_base64(request: PredictionRequest):
    """裤子分类 - 通过base64编码的图片进行预测，返回置信度最高的类别"""
    if pants_model is None:
        raise HTTPException(status_code=503, detail="裤子模型未加载")
    
    try:
        # 解码base64图片
        try:
            if ',' in request.image_base64:
                request.image_base64 = request.image_base64.split(',')[1]
            image_bytes = base64.b64decode(request.image_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"base64解码失败: {str(e)}")
        
        # 预处理图片
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode == 'P' and 'transparency' in image.info:
            image = image.convert('RGBA')
        image = image.convert('RGB')
        image_tensor = pants_transform(image).unsqueeze(0).to(device)
        
        # 进行预测
        with torch.no_grad():
            outputs = pants_model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            top_prob, top_idx = torch.max(probabilities[0], dim=0)
            
            class_name = pants_class_names[top_idx.item()]
            confidence = top_prob.item()
        
        return PredictionResponse(
            class_name=class_name,
            confidence=confidence
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"裤子分类预测失败: {e}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


if __name__ == "__main__":
    # 启动配置
    host = "0.0.0.0"
    port = 8000
    
    print("=" * 60)
    print("🚀 启动服饰分类API服务（衣服+裤子）")
    print("=" * 60)
    print(f"📡 服务地址: http://{host}:{port}")
    print(f"📚 API文档: http://{host}:{port}/docs")
    print(f"🔧 健康检查: http://{host}:{port}/health")
    print(f"👔 衣服分类: http://{host}:{port}/predict/upload")
    print(f"👖 裤子分类: http://{host}:{port}/predict/pants/upload")
    print("=" * 60)
    
    # 启动服务
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    ) 