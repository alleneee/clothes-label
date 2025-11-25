"""
简单的推理脚本
用于单张图片或批量图片预测
"""

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import argparse
import json
from pathlib import Path
import yaml
from tqdm import tqdm

from .train import ProductClassifier


def load_config(config_path: str = 'config.yaml'):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def predict_single_image(model_path: str, image_path: str, config_path: str = 'config.yaml'):
    """预测单张图片"""
    
    # 加载配置和模型
    config = load_config(config_path)
    model = ProductClassifier.load_from_checkpoint(model_path, config=config)
    
    # 预测
    result = model.predict_single(image_path)
    
    # 输出结果
    print(f"\n图片: {image_path}")
    print(f"预测类别: {result['predicted_class']}")
    print(f"置信度: {result['confidence']:.4f}")
    print("\n各类别概率:")
    for class_name, prob in result['probabilities'].items():
        print(f"  {class_name}: {prob:.4f}")
    
    return result


def predict_batch_images(model_path: str, image_folder: str, config_path: str = 'config.yaml', 
                        output_file: str = 'predictions.json'):
    """批量预测图片"""
    
    # 加载配置和模型
    config = load_config(config_path)
    model = ProductClassifier.load_from_checkpoint(model_path, config=config)
    
    # 获取所有图片文件
    image_folder = Path(image_folder)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(image_folder.glob(f'*{ext}'))
        image_files.extend(image_folder.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"在 {image_folder} 中没有找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 张图片，开始批量预测...")
    
    # 批量预测
    results = []
    for img_path in tqdm(image_files, desc="预测进度"):
        try:
            result = model.predict_single(str(img_path))
            result['filename'] = img_path.name
            result['filepath'] = str(img_path)
            results.append(result)
        except Exception as e:
            print(f"预测失败 {img_path}: {e}")
            results.append({
                'filename': img_path.name,
                'filepath': str(img_path),
                'error': str(e),
                'predicted_class': None,
                'confidence': 0.0
            })
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 统计结果
    successful = len([r for r in results if 'error' not in r])
    class_counts = {}
    total_confidence = 0
    
    for result in results:
        if 'error' not in result:
            class_name = result['predicted_class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            total_confidence += result['confidence']
    
    # 输出统计
    print(f"\n批量预测完成!")
    print(f"总图片数: {len(image_files)}")
    print(f"成功预测: {successful}")
    print(f"平均置信度: {total_confidence/successful:.4f}")
    print(f"结果保存到: {output_file}")
    
    print("\n类别分布:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")
    
    return results


def create_web_interface(model_path: str, config_path: str = 'config.yaml'):
    """创建简单的Web界面（需要安装gradio）"""
    
    try:
        import gradio as gr
    except ImportError:
        print("请安装gradio: pip install gradio")
        return
    
    # 加载配置和模型
    config = load_config(config_path)
    model = ProductClassifier.load_from_checkpoint(model_path, config=config)
    
    def predict_image(image):
        """预测函数"""
        if image is None:
            return "请上传图片"
        
        # 保存临时图片
        temp_path = "temp_image.jpg"
        image.save(temp_path)
        
        # 预测
        result = model.predict_single(temp_path)
        
        # 格式化输出
        output = f"预测类别: {result['predicted_class']}\n"
        output += f"置信度: {result['confidence']:.4f}\n\n"
        output += "各类别概率:\n"
        for class_name, prob in result['probabilities'].items():
            output += f"  {class_name}: {prob:.4f}\n"
        
        return output
    
    # 创建界面
    interface = gr.Interface(
        fn=predict_image,
        inputs=gr.Image(type="pil"),
        outputs=gr.Textbox(lines=10),
        title="商品图片分类器",
        description="上传商品图片，自动识别是衣服、裤子还是配饰"
    )
    
    # 启动界面
    interface.launch(share=True)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='商品分类预测工具')
    
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--mode', type=str, default='single', 
                       choices=['single', 'batch', 'web'], help='预测模式')
    
    # 单张图片预测
    parser.add_argument('--image_path', type=str, help='图片路径（单张预测）')
    
    # 批量预测
    parser.add_argument('--image_folder', type=str, help='图片文件夹（批量预测）')
    parser.add_argument('--output_file', type=str, default='predictions.json', 
                       help='结果保存文件（批量预测）')
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not Path(args.model_path).exists():
        print(f"错误: 模型文件不存在: {args.model_path}")
        return
    
    # 检查配置文件
    if not Path(args.config).exists():
        print(f"错误: 配置文件不存在: {args.config}")
        return
    
    if args.mode == 'single':
        # 单张图片预测
        if not args.image_path:
            print("单张预测模式需要指定 --image_path")
            return
        
        if not Path(args.image_path).exists():
            print(f"错误: 图片文件不存在: {args.image_path}")
            return
        
        predict_single_image(args.model_path, args.image_path, args.config)
        
    elif args.mode == 'batch':
        # 批量预测
        if not args.image_folder:
            print("批量预测模式需要指定 --image_folder")
            return
        
        if not Path(args.image_folder).exists():
            print(f"错误: 图片文件夹不存在: {args.image_folder}")
            return
        
        predict_batch_images(args.model_path, args.image_folder, args.config, args.output_file)
        
    elif args.mode == 'web':
        # Web界面
        create_web_interface(args.model_path, args.config)


if __name__ == '__main__':
    main()


# 使用示例
"""
# 单张图片预测
python predict.py --model_path checkpoints/best.ckpt --mode single --image_path test.jpg

# 批量预测
python predict.py --model_path checkpoints/best.ckpt --mode batch --image_folder test_images/

# Web界面
python predict.py --model_path checkpoints/best.ckpt --mode web
"""
