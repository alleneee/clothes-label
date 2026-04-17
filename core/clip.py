"""
商品图标签检测、裁剪与拼接工具

使用 Qwen-VL 模型识别商品图中的吊牌、水洗标、合格证，
裁剪后将同类型标签左右并排拼接。
"""

from __future__ import annotations

import base64
import logging
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from openai import OpenAI
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# 数据模型
# ============================================================


class LabelType(Enum):
    HANG_TAG = "吊牌"
    WASH_LABEL = "水洗标"
    CERTIFICATE = "合格证"

    @classmethod
    def from_str(cls, s: str) -> LabelType | None:
        mapping = {
            "吊牌": cls.HANG_TAG,
            "hang_tag": cls.HANG_TAG,
            "水洗标": cls.WASH_LABEL,
            "wash_label": cls.WASH_LABEL,
            "合格证": cls.CERTIFICATE,
            "certificate": cls.CERTIFICATE,
        }
        return mapping.get(s.strip())


@dataclass(frozen=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class DetectedLabel:
    label_type: LabelType
    bbox: BoundingBox
    source_image: Path
    confidence: float = 0.0


# ============================================================
# 检测器：抽象基类 + Qwen-VL 实现
# ============================================================


class LabelDetector(ABC):
    """标签检测器抽象基类（依赖倒置 + 开闭原则）"""

    @abstractmethod
    def detect(self, image_path: Path) -> list[DetectedLabel]: ...


class QwenVLDetector(LabelDetector):
    """基于 Qwen-VL 的标签检测器，使用 OpenAI 兼容格式 + 原生 grounding"""

    PROMPT = (
        "请检测并框出图中所有的标签（吊牌、水洗标、合格证），"
        "对每个标签用<ref>标签类型</ref><box>坐标</box>格式输出。"
    )

    # 匹配 <ref>类型</ref><box>x1,y1,x2,y2</box>
    _GROUNDING_PATTERN = re.compile(
        r"<ref>(.*?)</ref>\s*<box>(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)</box>"
    )

    def __init__(self, base_url: str, api_key: str, model: str = "qwen-vl-max"):
        self._client = OpenAI(api_key="sk-ad248075acc34cdb9c896724d301be2b", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        self._model = model

    def detect(self, image_path: Path) -> list[DetectedLabel]:
        b64 = self._encode_image(image_path)
        img_w, img_h = Image.open(image_path).size

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                }
            ],
            temperature=0.1,
        )

        content = response.choices[0].message.content or ""
        logger.debug("模型原始返回: %s", content)
        labels = self._parse_grounding(content, image_path, img_w, img_h)
        logger.info("图片 %s 检测到 %d 个标签", image_path.name, len(labels))
        return labels

    def _encode_image(self, path: Path) -> str:
        return base64.b64encode(path.read_bytes()).decode("utf-8")

    def _parse_grounding(
        self, content: str, source: Path, img_w: int, img_h: int
    ) -> list[DetectedLabel]:
        results: list[DetectedLabel] = []
        for match in self._GROUNDING_PATTERN.finditer(content):
            type_str = match.group(1).strip()
            label_type = LabelType.from_str(type_str)
            if label_type is None:
                logger.warning("未知标签类型: %s", type_str)
                continue

            # Qwen-VL 原生 grounding 坐标范围 0-999
            raw = [int(match.group(i)) for i in range(2, 6)]
            bbox = self._to_pixel_bbox(raw, img_w, img_h)
            if bbox is None:
                continue

            results.append(
                DetectedLabel(
                    label_type=label_type,
                    bbox=bbox,
                    source_image=source,
                )
            )
        return results

    @staticmethod
    def _to_pixel_bbox(
        raw: list[int], img_w: int, img_h: int
    ) -> BoundingBox | None:
        x1 = int(max(0, min(raw[0], 999)) / 999 * img_w)
        y1 = int(max(0, min(raw[1], 999)) / 999 * img_h)
        x2 = int(max(0, min(raw[2], 999)) / 999 * img_w)
        y2 = int(max(0, min(raw[3], 999)) / 999 * img_h)

        if x2 <= x1 or y2 <= y1:
            return None
        return BoundingBox(x1, y1, x2, y2)


# ============================================================
# 裁剪器（单一职责）
# ============================================================


class LabelCropper:
    """从原图中裁剪标签区域"""

    def __init__(self, margin: int = 40):
        self._margin = margin

    def crop(self, label: DetectedLabel) -> Image.Image:
        with Image.open(label.source_image) as img:
            img_w, img_h = img.size
            m = self._margin
            x1 = max(0, label.bbox.x1 - m)
            y1 = max(0, label.bbox.y1 - m)
            x2 = min(img_w, label.bbox.x2 + m)
            y2 = min(img_h, label.bbox.y2 + m)
            return img.crop((x1, y1, x2, y2)).copy()


# ============================================================
# 分组器（单一职责）
# ============================================================


class LabelGrouper:
    """按标签类型对检测结果进行分组汇总"""

    def group(
        self, labels: list[DetectedLabel]
    ) -> dict[LabelType, list[DetectedLabel]]:
        groups: dict[LabelType, list[DetectedLabel]] = defaultdict(list)
        for label in labels:
            groups[label.label_type].append(label)
        return dict(groups)


# ============================================================
# 拼接器（单一职责）
# ============================================================


class LabelStitcher:
    """将两张标签图左右并排拼接，标签占满画面，浅色背景"""

    def __init__(
        self,
        bg_color: tuple[int, int, int] = (245, 245, 245),
        padding: int = 20,
    ):
        self._bg_color = bg_color
        self._padding = padding

    def stitch(self, img1: Image.Image, img2: Image.Image) -> Image.Image:
        target_h = max(img1.height, img2.height)
        r1 = self._resize_to_height(img1, target_h)
        r2 = self._resize_to_height(img2, target_h)

        p = self._padding
        canvas_w = p + r1.width + p + r2.width + p
        canvas_h = p + target_h + p

        canvas = Image.new("RGB", (canvas_w, canvas_h), self._bg_color)
        canvas.paste(r1, (p, p))
        canvas.paste(r2, (p + r1.width + p, p))

        r1.close()
        r2.close()
        return canvas

    @staticmethod
    def _resize_to_height(img: Image.Image, target_h: int) -> Image.Image:
        if img.height == target_h:
            return img.copy()
        ratio = target_h / img.height
        new_w = max(1, int(img.width * ratio))
        return img.resize((new_w, target_h), Image.Resampling.LANCZOS)


# ============================================================
# 管道编排（依赖倒置：依赖抽象而非具体实现）
# ============================================================


class LabelPipeline:
    """标签处理管道：检测 -> 分组汇总 -> 裁剪拼接"""

    def __init__(
        self,
        detector: LabelDetector,
        cropper: LabelCropper,
        grouper: LabelGrouper,
        stitcher: LabelStitcher,
    ):
        self._detector = detector
        self._cropper = cropper
        self._grouper = grouper
        self._stitcher = stitcher

    def detect_and_group(
        self, image_paths: list[Path]
    ) -> dict[LabelType, list[DetectedLabel]]:
        """检测所有图片中的标签并按类型汇总分组"""
        all_labels: list[DetectedLabel] = []
        for path in image_paths:
            if not path.exists():
                logger.warning("图片不存在，跳过: %s", path)
                continue
            all_labels.extend(self._detector.detect(path))

        groups = self._grouper.group(all_labels)
        for lt, labels in groups.items():
            logger.info("类型 [%s] 共检测到 %d 个标签", lt.value, len(labels))
        return groups

    def stitch_groups(
        self,
        groups: dict[LabelType, list[DetectedLabel]],
        output_dir: Path,
    ) -> dict[str, Path]:
        """对每种类型选取两张标签进行裁剪拼接，输出到指定目录"""
        output_dir.mkdir(parents=True, exist_ok=True)
        results: dict[str, Path] = {}

        for label_type, labels in groups.items():
            pair = self._select_pair(labels)
            if pair is None:
                logger.warning(
                    "类型 [%s] 不足两张标签，跳过拼接", label_type.value
                )
                continue

            img1 = self._cropper.crop(pair[0])
            img2 = self._cropper.crop(pair[1])
            stitched = self._stitcher.stitch(img1, img2)

            out_path = output_dir / f"{label_type.value}_拼接.jpg"
            stitched.save(out_path, quality=95)

            img1.close()
            img2.close()
            stitched.close()

            results[label_type.value] = out_path
            logger.info("类型 [%s] 拼接完成 -> %s", label_type.value, out_path)

        return results

    def process(
        self, image_paths: list[Path], output_dir: Path
    ) -> dict[str, Path]:
        """完整流程：检测 -> 分组 -> 拼接"""
        groups = self.detect_and_group(image_paths)
        if not groups:
            logger.warning("未检测到任何标签")
            return {}
        return self.stitch_groups(groups, output_dir)

    @staticmethod
    def _select_pair(
        labels: list[DetectedLabel],
    ) -> tuple[DetectedLabel, DetectedLabel] | None:
        if len(labels) < 2:
            return None
        # 优先选来自不同图片的标签
        for i, a in enumerate(labels):
            for b in labels[i + 1 :]:
                if a.source_image != b.source_image:
                    return (a, b)
        return (labels[0], labels[1])


# ============================================================
# CLI 入口
# ============================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="商品图标签检测、裁剪与拼接工具"
    )
    parser.add_argument("images", nargs="+", help="输入图片路径")
    parser.add_argument("-o", "--output", default="output", help="输出目录")
    parser.add_argument(
        "--base-url", required=True, help="Qwen-VL API base URL"
    )
    parser.add_argument("--api-key", default="EMPTY", help="API key")
    parser.add_argument(
        "--model", default="qwen-vl-plus", help="模型名称"
    )
    parser.add_argument(
        "--bg-color",
        default="245,245,245",
        help="拼接背景色 R,G,B（默认浅灰）",
    )
    parser.add_argument(
        "--padding", type=int, default=20, help="拼接内边距像素"
    )
    args = parser.parse_args()

    bg_parts = [int(c) for c in args.bg_color.split(",")]
    bg = (bg_parts[0], bg_parts[1], bg_parts[2])

    detector = QwenVLDetector(args.base_url, args.api_key, args.model)
    pipeline = LabelPipeline(
        detector=detector,
        cropper=LabelCropper(),
        grouper=LabelGrouper(),
        stitcher=LabelStitcher(bg_color=bg, padding=args.padding),
    )

    paths = [Path(p) for p in args.images]
    results = pipeline.process(paths, Path(args.output))

    if results:
        print("\n拼接结果:")
        for label_type, path in results.items():
            print(f"  {label_type}: {path}")
    else:
        print("\n未生成任何拼接结果")


if __name__ == "__main__":
    main()
