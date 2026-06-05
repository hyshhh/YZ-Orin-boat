"""
ShipDetector — 基于 TensorRT YOLO 的船只检测与跟踪

使用 TensorRT 引擎进行 YOLO 推理，支持 Orin 平台加速。
输出带 track ID 的检测框。
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)

# TensorRT 相关导入（延迟导入，避免非 Orin 环境报错）
_trt = None
_pycuda_driver = None
_pycuda_autoinit = None


def _import_trt():
    """延迟导入 TensorRT 和 PyCUDA"""
    global _trt, _pycuda_driver, _pycuda_autoinit
    if _trt is None:
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            _trt = trt
            _pycuda_driver = cuda
            _pycuda_autoinit = pycuda.autoinit
            logger.info("TensorRT 和 PyCUDA 导入成功")
        except ImportError as e:
            logger.error("TensorRT/PyCUDA 导入失败: %s", e)
            raise


@dataclass
class Detection:
    """单个检测结果。"""
    track_id: int
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    crop: np.ndarray | None = None


class HostDeviceMem:
    """TensorRT 内存管理辅助类"""
    def __init__(self, host_mem: np.ndarray, device_mem: Any):
        self.host = host_mem
        self.device = device_mem


def _build_tracker_yaml(tracker_type: str, tracker_params: dict[str, Any] | None) -> str:
    if not tracker_params:
        return f"{tracker_type}.yaml"
    cfg: dict[str, Any] = {"tracker_type": tracker_type}
    cfg.update(tracker_params)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", prefix=f"{tracker_type}_", delete=False, encoding="utf-8")
    yaml.dump(cfg, tmp, default_flow_style=False, allow_unicode=True)
    tmp.close()
    return tmp.name


class ShipDetector:
    """TensorRT YOLO 船只检测器（带原生跟踪）。"""

    def __init__(
        self,
        model_path: str = "yolov8n.engine",
        device: str = "0",
        input_size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        tracker_type: str = "bytetrack",
        tracker_params: dict[str, Any] | None = None,
        classes: list[int] | None = None,
    ):
        _import_trt()

        self._model_path = str(model_path)
        self._device = str(device)
        self._input_size = int(input_size)
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold
        self._classes = classes
        self._tracker_yaml = _build_tracker_yaml(tracker_type, tracker_params)
        self._tracker_type = tracker_type
        self._tracker_tmp_file: str | None = self._tracker_yaml if self._tracker_yaml != f"{tracker_type}.yaml" else None

        # TensorRT 引擎初始化
        self.logger = _trt.Logger(_trt.Logger.WARNING)
        self.engine = None
        self.context = None
        self.stream = None
        self.trt10 = False
        self.input_name = ""
        self.input_shape: Tuple[int, ...] = (1, 3, self._input_size, self._input_size)
        self.input_dtype = np.float32
        self.inputs: Dict[str, HostDeviceMem] = {}
        self.outputs: Dict[str, HostDeviceMem] = {}
        self.output_shapes: Dict[str, Tuple[int, ...]] = {}
        self.bindings: List[int] = []
        self.debug = os.environ.get("TRT_DETECTOR_DEBUG", "0").strip().lower() in {"1", "true", "yes", "y"}
        self._debug_printed = False

        # 类别过滤配置
        env_ids = os.environ.get("TRT_SHIP_CLASS_IDS", "").strip()
        self.keep_all_classes = False
        if classes is not None:
            self.ship_class_ids = [int(x) for x in classes]
        elif env_ids:
            if env_ids.lower() in {"all", "*", "-1"}:
                self.ship_class_ids = None
                self.keep_all_classes = True
            else:
                self.ship_class_ids = [int(x.strip()) for x in env_ids.split(",") if x.strip()]
        else:
            self.ship_class_ids = None

        self._load_engine()
        self._allocate_buffers()

        # 预热
        try:
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.detect(dummy, 0)
            logger.info("TensorRT 引擎预热完成")
        except Exception as e:
            logger.warning("TensorRT 预热失败（不影响后续使用）: %s", e)

        logger.info("TensorRT YOLO 模型加载完成: %s", model_path)

    def _load_engine(self):
        """加载 TensorRT 引擎"""
        with open(self._model_path, "rb") as f, _trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self._model_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self.stream = _pycuda_driver.Stream()
        self.trt10 = hasattr(self.engine, "num_io_tensors")

    @staticmethod
    def _volume(shape: Tuple[int, ...]) -> int:
        v = 1
        for x in shape:
            v *= int(x)
        return int(v)

    def _resolve_shape(self, shape: Tuple[int, ...], is_input: bool) -> Tuple[int, ...]:
        shape = tuple(int(x) for x in shape)
        if is_input and any(x <= 0 for x in shape):
            return (1, 3, self._input_size, self._input_size)
        if any(x <= 0 for x in shape):
            raise RuntimeError(f"Dynamic output shape is unresolved: {shape}")
        return shape

    def _allocate_buffers(self) -> None:
        """分配 TensorRT 输入输出缓冲区"""
        if self.trt10:
            tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]

            for name in tensor_names:
                mode = self.engine.get_tensor_mode(name)
                if mode == _trt.TensorIOMode.INPUT:
                    self.input_name = name
                    shape = self._resolve_shape(tuple(self.engine.get_tensor_shape(name)), is_input=True)
                    self.input_shape = shape
                    if hasattr(self.context, "set_input_shape"):
                        self.context.set_input_shape(name, shape)

            for name in tensor_names:
                mode = self.engine.get_tensor_mode(name)
                dtype = _trt.nptype(self.engine.get_tensor_dtype(name))
                if mode == _trt.TensorIOMode.INPUT:
                    shape = tuple(self.context.get_tensor_shape(name))
                    shape = self._resolve_shape(shape, is_input=True)
                    self.input_shape = shape
                    self.input_dtype = dtype
                    host = _pycuda_driver.pagelocked_empty(self._volume(shape), dtype)
                    device = _pycuda_driver.mem_alloc(host.nbytes)
                    self.inputs[name] = HostDeviceMem(host, device)
                    self.context.set_tensor_address(name, int(device))
                else:
                    shape = tuple(self.context.get_tensor_shape(name))
                    shape = self._resolve_shape(shape, is_input=False)
                    host = _pycuda_driver.pagelocked_empty(self._volume(shape), dtype)
                    device = _pycuda_driver.mem_alloc(host.nbytes)
                    self.outputs[name] = HostDeviceMem(host, device)
                    self.output_shapes[name] = shape
                    self.context.set_tensor_address(name, int(device))
            return

        self.bindings = [0] * int(self.engine.num_bindings)
        for idx in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(idx)
            is_input = bool(self.engine.binding_is_input(idx))
            if is_input:
                self.input_name = name
                shape = self._resolve_shape(tuple(self.engine.get_binding_shape(idx)), is_input=True)
                self.input_shape = shape
                if any(int(x) <= 0 for x in self.engine.get_binding_shape(idx)):
                    self.context.set_binding_shape(idx, shape)

        for idx in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(idx)
            is_input = bool(self.engine.binding_is_input(idx))
            dtype = _trt.nptype(self.engine.get_binding_dtype(idx))
            shape = tuple(self.context.get_binding_shape(idx))
            shape = self._resolve_shape(shape, is_input=is_input)
            host = _pycuda_driver.pagelocked_empty(self._volume(shape), dtype)
            device = _pycuda_driver.mem_alloc(host.nbytes)
            self.bindings[idx] = int(device)
            if is_input:
                self.inputs[name] = HostDeviceMem(host, device)
                self.input_shape = shape
                self.input_dtype = dtype
            else:
                self.outputs[name] = HostDeviceMem(host, device)
                self.output_shapes[name] = shape

    @staticmethod
    def _bbox_area(bbox: List[int]) -> float:
        return float(max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1]))

    @staticmethod
    def _bbox_iou(box_a: List[int], box_b: List[int]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter_area = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
        return inter_area / max(1.0, ShipDetector._bbox_area(box_a) + ShipDetector._bbox_area(box_b) - inter_area)

    @staticmethod
    def _intersection_over_small(box_a: List[int], box_b: List[int]) -> float:
        """Intersection / smaller-box area."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter_area = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
        area_a = max(1.0, ShipDetector._bbox_area(box_a))
        area_b = max(1.0, ShipDetector._bbox_area(box_b))
        return inter_area / min(area_a, area_b)

    @staticmethod
    def _center_in_box(inner: List[int], outer: List[int]) -> bool:
        x1, y1, x2, y2 = inner
        ox1, oy1, ox2, oy2 = outer
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return ox1 <= cx <= ox2 and oy1 <= cy <= oy2

    def _suppress_duplicates(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        kept: List[Dict[str, Any]] = []
        ordered = sorted(
            detections,
            key=lambda item: (float(item["det_conf"]), self._bbox_area(item["bbox"])),
            reverse=True,
        )
        for det in ordered:
            box = det["bbox"]
            area = self._bbox_area(box)
            duplicate = False
            for other in kept:
                other_box = other["bbox"]
                other_area = self._bbox_area(other_box)
                iou = self._bbox_iou(box, other_box)
                ios = self._intersection_over_small(box, other_box)
                smaller_ratio = min(area, other_area) / max(1.0, max(area, other_area))

                if iou >= 0.45:
                    duplicate = True
                    break
                if ios >= 0.72 and smaller_ratio <= 0.70:
                    duplicate = True
                    break
                if area < other_area * 0.55 and self._center_in_box(box, other_box):
                    duplicate = True
                    break
            if not duplicate:
                kept.append(det)
        return kept

    @staticmethod
    def _compute_crop_bbox(bbox: List[int], width: int, height: int, expand_ratio: float = 0.20) -> List[int]:
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        dx = int(bw * expand_ratio)
        dy = int(bh * expand_ratio)
        x1 = max(0, min(x1 - dx, width - 1))
        y1 = max(0, min(y1 - dy, height - 1))
        x2 = max(x1 + 1, min(x2 + dx, width))
        y2 = max(y1 + 1, min(y2 + dy, height))
        return [x1, y1, x2, y2]

    def crop_from_frame(self, frame: Any, bbox: List[int], expand_ratio: float = 0.20) -> Dict[str, Any]:
        height, width = frame.shape[:2]
        crop_bbox = self._compute_crop_bbox(bbox, width, height, expand_ratio=expand_ratio)
        x1, y1, x2, y2 = crop_bbox
        crop_bgr = frame[y1:y2, x1:x2].copy()
        return {"image": crop_bgr, "path": None, "crop_bbox": crop_bbox}

    def _letterbox(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        h, w = frame.shape[:2]
        new_h = int(self.input_shape[2]) if len(self.input_shape) == 4 else self._input_size
        new_w = int(self.input_shape[3]) if len(self.input_shape) == 4 else self._input_size
        r = min(new_w / max(1, w), new_h / max(1, h))
        resized_w, resized_h = int(round(w * r)), int(round(h * r))
        resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((new_h, new_w, 3), 114, dtype=np.uint8)
        pad_x = (new_w - resized_w) // 2
        pad_y = (new_h - resized_h) // 2
        canvas[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized
        return canvas, r, (float(pad_x), float(pad_y))

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        img, ratio, pad = self._letterbox(frame)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        img = img[None, ...]
        return img.astype(self.input_dtype, copy=False), ratio, pad

    def _infer(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        inp = next(iter(self.inputs.values()))
        np.copyto(inp.host, input_tensor.ravel())
        _pycuda_driver.memcpy_htod_async(inp.device, inp.host, self.stream)

        if self.trt10:
            ok = self.context.execute_async_v3(stream_handle=self.stream.handle)
        else:
            ok = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        if not ok:
            raise RuntimeError("TensorRT inference failed")

        for name, mem in self.outputs.items():
            _pycuda_driver.memcpy_dtoh_async(mem.host, mem.device, self.stream)
        self.stream.synchronize()

        outputs: List[np.ndarray] = []
        for name, mem in self.outputs.items():
            outputs.append(np.array(mem.host).reshape(self.output_shapes[name]).copy())
        return outputs

    @staticmethod
    def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        out = np.empty_like(boxes, dtype=np.float32)
        out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        return out

    @staticmethod
    def _maybe_scale_normalized_boxes(boxes: np.ndarray, input_w: int, input_h: int) -> np.ndarray:
        if boxes.size and np.nanmax(np.abs(boxes)) <= 2.0:
            boxes = boxes.copy()
            boxes[:, [0, 2]] *= float(input_w)
            boxes[:, [1, 3]] *= float(input_h)
        return boxes

    def _decode_yolo_output(self, outputs: List[np.ndarray], min_conf: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        pred = max(outputs, key=lambda x: x.size)
        raw_shape = tuple(pred.shape)
        pred = np.asarray(pred)
        pred = np.squeeze(pred)

        if pred.ndim == 1:
            pred = pred.reshape(1, -1)
        if pred.ndim != 2:
            pred = pred.reshape(-1, pred.shape[-1])

        if pred.shape[0] < pred.shape[1] and pred.shape[0] <= 512:
            pred = pred.T

        if pred.shape[1] < 5:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                0,
            )

        input_h = int(self.input_shape[2]) if len(self.input_shape) == 4 else self._input_size
        input_w = int(self.input_shape[3]) if len(self.input_shape) == 4 else self._input_size

        if pred.shape[1] in (6, 7):
            boxes = pred[:, 0:4].astype(np.float32)
            scores = pred[:, 4].astype(np.float32)
            class_ids = pred[:, 5].astype(np.int32)
            boxes = self._maybe_scale_normalized_boxes(boxes, input_w, input_h)
            keep = scores >= float(min_conf)
            if self.debug and not self._debug_printed:
                print(f"[TRT debug] raw_output_shape={raw_shape}, decoded_as=nms, rows={pred.shape[0]}, keep={int(np.sum(keep))}, max_score={float(np.nanmax(scores)) if scores.size else 0:.4f}")
            return boxes[keep], scores[keep], class_ids[keep], pred.shape[1] - 5

        boxes_xywh = pred[:, 0:4].astype(np.float32)
        scores_part = pred[:, 4:].astype(np.float32)
        num_score_cols = int(scores_part.shape[1])

        if num_score_cols <= 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                0,
            )

        use_objectness = False
        total_cols = int(pred.shape[1])
        if total_cols == 85:
            use_objectness = True
        if os.environ.get("TRT_YOLO_HAS_OBJECTNESS", "0").strip().lower() in {"1", "true", "yes", "y"}:
            use_objectness = True

        if use_objectness and num_score_cols >= 2:
            obj = np.clip(scores_part[:, 0], 0.0, 1.0)
            cls_scores = scores_part[:, 1:]
            class_ids = np.argmax(cls_scores, axis=1).astype(np.int32)
            scores = obj * cls_scores[np.arange(cls_scores.shape[0]), class_ids]
            effective_score_cols = int(cls_scores.shape[1])
            decoded_as = "xywh_obj_cls"
        else:
            class_ids = np.argmax(scores_part, axis=1).astype(np.int32)
            scores = scores_part[np.arange(scores_part.shape[0]), class_ids]
            effective_score_cols = int(scores_part.shape[1])
            decoded_as = "xywh_cls"

        keep = scores >= float(min_conf)
        boxes = self._xywh_to_xyxy(boxes_xywh[keep])
        boxes = self._maybe_scale_normalized_boxes(boxes, input_w, input_h)

        if self.debug and not self._debug_printed:
            max_score = float(np.nanmax(scores)) if scores.size else 0.0
            top_cls = int(class_ids[int(np.nanargmax(scores))]) if scores.size else -1
            print(
                f"[TRT debug] raw_output_shape={raw_shape}, after={tuple(pred.shape)}, "
                f"decoded_as={decoded_as}, score_cols={effective_score_cols}, "
                f"max_score={max_score:.4f}, top_cls={top_cls}, keep={int(np.sum(keep))}"
            )

        return boxes, scores[keep].astype(np.float32), class_ids[keep], effective_score_cols

    def _class_filter(self, class_ids: np.ndarray, num_score_cols: int) -> np.ndarray:
        if self.keep_all_classes:
            return np.ones_like(class_ids, dtype=bool)
        if self.ship_class_ids is not None:
            allow = set(int(x) for x in self.ship_class_ids)
            return np.array([int(c) in allow for c in class_ids], dtype=bool)

        if num_score_cols <= 1:
            return np.ones_like(class_ids, dtype=bool)

        if num_score_cols >= 80:
            return class_ids == 8

        return np.ones_like(class_ids, dtype=bool)

    def _postprocess(
        self,
        outputs: List[np.ndarray],
        ratio: float,
        pad: Tuple[float, float],
        frame_shape: Tuple[int, int],
        min_conf: float,
    ) -> List[Dict[str, Any]]:
        boxes, scores, class_ids, num_score_cols = self._decode_yolo_output(outputs, min_conf=min_conf)
        if boxes.size == 0:
            if self.debug and not self._debug_printed:
                print("[TRT debug] no boxes after confidence filtering")
                self._debug_printed = True
            return []

        cls_keep = self._class_filter(class_ids, num_score_cols=num_score_cols)
        if self.debug and not self._debug_printed:
            kept_before = int(boxes.shape[0])
            kept_after = int(np.sum(cls_keep))
            unique_cls = sorted(set(int(x) for x in class_ids[:50]))
            print(f"[TRT debug] class_filter before={kept_before}, after={kept_after}, unique_cls_first50={unique_cls}, num_score_cols={num_score_cols}")

        boxes, scores, class_ids = boxes[cls_keep], scores[cls_keep], class_ids[cls_keep]
        if boxes.size == 0:
            if self.debug and not self._debug_printed:
                print("[TRT debug] no boxes after class filtering. Try: export TRT_SHIP_CLASS_IDS=all")
                self._debug_printed = True
            return []

        pad_x, pad_y = pad
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / max(1e-6, ratio)
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / max(1e-6, ratio)

        h, w = frame_shape[:2]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)

        nms_boxes = []
        nms_scores = []
        for b, s in zip(boxes, scores):
            x1, y1, x2, y2 = b.tolist()
            bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
            if bw < 2 or bh < 2:
                continue
            nms_boxes.append([int(x1), int(y1), int(bw), int(bh)])
            nms_scores.append(float(s))

        if not nms_boxes:
            if self.debug and not self._debug_printed:
                print("[TRT debug] no boxes after valid-size filtering")
                self._debug_printed = True
            return []

        idxs = cv2.dnn.NMSBoxes(nms_boxes, nms_scores, float(min_conf), self._iou_threshold)
        if len(idxs) == 0:
            if self.debug and not self._debug_printed:
                print("[TRT debug] no boxes after NMS")
                self._debug_printed = True
            return []

        idxs = np.array(idxs).reshape(-1).tolist()
        detections: List[Dict[str, Any]] = []
        for i in idxs:
            x, y, bw, bh = nms_boxes[i]
            detections.append({
                "bbox": [int(x), int(y), int(x + bw), int(y + bh)],
                "det_conf": float(nms_scores[i]),
                "class_name": "ship",
                "class_id": int(class_ids[i]),
            })

        if self.debug and not self._debug_printed:
            print(f"[TRT debug] final_detections={detections[:3]}")
            self._debug_printed = True

        return self._suppress_duplicates(detections)

    def detect(self, frame: np.ndarray, frame_id: int = 0, min_conf: float | None = None) -> list[Detection]:
        """TensorRT YOLO 检测"""
        threshold = self._conf_threshold if min_conf is None else float(min_conf)
        try:
            input_tensor, ratio, pad = self._preprocess(frame)
            outputs = self._infer(input_tensor)
            raw_detections = self._postprocess(outputs, ratio, pad, frame.shape[:2], min_conf=threshold)

            # 转换为 Detection 对象（带跟踪 ID）
            detections: list[Detection] = []
            for i, det in enumerate(raw_detections):
                x1, y1, x2, y2 = det["bbox"]
                conf = det["det_conf"]

                # 裁剪（加 padding）
                h, w = frame.shape[:2]
                pad_size = 20
                cx1, cy1 = max(0, x1 - pad_size), max(0, y1 - pad_size)
                cx2, cy2 = min(w, x2 + pad_size), min(h, y2 + pad_size)
                crop = frame[cy1:cy2, cx1:cx2].copy()

                crop_h, crop_w = crop.shape[:2]
                if crop_w < 80 or crop_h < 80:
                    continue

                # 尺寸归一化：统一到 256~512px
                target_min, target_max = 256, 512
                max_dim = max(crop_w, crop_h)
                if max_dim < target_min:
                    scale = target_min / max_dim
                    crop = cv2.resize(crop, (int(crop_w * scale), int(crop_h * scale)), interpolation=cv2.INTER_LINEAR)
                elif max_dim > target_max:
                    scale = target_max / max_dim
                    crop = cv2.resize(crop, (int(crop_w * scale), int(crop_h * scale)), interpolation=cv2.INTER_AREA)

                # 使用检测索引作为临时 track_id（后续由 TrackManager 分配）
                track_id = i + 1
                detections.append(Detection(track_id=track_id, bbox=(x1, y1, x2, y2), confidence=conf, crop=crop))

            return detections

        except Exception as e:
            logger.error("TensorRT YOLO 检测异常 (frame=%d): %s", frame_id, e)
            return []

    def cleanup(self) -> None:
        """清理资源"""
        if self._tracker_tmp_file:
            try:
                Path(self._tracker_tmp_file).unlink(missing_ok=True)
            except Exception:
                pass
            self._tracker_tmp_file = None

    def __del__(self) -> None:
        self.cleanup()
