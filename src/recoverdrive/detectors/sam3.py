"""SAM 3 text-prompted camera detector (via transformers), with a content-addressed result cache."""

from __future__ import annotations

import threading

import numpy as np
import torch
from PIL import Image

from recoverdrive.cache import KVCache, digest, image_digest
from recoverdrive.types import Det2D


class Sam3Detector:
    def __init__(self, prompts: list[str], score_threshold: float = 0.5, model_id: str = "facebook/sam3",
                 device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        from transformers import Sam3Model, Sam3Processor

        self.prompts = list(prompts)
        self.score_threshold = score_threshold
        self.device = device
        self.dtype = dtype
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id, dtype=dtype).to(device).eval()

    @torch.inference_mode()
    def detect(self, image: np.ndarray) -> list[Det2D]:
        """Run the vision backbone once, then the text-conditioned decoder once per prompt."""
        pil = Image.fromarray(image)
        img_inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        vision_embeds = self.model.get_vision_features(pixel_values=img_inputs.pixel_values.to(self.dtype))
        dets = []
        for prompt in self.prompts:
            text_inputs = self.processor(text=prompt, return_tensors="pt").to(self.device)
            outputs = self.model(vision_embeds=vision_embeds, **text_inputs)
            res = self.processor.post_process_object_detection(outputs, threshold=self.score_threshold,
                                                               target_sizes=[pil.size[::-1]])[0]
            for box, score in zip(res["boxes"].float().cpu().numpy(), res["scores"].float().cpu().numpy()):
                dets.append(Det2D(box=box.astype(float), label=prompt, score=float(score)))
        return dets

    @torch.inference_mode()
    def detect_batched(self, image: np.ndarray) -> list[Det2D]:
        """E0 path: one batch item per prompt (runs the backbone once per prompt)."""
        pil = Image.fromarray(image)
        inputs = self.processor(images=[pil] * len(self.prompts), text=self.prompts, return_tensors="pt")
        inputs = inputs.to(self.device, dtype=self.dtype)
        outputs = self.model(**inputs)
        results = self.processor.post_process_object_detection(
            outputs, threshold=self.score_threshold, target_sizes=[pil.size[::-1]] * len(self.prompts))
        dets = []
        for prompt, res in zip(self.prompts, results):
            for box, score in zip(res["boxes"].float().cpu().numpy(), res["scores"].float().cpu().numpy()):
                dets.append(Det2D(box=box.astype(float), label=prompt, score=float(score)))
        return dets


class CachedDetector:
    """Thread-safe SAM 3 front end: cache lookup by image content; the model loads on the first miss only."""

    def __init__(self, cache: KVCache, prompts: list[str], score_threshold: float = 0.5,
                 model_id: str = "facebook/sam3", device: str = "cuda"):
        self.cache = cache
        self.prompts, self.score_threshold, self.model_id, self.device = list(prompts), score_threshold, model_id, device
        self._base = {"model": model_id, "prompts": self.prompts, "threshold": score_threshold, "v": 2}
        self._detector: Sam3Detector | None = None
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def key(self, image: np.ndarray, image_id: str | None = None) -> str:
        return "sam3:" + digest({**self._base, "image": image_id or image_digest(image)})

    def detect(self, image: np.ndarray, image_id: str | None = None) -> list[Det2D]:
        key = self.key(image, image_id)
        hit = self.cache.get(key)
        if hit is None:
            with self._lock:
                hit = self.cache.get(key)
                if hit is None:
                    if self._detector is None:
                        self._detector = Sam3Detector(self.prompts, self.score_threshold, self.model_id, self.device)
                    dets = self._detector.detect(image)
                    hit = [{"box": d.box.tolist(), "label": d.label, "score": d.score} for d in dets]
                    self.cache.set(key, hit)
                    self.misses += 1
                    return dets
        self.hits += 1
        return [Det2D(box=np.array(d["box"], float), label=d["label"], score=d["score"]) for d in hit]
