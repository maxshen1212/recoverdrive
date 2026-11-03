"""SAM 3 text-prompted camera detector (via transformers)."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

from recoverdrive.types import Det2D


class Sam3Detector:
    def __init__(self, prompts: list[str], score_threshold: float = 0.5, model_id: str = "facebook/sam3",
                 device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.prompts = list(prompts)
        self.score_threshold = score_threshold
        self.device = device
        self.dtype = dtype
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id, dtype=dtype).to(device).eval()

    @torch.inference_mode()
    def detect(self, image: np.ndarray) -> list[Det2D]:
        """Detect every prompt class in one batched forward pass (one batch item per prompt)."""
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
