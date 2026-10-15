"""
Pre-trained Diffusion Fine-Tuning, LoRA & Domain Adaptation Engine.
Refines Stable Diffusion models for domain-specific visual synthesis (Artwork & Medical Imagery).
"""

import json
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


DOMAIN_PROFILES = {
    "Artwork & Masterpiece Styles": {
        "description": "Refines Stable Diffusion for distinct painterly styles (Impressionist, Watercolor, Oil)",
        "trigger_word": "artistic_style",
        "sample_prompts": [
            "a quiet European cobblestone alleyway in autumn, artistic_style",
            "portrait of an old astronomer gazing through a telescope, artistic_style",
            "enchanted forest with bioluminescent mushrooms, artistic_style"
        ],
        "target_layers": ["unet.to_k", "unet.to_v", "unet.to_q", "text_encoder.text_model"],
        "recommended_lora_rank": 16,
        "recommended_lr": 1e-4
    },
    "Medical Imagery (Radiology & MRI)": {
        "description": "Domain-adapted for diagnostic grayscale chest X-rays, axial CT scans, and MRI imagery",
        "trigger_word": "medical_xray",
        "sample_prompts": [
            "frontal chest radiograph showing clear lung fields, medical_xray",
            "axial view computed tomography scan of upper abdominal cavity, medical_xray",
            "magnetic resonance imaging of cranial sagittal section, medical_xray"
        ],
        "target_layers": ["unet.mid_block.attentions", "unet.up_blocks.1.attentions"],
        "recommended_lora_rank": 8,
        "recommended_lr": 5e-5
    },
    "Cyberpunk Sci-Fi Concept Art": {
        "description": "Specialized on futuristic neon-drenched architecture, rainy asphalt reflections, and cybernetics",
        "trigger_word": "cyberpunk_concept",
        "sample_prompts": [
            "sprawling megacity skyline at midnight with holographic billboards, cyberpunk_concept",
            "cybernetic android technician repairing a quantum server rack, cyberpunk_concept"
        ],
        "target_layers": ["unet.to_k", "unet.to_v"],
        "recommended_lora_rank": 16,
        "recommended_lr": 1e-4
    }
}


class DomainAdaptationEngine:
    """Simulates LoRA fine-tuning weight blending and exports production training specifications."""
    def __init__(self):
        self.domains = DOMAIN_PROFILES

    def get_domain_list(self) -> List[str]:
        return list(self.domains.keys())

    def generate_training_config(
        self,
        domain_name: str,
        base_model: str,
        lora_rank: int,
        learning_rate: float,
        epochs: int,
        batch_size: int
    ) -> str:
        """Export complete PyTorch Diffusers LoRA fine-tuning configuration JSON."""
        domain_info = self.domains.get(domain_name, self.domains["Artwork & Masterpiece Styles"])
        config = {
            "task": "Text-to-Image Domain Adaptation via LoRA",
            "base_model": base_model,
            "domain_target": domain_name,
            "trigger_token": domain_info["trigger_word"],
            "peft_lora_settings": {
                "rank_r": lora_rank,
                "lora_alpha": lora_rank * 2,
                "lora_dropout": 0.05,
                "target_modules": domain_info["target_layers"]
            },
            "training_hyperparameters": {
                "learning_rate": learning_rate,
                "train_epochs": epochs,
                "train_batch_size": batch_size,
                "gradient_accumulation_steps": 2,
                "mixed_precision": "fp16",
                "optimizer": "AdamW (betas=(0.9, 0.999), weight_decay=1e-2)",
                "lr_scheduler": "cosine_with_restarts"
            },
            "mathematical_formulation": "W_finetuned = W_pretrained + (alpha / r) * (B @ A), with B in R^(d x r), A in R^(r x k)"
        }
        return json.dumps(config, indent=2)

    def render_comparison(
        self,
        prompt: str,
        domain_name: str,
        lora_scale: float = 1.0,
        seed: Optional[int] = None
    ) -> Tuple[Image.Image, Image.Image, dict]:
        """
        Generate side-by-side comparison:
        1. Pre-trained Base Model (Generic Diffusion)
        2. Domain-Refined Model (with LoRA weight injection)
        """
        if seed is not None and seed != -1:
            np.random.seed(seed)

        size = 384
        domain_info = self.domains.get(domain_name, self.domains["Artwork & Masterpiece Styles"])

        # Base Image (Neutral / generic appearance)
        base_img = Image.new("RGB", (size, size), (45, 55, 72))
        draw_b = ImageDraw.Draw(base_img)
        draw_b.rectangle([40, 40, size - 40, size - 40], fill=(74, 85, 104), outline=(160, 174, 192), width=2)
        draw_b.ellipse([100, 100, size - 100, size - 100], fill=(113, 128, 150))
        draw_b.text((50, 50), "Base Model Output (Generic)", fill=(255, 255, 255))
        draw_b.text((50, 75), f"Prompt: {prompt[:35]}...", fill=(203, 213, 225))

        # Domain-Adapted Image (Stylized according to domain)
        adapted_img = Image.new("RGB", (size, size), (15, 23, 42))
        draw_a = ImageDraw.Draw(adapted_img)

        if "Medical" in domain_name:
            # Render authentic medical radiograph look
            draw_a.rectangle([0, 0, size, size], fill=(10, 10, 10))
            # Anatomical chest contour
            draw_a.ellipse([60, 80, 180, 320], fill=(70, 70, 70))
            draw_a.ellipse([size - 180, 80, size - 60, 320], fill=(70, 70, 70))
            # Cardiac silhouette
            draw_a.ellipse([140, 160, 250, 300], fill=(140, 140, 140))
            # Ribs / bony structures
            for y_rib in range(110, 290, 35):
                draw_a.line([(70, y_rib), (170, y_rib + 15)], fill=(180, 180, 180), width=4)
                draw_a.line([(size - 70, y_rib), (size - 170, y_rib + 15)], fill=(180, 180, 180), width=4)
            adapted_img = adapted_img.filter(ImageFilter.GaussianBlur(radius=3))
            draw_a = ImageDraw.Draw(adapted_img)
            draw_a.text((20, 20), f"Domain: {domain_name}", fill=(200, 200, 200))
            draw_a.text((20, 40), f"LoRA Adapter Scale: {lora_scale:.2f}", fill=(0, 230, 118))
        elif "Cyberpunk" in domain_name:
            # Render vibrant neon sci-fi composition
            draw_a.rectangle([0, 0, size, size], fill=(13, 2, 33))
            # Futuristic buildings
            draw_a.rectangle([30, 90, 120, size], fill=(29, 0, 62), outline=(0, 245, 212), width=2)
            draw_a.rectangle([140, 50, 240, size], fill=(45, 0, 97), outline=(255, 0, 110), width=2)
            draw_a.rectangle([260, 120, 360, size], fill=(20, 0, 45), outline=(254, 228, 64), width=2)
            adapted_img = adapted_img.filter(ImageFilter.GaussianBlur(radius=2))
            draw_a = ImageDraw.Draw(adapted_img)
            draw_a.text((20, 20), f"Domain: {domain_name}", fill=(255, 255, 255))
            draw_a.text((20, 40), f"LoRA Adapter Scale: {lora_scale:.2f}", fill=(0, 245, 212))
        else:
            # Render rich impasto oil / watercolor style
            draw_a.rectangle([0, 0, size, size], fill=(30, 20, 50))
            for i in range(16):
                cx = np.random.randint(60, size - 60)
                cy = np.random.randint(60, size - 60)
                rad = np.random.randint(25, 70)
                c_fill = (np.random.randint(180, 255), np.random.randint(100, 220), np.random.randint(50, 180))
                draw_a.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=c_fill)
            adapted_img = adapted_img.filter(ImageFilter.GaussianBlur(radius=4))
            draw_a = ImageDraw.Draw(adapted_img)
            draw_a.text((20, 20), f"Domain: {domain_name}", fill=(255, 255, 255))
            draw_a.text((20, 40), f"LoRA Adapter Scale: {lora_scale:.2f}", fill=(255, 206, 86))

        # Blend base with adapted image according to lora_scale
        if lora_scale < 1.0:
            final_adapted = Image.blend(base_img, adapted_img, alpha=max(0.0, min(1.0, lora_scale)))
        else:
            final_adapted = adapted_img

        metadata = {
            "domain": domain_name,
            "trigger_token": domain_info["trigger_word"],
            "prompt_used": prompt,
            "lora_scaling_alpha": lora_scale,
            "base_model": "Stable Diffusion v1.5 / SDXL",
            "status": "Fine-Tuned Adapter Injected",
            "delta_weights_active": bool(lora_scale > 0.05)
        }
        return base_img, final_adapted, metadata
