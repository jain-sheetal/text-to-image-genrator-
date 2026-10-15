"""
Comprehensive Unified Text-to-Image Generation Pipeline.
Integrates text preprocessing, semantic embeddings, conditional GANs,
and Stable Diffusion inference with VRAM optimizations and cloud fallback.
"""

import gc
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch

from src.cgan import CGANPipeline
from src.attention_gan import AttentionGANPipeline
from src.text_processor import TextPreprocessor


class UnifiedTextToImagePipeline:
    """
    End-to-End Production Pipeline coordinating:
    1. Text Tokenization & Embedding Extraction (HF Transformers)
    2. Model Selection:
       - Mode A: Stable Diffusion (Local CUDA / CPU with attention/VAE slicing)
       - Mode B: Cloud Serverless Engine (Fast, zero-VRAM cloud fallback)
       - Mode C: Conditional GAN (Shape & category synthesis)
       - Mode D: Attention GAN (Cross-attention token mapping)
    3. Performance & Memory Profiling
    4. Image & Metadata Archival
    """
    def __init__(self, device: str = "auto"):
        self.device = self._detect_device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        
        # Sub-modules
        self.text_processor = TextPreprocessor()
        self.cgan_pipeline = CGANPipeline(device=str(self.device))
        self.attention_pipeline = AttentionGANPipeline(device=str(self.device))
        
        # Local diffusion pipeline state (loaded on-demand to conserve RAM/VRAM)
        self.diffusion_pipe = None
        self.current_model_id = None
        self.current_scheduler = "euler_a"

    def _detect_device(self, req: str) -> torch.device:
        if req == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(req)

    def load_local_diffusion(self, model_id: str = "runwayml/stable-diffusion-v1-5") -> str:
        """Load local Stable Diffusion with aggressive VRAM optimizations for 4GB RTX 3050 GPUs."""
        try:
            from diffusers import StableDiffusionPipeline
            print(f"Loading local diffusion pipeline: {model_id} on {self.device}")
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                safety_checker=None,
                requires_safety_checker=False
            )
            # Enable memory optimizations
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()

            if self.device.type == "cuda":
                try:
                    pipe = pipe.to(self.device)
                except RuntimeError:
                    # Low VRAM fallback (RTX 3050 4GB)
                    pipe.enable_model_cpu_offload()
            else:
                pipe.enable_sequential_cpu_offload()

            self.diffusion_pipe = pipe
            self.current_model_id = model_id
            return f"Successfully loaded {model_id} on {self.device}"
        except Exception as e:
            return f"Local pipeline load notice: {str(e)}"

    def set_scheduler(self, scheduler_name: str) -> bool:
        """Switch denoising scheduler dynamically."""
        if self.diffusion_pipe is None:
            return False
        from diffusers import (
            EulerAncestralDiscreteScheduler,
            EulerDiscreteScheduler,
            DPMSolverMultistepScheduler,
            DDIMScheduler,
            LMSDiscreteScheduler
        )
        scheduler_map = {
            "euler_a": EulerAncestralDiscreteScheduler,
            "euler": EulerDiscreteScheduler,
            "ddim": DDIMScheduler,
            "dpm_solver": DPMSolverMultistepScheduler,
            "lms": LMSDiscreteScheduler
        }
        if scheduler_name in scheduler_map:
            s_class = scheduler_map[scheduler_name]
            self.diffusion_pipe.scheduler = s_class.from_config(self.diffusion_pipe.scheduler.config)
            self.current_scheduler = scheduler_name
            return True
        return False

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        model_type: str = "Stable Diffusion 1.5",
        scheduler: str = "euler_a",
        steps: int = 20,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None,
        hf_token: Optional[str] = None
    ) -> Tuple[Image.Image, dict]:
        """
        Unified generation routing:
        1. Preprocesses text tokens & embeddings.
        2. Routes to requested model engine.
        3. Formats comprehensive diagnostics.
        """
        if not prompt.strip():
            prompt = "a serene mountain landscape at sunrise, oil on canvas, highly detailed"

        start_time = time.time()
        # Step 1: Text Preprocessing & Embedding
        embedding, emb_stats = self.text_processor.encode_to_embedding(prompt)
        token_info = self.text_processor.tokenize(prompt)

        # Ensure seed consistency
        if seed is None or seed == -1:
            seed = int(torch.randint(0, 2**32, (1,)).item())
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Step 2: Route by Model Type
        image = None
        engine_used = ""

        # Route A: Conditional GAN
        if "Conditional GAN" in model_type:
            shape = "circle"
            for s in ["circle", "square", "triangle", "star", "hexagon", "heart", "diamond"]:
                if s in prompt.lower():
                    shape = s
                    break
            image, meta = self.cgan_pipeline.generate(shape, seed=seed)
            engine_used = "Conditional GAN (CGAN Engine)"

        # Route B: Attention-Enhanced GAN
        elif "Attention GAN" in model_type:
            image, _, meta = self.attention_pipeline.generate_with_attention(prompt, seed=seed)
            engine_used = "Attention-Enhanced GAN (SAGAN & AttnGAN Engine)"

        # Route C: Stable Diffusion (Cloud Serverless / Local Hybrid)
        else:
            # Check if local pipeline is explicitly requested and loaded
            if "Local" in model_type:
                if self.diffusion_pipe is None:
                    # Attempt quick local loading
                    target_id = "stabilityai/sd-turbo" if "Turbo" in model_type else "runwayml/stable-diffusion-v1-5"
                    load_msg = self.load_local_diffusion(target_id)
                if self.diffusion_pipe is not None:
                    self.set_scheduler(scheduler)
                    generator = torch.Generator(device=self.device).manual_seed(seed)
                    with torch.inference_mode():
                        res = self.diffusion_pipe(
                            prompt=prompt,
                            negative_prompt=negative_prompt if negative_prompt else None,
                            width=(width // 8) * 8,
                            height=(height // 8) * 8,
                            num_inference_steps=steps,
                            guidance_scale=guidance_scale,
                            generator=generator
                        )
                    image = res.images[0]
                    engine_used = f"Local PyTorch Diffusers ({self.device})"
                else:
                    # Fallback to high-speed cloud diffusion
                    image, engine_used = self._generate_cloud_or_fallback(prompt, negative_prompt, hf_token, seed, width, height)
            else:
                # Default: High-speed Cloud Diffusion Engine (Photorealistic FLUX / SDXL)
                image, engine_used = self._generate_cloud_or_fallback(prompt, negative_prompt, hf_token, seed, width, height)

        elapsed = round(time.time() - start_time, 2)
        memory_stats = self.get_vram_info()

        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "model_type": model_type,
            "engine": engine_used,
            "scheduler": scheduler,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "resolution": f"{width}x{height}",
            "seed": seed,
            "generation_time_sec": elapsed,
            "active_tokens": token_info["token_count"],
            "embedding_norm": emb_stats["l2_norm"],
            "vram_info": memory_stats
        }

        # Clean cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return image, metadata

    def _generate_cloud_or_fallback(
        self,
        prompt: str,
        negative_prompt: str,
        hf_token: Optional[str],
        seed: int,
        width: int = 512,
        height: int = 512
    ) -> Tuple[Image.Image, str]:
        """High-speed real AI generation via Cloud Diffusion Engine (FLUX / SDXL) or Hugging Face."""
        # 1. Try Hugging Face InferenceClient if token is provided
        if hf_token and hf_token.strip():
            try:
                from huggingface_hub import InferenceClient
                client = InferenceClient(token=hf_token.strip())
                res_img = client.text_to_image(prompt, model="black-forest-labs/FLUX.1-schnell")
                return res_img, "Hugging Face Serverless Cloud GPU (FLUX.1-schnell)"
            except Exception:
                pass

        # 2. High-speed Cloud Diffusion API (FLUX.1 / SDXL, real photorealistic AI synthesis, no token required)
        try:
            import urllib.request
            import urllib.parse
            import io

            w = min(1024, max(256, (width // 64) * 64))
            h = min(1024, max(256, (height // 64) * 64))
            clean_prompt = prompt.replace("\n", " ").strip()
            encoded_prompt = urllib.parse.quote(clean_prompt)
            url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={w}&height={h}&seed={seed}&nologo=true"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                res_img = Image.open(io.BytesIO(resp.read())).convert("RGB")
                return res_img, "Cloud Diffusion Engine (FLUX.1 / SDXL - Photorealistic)"
        except Exception:
            # Emergency fallback: procedural synthesis
            return self._generate_procedural_visual(prompt, seed), "High-Speed Procedural Simulator (Offline)"

    def _generate_procedural_visual(self, prompt: str, seed: int) -> Image.Image:
        """Generates rich artistic rendering based on prompt keywords and latent seed."""
        from PIL import ImageDraw, ImageFilter
        size = 512
        img = Image.new("RGB", (size, size), (18, 22, 32))
        draw = ImageDraw.Draw(img)

        p_hash = hash(prompt + str(seed))
        hue = abs(p_hash) % 360
        r = int(127 + 120 * np.sin(hue * np.pi / 180))
        g = int(127 + 120 * np.sin((hue + 120) * np.pi / 180))
        b = int(127 + 120 * np.sin((hue + 240) * np.pi / 180))

        # Base atmospheric layers
        for i in range(18):
            cx = (abs(p_hash * (i + 3)) % 400) + 56
            cy = (abs(p_hash * (i + 9)) % 400) + 56
            rad = (abs(p_hash * (i + 5)) % 100) + 30
            c = ((r + i * 8) % 256, (g + i * 12) % 256, (b + i * 16) % 256)
            draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=c)

        img = img.filter(ImageFilter.GaussianBlur(radius=12))
        draw = ImageDraw.Draw(img)

        # Golden ratio focal composition
        draw.ellipse([180, 180, 332, 332], outline=(255, 255, 255), width=3)
        draw.line([(64, 448), (448, 448)], fill=(200, 220, 255), width=2)
        
        # Caption watermark on bottom
        draw.rectangle([0, 470, 512, 512], fill=(10, 14, 24, 220))
        draw.text((20, 482), f"Prompt: {prompt[:50]}...", fill=(220, 220, 240))
        return img

    def get_vram_info(self) -> dict:
        """Track GPU VRAM utilization on local CUDA machines (RTX 3050)."""
        if torch.cuda.is_available():
            return {
                "device": torch.cuda.get_device_name(0),
                "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
                "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
                "total_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
            }
        return {"device": "CPU", "allocated_gb": 0.0, "total_vram_gb": "N/A (CPU Mode)"}

    def save_generation(self, image: Image.Image, metadata: dict, output_dir: str = "outputs") -> str:
        """Save generated image and metadata text file to disk."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gen_{timestamp}_s{metadata['seed']}.png"
        filepath = os.path.join(output_dir, filename)
        image.save(filepath)

        meta_path = filepath.replace(".png", "_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return filepath
