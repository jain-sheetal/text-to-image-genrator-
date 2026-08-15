import gc
import os
import time
from datetime import datetime
from typing import Optional, Tuple

import streamlit as st
import torch
from PIL import Image
from torch import autocast

from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
)

st.set_page_config(page_title="Stable Diffusion Generator", page_icon="🎨", layout="wide")

# ----------------------------------------------------
# Core Generator Class
# ----------------------------------------------------
class StableDiffusionGenerator:
    def __init__(self, model_id: str = "stabilityai/sd-turbo", device: str = "auto"):
        self.device = self._setup_device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.pipe = self._load_pipeline(model_id)
        self.current_scheduler = "euler_a"
        self.schedulers = {
            "euler_a": ("Euler Ancestral", "Fast, creative randomness"),
            "euler": ("Euler", "Deterministic, consistent results"),
            "ddim": ("DDIM", "Classic, high quality, slower"),
            "dpm_solver": ("DPM Solver", "High quality, efficient"),
            "lms": ("LMS", "Linear multistep, stable"),
        }

    def _setup_device(self, device: str) -> torch.device:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)

    def _load_pipeline(self, model_id: str) -> StableDiffusionPipeline:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()

        if self.device.type == "cuda":
            pipe = pipe.to("cuda")
        else:
            pipe = pipe.to("cpu")

        return pipe

    def set_scheduler(self, scheduler_name: str):
        if scheduler_name == self.current_scheduler:
            return
        scheduler_map = {
            "euler_a": EulerAncestralDiscreteScheduler,
            "euler": EulerDiscreteScheduler,
            "ddim": DDIMScheduler,
            "dpm_solver": DPMSolverMultistepScheduler,
            "lms": LMSDiscreteScheduler,
        }
        if scheduler_name in scheduler_map:
            scheduler_class = scheduler_map[scheduler_name]
            self.pipe.scheduler = scheduler_class.from_config(self.pipe.scheduler.config)
            self.current_scheduler = scheduler_name

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        scheduler: str = "euler_a",
    ) -> Tuple[Image.Image, dict]:
        self.set_scheduler(scheduler)
        if seed is None or seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()

        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        width = (width // 8) * 8
        height = (height // 8) * 8

        start_time = time.time()
        try:
            with torch.inference_mode():
                if self.device.type == "cuda" and self.dtype == torch.float16:
                    with autocast(self.device.type):
                        result = self.pipe(
                            prompt=prompt,
                            negative_prompt=negative_prompt if negative_prompt else None,
                            width=width,
                            height=height,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            generator=generator,
                        )
                else:
                    result = self.pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt if negative_prompt else None,
                        width=width,
                        height=height,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    )
            generation_time = time.time() - start_time
            metadata = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "scheduler": scheduler,
                "seed": seed,
                "generation_time": round(generation_time, 2),
                "device": str(self.device),
                "dtype": str(self.dtype),
            }
            return result.images[0], metadata
        finally:
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()


# ----------------------------------------------------
# UI & State Management
# ----------------------------------------------------
st.title("🎨 Educational Stable Diffusion Generator")

if "gallery" not in st.session_state:
    st.session_state.gallery = []

tab1, tab2, tab3 = st.tabs(["🚀 Setup & Generation", "📚 Learning Resources", "🖼️ Gallery"])

with tab1:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("🔧 Model Configuration")
        model_choice = st.selectbox(
            "Model Selection",
            [
                "SD-Turbo (Ultra Fast CPU / 1-4 Steps)",
                "Stable Diffusion 1.5 (Recommended)",
                "Stable Diffusion 2.1",
                "Realistic Vision (RealVisXL)",
            ],
            key="model_choice_box",
        )
        device_choice = st.selectbox("Device Selection", ["Auto (Recommended)", "GPU (CUDA)", "CPU (Slower)"])

        # Dynamic parameter recommendations and defaults based on selection
        if "Turbo" in model_choice:
            st.info("⚡ **SD-Turbo Selected**: Keep **Steps = 1 to 4** and **Guidance Scale = 0.0 to 1.0** for fast, crisp results on CPU.")
            default_steps = 2
            default_guidance = 0.0
        elif "2.1" in model_choice:
            st.info("🎯 **SD 2.1 Selected**: Needs **Steps = 20 to 30** and **Guidance Scale = 7.5 to 9.0** to avoid noisy artifacts.")
            default_steps = 20
            default_guidance = 7.5
        elif "RealVis" in model_choice:
            st.info("📸 **RealVisXL Selected**: Best with **Steps = 25 to 35** and **Guidance Scale = 6.0 to 7.5**.")
            default_steps = 25
            default_guidance = 7.0
        else:
            st.info("🎨 **SD 1.5 Selected**: Best with **Steps = 20 to 25** and **Guidance Scale = 7.0 to 8.0**.")
            default_steps = 20
            default_guidance = 7.5

        model_map = {
            "SD-Turbo (Ultra Fast CPU / 1-4 Steps)": "stabilityai/sd-turbo",
            "Stable Diffusion 1.5 (Recommended)": "runwayml/stable-diffusion-v1-5",
            "Stable Diffusion 2.1": "stabilityai/stable-diffusion-2-1",
            "Realistic Vision (RealVisXL)": "SG161222/RealVisXL_V4.0",
        }
        device_map = {"Auto (Recommended)": "auto", "GPU (CUDA)": "cuda", "CPU (Slower)": "cpu"}

        if st.button("🚀 Load / Reload Model"):
            with st.spinner("Loading Pipeline weights into memory..."):
                try:
                    st.session_state.generator = StableDiffusionGenerator(
                        model_id=model_map[model_choice],
                        device=device_map[device_choice],
                    )
                    st.success("✅ Model loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Failed to load model: {e}")

        st.subheader("✨ Image Generation")
        prompt = st.text_area(
            "🎯 Prompt",
            placeholder="a serene mountain landscape at sunrise, oil on canvas",
            value="futuristic cyberpunk supercar driving through a neon lit city in heavy rain, reflections on wet asphalt, octane render, cinematic composition",
        )
        negative_prompt = st.text_area(
            "🚫 Negative Prompt",
            placeholder="blurry, low quality, bad anatomy",
            value="ugly, low resolution, daytime, flat lighting, blurry, distorted",
        )

        with st.expander("🔧 Advanced Settings & Parameters Guide", expanded=True):
            width = st.slider("Width", min_value=256, max_value=1024, value=512, step=64)
            height = st.slider("Height", min_value=256, max_value=1024, value=512, step=64)

            steps = st.slider(
                "Inference Steps",
                min_value=1,
                max_value=50,
                value=default_steps,
                step=1,
                help="1-4 for Turbo models, 20-30 for standard models.",
            )
            guidance = st.slider(
                "Guidance Scale (CFG)",
                min_value=0.0,
                max_value=20.0,
                value=default_guidance,
                step=0.5,
                help="0.0 for Turbo models, 7.0-8.5 for standard models.",
            )

            scheduler = st.selectbox("Scheduler", ["euler_a", "euler", "ddim", "dpm_solver", "lms"])
            seed = st.number_input("Seed (-1 for random)", value=-1, step=1)

            st.markdown("""
            **Optimal Presets Reference:**
            | Model Type | Recommended Steps | Recommended CFG | Resolution |
            | :--- | :--- | :--- | :--- |
            | **SD-Turbo** | `1 – 4` | `0.0 – 1.0` | 512 × 512 |
            | **SD 1.5** | `20 – 25` | `7.0 – 8.0` | 512 × 512 |
            | **SD 2.1** | `20 – 30` | `7.5 – 9.0` | 512 × 512 / 768 × 768 |
            | **RealVisXL** | `25 – 35` | `6.0 – 7.5` | 512 × 512 / 768 × 768 |
            """)

        generate_btn = st.button("🎨 Generate Image", type="primary")

    with col_right:
        st.subheader("🖼️ Output")
        if generate_btn:
            if "generator" not in st.session_state or st.session_state.generator is None:
                st.error("Please click '🚀 Load / Reload Model' first.")
            elif not prompt.strip():
                st.warning("Please enter a prompt.")
            else:
                with st.spinner("Denoising latent space..."):
                    try:
                        img, meta = st.session_state.generator.generate_image(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            width=width,
                            height=height,
                            num_inference_steps=steps,
                            guidance_scale=guidance,
                            scheduler=scheduler,
                            seed=None if seed == -1 else int(seed),
                        )
                        st.image(img, caption=prompt)
                        st.session_state.gallery.append({"image": img, "meta": meta})
                        st.json(meta)
                    except Exception as e:
                        st.error(f"Generation error: {e}")

with tab2:
    st.markdown("""
    ## 🧠 Understanding Diffusion Models
    * **CLIP / Text Encoder**: Converts textual tokens into dense semantic vector representations.
    * **Latent Space**: Diffusion operates in a lower-dimensional latent space compressed by a VAE rather than high-resolution pixel space.
    * **U-Net**: Iteratively predicts and subtracts noise conditioned on text embeddings.
    * **Variational Autoencoder (VAE)**: Decodes the cleaned latent representation back into RGB pixel values.
    * **Adversarial Distillation (SD-Turbo)**: Collapses multi-step iterative denoising down into 1 to 4 steps via single-step student network training.
    """)

with tab3:
    st.subheader("🖼️ Recent Generations")
    if st.session_state.gallery:
        cols = st.columns(3)
        for idx, item in enumerate(reversed(st.session_state.gallery[-6:])):
            with cols[idx % 3]:
                st.image(item["image"], caption=item["meta"]["prompt"][:40] + "...")
    else:
        st.info("No generated images yet.")