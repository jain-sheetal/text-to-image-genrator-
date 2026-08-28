# 🎨 Educational Stable Diffusion Text-to-Image Generator
LIVE DEMO : https://8ee60f5ffe1c491c9d.gradio.live/

An interactive text-to-image web application built with **Hugging Face Diffusers**, **PyTorch**, and **Gradio**. This project provides a full-featured UI to generate images using diffusion models while demonstrating key Generative AI concepts, scheduler comparisons, and VRAM memory optimization techniques.

---

## ✨ Features

- **Multi-Model Support:**
  - Stable Diffusion 1.5 (`runwayml/stable-diffusion-v1-5`)
  - Stable Diffusion 2.1 (`stabilityai/stable-diffusion-2-1`)
  - Realistic Vision (`SG161222/RealVisXL_V4.0`)
- **Schedulers Supported:**
  - Euler Ancestral (`euler_a`)
  - Euler (`euler`)
  - DDIM (`ddim`)
  - DPM-Solver (`dpm_solver`)
  - LMS (`lms`)
- **Memory Optimizations:**
  - Automatic `float16` precision for CUDA GPUs
  - Attention slicing (`pipe.enable_attention_slicing()`)[cite: 1]
  - VAE slicing (`pipe.enable_vae_slicing()`)[cite: 1]
  - Optional `xFormers` memory-efficient attention[cite: 1]
  - CPU offloading fallback (`pipe.enable_model_cpu_offload()` / `pipe.enable_sequential_cpu_offload()`)[cite: 1]
- **Real-Time Diagnostics:** Track VRAM allocation, reserved memory, total GPU memory, and generation time per inference[cite: 1].
- **Automatic Export:** Saves generated PNG images alongside detailed parameter metadata text files[cite: 1].
- **Interactive Gradio UI:** Includes parameter sliders (Width, Height, Steps, Guidance Scale, Seed), an example prompt picker, and a recent generation gallery[cite: 1].

---

## 🛠️ Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (or CPU mode)[cite: 1]

### Dependencies
- `torch`[cite: 1]
- `diffusers`[cite: 1]
- `transformers`[cite: 1]
- `accelerate`[cite: 1]
- `safetensors`[cite: 1]
- `gradio`[cite: 1]
- `Pillow`[cite: 1]
- `numpy`[cite: 1]
- `xformers` *(optional, for GPU acceleration)*[cite: 1]

---

