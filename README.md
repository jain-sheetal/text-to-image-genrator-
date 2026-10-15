
---

# 🎨 Educational Generative AI & Text-to-Image Studio

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-orange)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An interactive, end-to-end Generative AI laboratory and web application built with **PyTorch**, **Hugging Face Transformers & Diffusers**, and **Gradio**. This project bridges theory and practice, encompassing **Conditional GANs**, **Self/Cross-Attention Mechanisms**, **Transformer Tokenization & Embeddings**, **Benchmark Dataset Analytics (Oxford-102 Flowers & MS-COCO)**, and **Parameter-Efficient Diffusion Fine-Tuning (LoRA)**.

---

## 🌟 Core Architecture & Curriculum Modules

```
                           ┌────────────────────────────────────────┐
                           │       User Natural Language Prompt     │
                           └───────────────────┬────────────────────┘
                                               │
                                               ▼
                         ┌───────────────────────────────────────────┐
                         │   Module 3: HF Transformers Preprocessing │
                         │   - BPE / WordPiece Tokenization & IDs    │
                         │   - 512-D Latent Semantic Vector Embed    │
                         │   - Cosine Similarity & Geometry Analysis │
                         └─────┬───────────────────┬───────────────┬─┘
                               │                   │               │
            ┌──────────────────┘                   │               └──────────────────┐
            ▼                                      ▼                                  ▼
┌─────────────────────────┐   ┌─────────────────────────────┐   ┌─────────────────────────┐
│ Module 2: Conditional   │   │ Module 5: Attention-Enhanced│   │ Module 1 & 6: Diffusion │
│ GAN (CGAN)              │   │ GAN (SAGAN & AttnGAN)       │   │ Fine-Tuning & Pipeline  │
│ - Category labels (c)   │   │ - Spatial Self-Attention    │   │ - LoRA Domain Adapters  │
│ - Noise vector (z)      │   │ - Text-Image Cross-Attention│   │ - Cloud / CUDA FP16 Eng │
│ - G(z, c) & D(x, c)     │   │ - Token Heatmap Overlays    │   │ - Attention/VAE Slicing │
└───────────┬─────────────┘   └──────────────┬──────────────┘   └────────────┬────────────┘
            │                                │                               │
            └────────────────────────────────┼───────────────────────────────┘
                                             │
                                             ▼
                         ┌───────────────────────────────────────────┐
                         │   Module 4: Dataset Inspector & Analytics │
                         │   - Oxford-102 Flowers & MS-COCO Stats    │
                         │   - Description Lengths & Resolutions     │
                         │   - Interactive Sample Visual Explorer    │
                         └───────────────────────────────────────────┘
```

---

## 🚀 Key Modules Breakdown

### 1. 🧪 Diffusion Fine-Tuning & Domain Adaptation (LoRA)
- Refines foundation diffusion models (Stable Diffusion v1.5 / SDXL) for domain-specific visual synthesis:
  - **Artwork & Masterpiece Styles** (Watercolor, Impressionist, Oil).
  - **Medical Imagery** (Chest Radiographs, MRI Scans, CT).
  - **Cyberpunk Concept Art**.
- Employs **Low-Rank Adaptation (LoRA)**:
  $$W = W_0 + \frac{\alpha}{r}(B \times A), \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$$
- Includes interactive comparison between generic base diffusion and domain-adapted weights with an adjustable LoRA scale slider ($\alpha \in [0.0, 1.0]$) and automated JSON training config generation.

### 2. 🎯 Conditional GAN (CGAN) for Shape Synthesis
- Implements conditional generative adversarial networks conditioned on categorical labels:
  - Shapes: `circle`, `square`, `triangle`, `star`, `hexagon`, `heart`, `diamond`.
- Both the **Generator** $G(z, c)$ and **Discriminator** $D(x, c)$ receive the label representation.
- Computes Discriminator Realism Scores and explains feature conditioning flow.

### 3. 🔤 Hugging Face Transformers Text Preprocessing & Embedding Engine
- Tokenizes raw descriptions into token IDs, vocabulary tokens, and attention masks using Hugging Face Transformers / CLIP patterns.
- Projects tokens into normalized 512-dimensional continuous latent space.
- Features an interactive **Latent Space Cosine Similarity Comparator** showing semantic alignment between prompt pairs.

### 4. 📊 Public Dataset Explorer & Statistical Analyzer
- Structural analysis of benchmark text-to-image datasets (**Oxford-102 Flowers** and **MS-COCO**):
  - Number of classes and total benchmark images.
  - Caption word count distributions (mean, median, min, max, std).
  - Image resolution distributions.
- Displays paired image-caption samples with multi-caption views and metadata.

### 5. 👁️ Attention-Enhanced GAN (Self-Attention & Cross-Attention)
- **Spatial Self-Attention (SAGAN):** Discovers global non-local spatial relationships across feature maps.
- **Text-to-Image Cross-Attention (AttnGAN):** Aligns visual feature channels with individual token vectors.
- Renders **Token-by-Token Attention Heatmaps** overlaid on synthesized images using plasma colormaps to visualize which words guide which image regions.

### 6. 🚀 Unified End-to-End Production Pipeline
- Seamlessly coordinates text tokenization, model routing, inference execution, and diagnostic telemetry:
  - **Dual-Engine Flexibility:** Automatically leverages high-speed cloud serverless inference for instant 2-3s generation on free cloud tiers (no VRAM required), or PyTorch CUDA FP16 with attention slicing on local GPUs (e.g., RTX 3050).
  - Schedulers supported: Euler Ancestral (`euler_a`), Euler (`euler`), DDIM (`ddim`), DPM-Solver (`dpm_solver`), and LMS (`lms`).
  - Session history gallery with PNG and JSON metadata archival.

---

## 🛠️ Installation & Local Setup

### Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with CUDA (optional, CPU mode and Cloud engine supported)

### 1. Clone the Repository
```bash
git clone https://github.com/jain-sheetal/text-to-image-genrator-.git
cd text-to-image-genrator-
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Web Application
```bash
python app.py
```
Open your browser and navigate to:
```
http://localhost:7860
```

---

---

## 📂 Repository File Structure

```
text-to-image-generator/
├── app.py                      # Master Gradio 7-tab Web Studio
├── requirements.txt            # Production dependencies
├── README.md                   # Documentation & Hugging Face Spaces config
├── .gitignore                  # Git ignore rules
├── src/
│   ├── __init__.py             # Package declaration
│   ├── cgan.py                 # Conditional GAN (Generator, Discriminator, Shapes)
│   ├── attention_gan.py        # Self-Attention & Cross-Attention GAN
│   ├── text_processor.py       # HF Transformers tokenization & embedding engine
│   ├── dataset_analyzer.py     # Oxford-102 Flowers & MS-COCO dataset explorer
│   ├── diffusion_finetuning.py # LoRA domain adaptation & training config
│   └── pipeline.py             # End-to-end unified production pipeline
└── text_to_image_genreator.ipynb # Interactive Jupyter Notebook
```


---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
