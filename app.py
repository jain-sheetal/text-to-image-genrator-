"""
🎨 Generative AI Studio: Unified Text-to-Image & Generative Model Lab
A state-of-the-art interactive platform integrating:
- Module 1: Pre-trained Diffusion Fine-Tuning & Domain Adaptation (LoRA)
- Module 2: Conditional GAN (CGAN) for Category & Shape Synthesis
- Module 3: Hugging Face Transformers Text Preprocessing & Embedding Engine
- Module 4: Public Dataset Explorer & Statistical Analyzer (Oxford-102 & COCO)
- Module 5: Attention-Enhanced GAN (Self-Attention & Cross-Attention Heatmaps)
- Module 6: End-to-End Production Text-to-Image Generation Pipeline
"""

import os
import sys
import torch
import gradio as gr
from PIL import Image

# Ensure local src package is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.pipeline import UnifiedTextToImagePipeline
from src.cgan import CGANPipeline, SHAPE_CLASSES
from src.attention_gan import AttentionGANPipeline
from src.text_processor import TextPreprocessor
from src.dataset_analyzer import DatasetAnalyzer
from src.diffusion_finetuning import DomainAdaptationEngine


# Initialize core components
pipeline = UnifiedTextToImagePipeline(device="auto")
text_processor = TextPreprocessor()
cgan_engine = CGANPipeline(device=str(pipeline.device))
attention_engine = AttentionGANPipeline(device=str(pipeline.device))
dataset_analyzer = DatasetAnalyzer()
domain_engine = DomainAdaptationEngine()

# In-memory gallery history
gallery_history = []


# ---------------------------------------------------------------------------
# Tab Handlers
# ---------------------------------------------------------------------------

def handle_text_to_image(
    prompt: str,
    negative_prompt: str,
    model_choice: str,
    scheduler: str,
    steps: int,
    guidance: float,
    width: int,
    height: int,
    seed: int,
    hf_token: str
):
    try:
        img, metadata = pipeline.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            model_type=model_choice,
            scheduler=scheduler,
            steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            seed=None if seed == -1 else int(seed),
            hf_token=hf_token.strip() if hf_token else None
        )
        gallery_history.append(img)
        if len(gallery_history) > 12:
            gallery_history.pop(0)

        info_md = f"""
### ✅ Generation Diagnostics
- **Engine Used:** `{metadata['engine']}`
- **Generation Time:** `{metadata['generation_time_sec']}s`
- **Scheduler:** `{metadata['scheduler']}` | **Inference Steps:** `{metadata['steps']}`
- **Guidance Scale (CFG):** `{metadata['guidance_scale']}`
- **Resolution:** `{metadata['resolution']}` | **Latent Seed:** `{metadata['seed']}`
- **Active Tokens:** `{metadata['active_tokens']}` | **Embedding Norm:** `{metadata['embedding_norm']}`
- **Device & Memory:** `{metadata['vram_info'].get('device', 'N/A')}` ({metadata['vram_info'].get('allocated_gb', 0)} GB allocated)
"""
        return img, info_md, gallery_history
    except Exception as e:
        err_md = f"❌ Generation error: {str(e)}"
        return None, err_md, gallery_history


def handle_load_local_gpu(model_choice: str):
    target_id = "stabilityai/sd-turbo" if "Turbo" in model_choice else "runwayml/stable-diffusion-v1-5"
    status = pipeline.load_local_diffusion(target_id)
    return status


def handle_cgan_generation(shape_name: str, seed: int, latent_var: float):
    try:
        img, meta = cgan_engine.generate(
            shape_name=shape_name,
            seed=None if seed == -1 else int(seed),
            latent_variation=latent_var
        )
        info_md = f"""
### 🎯 Conditional GAN Generation Report
- **Conditioned Label:** `{meta['requested_label'].upper()}` (Class Index: `{meta['class_index']}`)
- **Discriminator Realism Score:** `{meta['discriminator_realism_score'] * 100:.1f}%`
- **Latent Vector Dim (z):** `{meta['latent_dim']}` dimensions
- **Conditioning Mechanism:** `{meta['conditioning_mechanism']}`
- **Architecture Flow:** `[Noise z ⊕ Class Embedding c] → Dense Projection → Transposed Convolutions → Image`
"""
        return img, info_md
    except Exception as e:
        return None, f"❌ CGAN error: {str(e)}"


def handle_tokenization(prompt: str):
    try:
        token_info = text_processor.tokenize(prompt)
        chart_img = text_processor.generate_token_diagnostic_plot(token_info)
        _, stats = text_processor.encode_to_embedding(prompt)

        tokens_table = "| Index | Token String | Token ID | Attention Mask |\n| :--- | :--- | :--- | :--- |\n"
        for idx, (t, t_id) in enumerate(token_info["active_tokens"]):
            tokens_table += f"| {idx} | `{t}` | `{t_id}` | `1 (Active)` |\n"

        stats_md = f"""
### 📊 Continuous Vector Embedding Properties (Hugging Face Transformers)
- **Embedding Dimension:** `{stats['embedding_dimension']}` continuous latent features
- **L2 Vector Norm:** `{stats['l2_norm']}` (Euclidean normalized)
- **Mean Activation:** `{stats['mean']}` | **Standard Deviation:** `{stats['std']}`
- **Active Token Count:** `{stats['active_tokens_count']}` / 77 tokens
- **Architecture:** `{stats['model_architecture']}`
"""
        return chart_img, tokens_table, stats_md
    except Exception as e:
        return None, f"❌ Error: {str(e)}", ""


def handle_prompt_similarity(prompt_a: str, prompt_b: str):
    try:
        sim_pct, explanation = text_processor.compare_prompts_similarity(prompt_a, prompt_b)
        result_md = f"""
### 📐 Semantic Cosine Similarity Analysis
**Match Score:** `{sim_pct}%` Semantic Alignment in Latent Embedding Space.

*{explanation}*

> **Educational Note:** High cosine similarity indicates that the transformer encodes both prompts into closely aligned vectors, prompting the generative model to produce correlated compositions.
"""
        return result_md
    except Exception as e:
        return f"❌ Comparison error: {str(e)}"


def handle_dataset_inspection(dataset_name: str, sample_idx: int):
    try:
        stats = dataset_analyzer.get_dataset_statistics(dataset_name)
        canvas, captions_text, meta = dataset_analyzer.render_sample_visual(dataset_name, sample_idx)
        chart_img = dataset_analyzer.generate_distribution_chart(dataset_name)

        stats_md = f"""
### 📈 Benchmark Dataset Statistics: {dataset_name}
- **Total Images in Benchmark:** `{stats['benchmark_total_images']:,}`
- **Total Categories / Classes:** `{stats['benchmark_num_classes']}`
- **Captions Per Image:** `{stats['captions_per_image']}`
- **Mean Caption Length:** `{stats['caption_word_lengths']['mean']} words` (Min: `{stats['caption_word_lengths']['min']}`, Max: `{stats['caption_word_lengths']['max']}`)
- **Native Resolution Average:** `{stats['resolution_stats']['mean_width']} × {stats['resolution_stats']['mean_height']} pixels`
"""
        return canvas, captions_text, chart_img, stats_md
    except Exception as e:
        return None, f"❌ Error: {str(e)}", None, ""


def handle_attention_gan(prompt: str, attention_type: str, seed: int):
    try:
        base_img, heatmaps, meta = attention_engine.generate_with_attention(
            prompt=prompt,
            attention_type=attention_type,
            seed=None if seed == -1 else int(seed)
        )
        gallery_items = [img for img, _ in heatmaps]
        labels_list = "\n".join([f"• {lbl}" for _, lbl in heatmaps])

        meta_md = f"""
### 👁️ Attention Analysis: {attention_type}
- **Tokens Analyzed:** `{', '.join(meta['tokens_analyzed'])}`
- **Spatial Grid:** `{meta['feature_resolution']}` feature map
- **Mechanism:** `{meta['attention_mechanism']}`
- **Benefit:** `{meta['benefit']}`

**Token Heatmap Breakdown:**
{labels_list}
"""
        return base_img, gallery_items, meta_md
    except Exception as e:
        return None, [], f"❌ Error: {str(e)}"


def handle_domain_adaptation(
    prompt: str,
    domain_choice: str,
    lora_scale: float,
    lora_rank: int,
    lr: float,
    epochs: int,
    seed: int
):
    try:
        base_img, adapted_img, meta = domain_engine.render_comparison(
            prompt=prompt,
            domain_name=domain_choice,
            lora_scale=lora_scale,
            seed=None if seed == -1 else int(seed)
        )
        config_json = domain_engine.generate_training_config(
            domain_name=domain_choice,
            base_model="runwayml/stable-diffusion-v1-5",
            lora_rank=lora_rank,
            learning_rate=lr,
            epochs=epochs,
            batch_size=2
        )
        report_md = f"""
### 🧪 Domain Adaptation Report
- **Target Domain:** `{meta['domain']}`
- **Trigger Token:** `{meta['trigger_token']}`
- **LoRA Adapter Scale (α/r):** `{meta['lora_scaling_alpha']}`
- **Base Model Weights:** `{meta['base_model']}`
- **Status:** `{meta['status']}`
"""
        return base_img, adapted_img, report_md, config_json
    except Exception as e:
        return None, None, f"❌ Error: {str(e)}", "{}"


# ---------------------------------------------------------------------------
# Gradio UI Construction
# ---------------------------------------------------------------------------

device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
vram_desc = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB VRAM" if torch.cuda.is_available() else "Standard CPU Mode"

custom_css = """
.gradio-container { max-width: 1280px !important; margin: auto; }
.hero-box { background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%); padding: 24px; border-radius: 12px; color: white; margin-bottom: 20px; }
.badge { background: #10b981; color: white; padding: 4px 10px; border-radius: 9999px; font-weight: bold; font-size: 0.85em; }
"""

with gr.Blocks(title="🎨 Generative AI Studio") as demo:
    gr.HTML(f"""
    <div class="hero-box">
        <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800;">🎨 Educational Generative AI Studio</h1>
        <p style="margin: 8px 0 0 0; font-size: 1.1rem; opacity: 0.9;">
            Unified Text-to-Image Pipeline, Conditional GANs, Attention Mechanisms & Hugging Face Transformers
        </p>
        <div style="margin-top: 14px; display: flex; gap: 12px; align-items: center;">
            <span class="badge">Hardware: {device_name} ({vram_desc})</span>
            <span class="badge" style="background: #6366f1;">PyTorch {torch.__version__}</span>
            <span class="badge" style="background: #0ea5e9;">Cloud & Local Dual-Engine</span>
        </div>
    </div>
    """)

    with gr.Tabs():
        # -------------------------------------------------------------------
        # TAB 1: Master Text-to-Image Generation
        # -------------------------------------------------------------------
        with gr.Tab("🚀 Unified Text-to-Image Studio"):
            with gr.Row():
                with gr.Column(scale=5):
                    prompt_input = gr.Textbox(
                        label="🎯 Text Description (Prompt)",
                        placeholder="a majestic snow-capped mountain at sunrise, photorealistic, cinematic lighting, 8k",
                        value="a serene mountain lake reflecting snow peaks at golden hour, photorealistic, 8k resolution, cinematic lighting",
                        lines=3
                    )
                    neg_prompt_input = gr.Textbox(
                        label="🚫 Negative Prompt (What to Avoid)",
                        placeholder="blurry, low quality, deformed, artifacts",
                        value="blurry, distorted, low resolution, bad anatomy, flat lighting",
                        lines=2
                    )

                    with gr.Row():
                        model_dropdown = gr.Dropdown(
                            choices=[
                                "Cloud Diffusion Engine (FLUX.1 / SDXL - Photorealistic)",
                                "Local Stable Diffusion 1.5 (RTX 3050 CUDA GPU)",
                                "Local SD-Turbo (1-4 Steps Fast GPU)",
                                "Conditional GAN (Shape Synthesis)",
                                "Attention-Enhanced GAN (Cross-Attention)"
                            ],
                            value="Cloud Diffusion Engine (FLUX.1 / SDXL - Photorealistic)",
                            label="Generative Architecture"
                        )
                        scheduler_dropdown = gr.Dropdown(
                            choices=["euler_a", "euler", "ddim", "dpm_solver", "lms"],
                            value="euler_a",
                            label="Denoising Scheduler"
                        )

                    with gr.Row():
                        load_gpu_btn = gr.Button("⚡ Load Model into RTX 3050 GPU VRAM", variant="secondary")
                        load_gpu_status = gr.Textbox(
                            label="GPU Model Status",
                            value="Ready (Cloud Diffusion active by default for instant photorealistic generation)",
                            interactive=False,
                            lines=1
                        )

                    with gr.Accordion("🔧 Advanced Hyperparameters", open=False):
                        with gr.Row():
                            steps_slider = gr.Slider(1, 50, value=20, step=1, label="Inference Steps")
                            guidance_slider = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="Guidance Scale (CFG)")
                        with gr.Row():
                            width_slider = gr.Slider(256, 1024, value=512, step=64, label="Width")
                            height_slider = gr.Slider(256, 1024, value=512, step=64, label="Height")
                        with gr.Row():
                            seed_input = gr.Number(-1, label="Seed (-1 for random)", precision=0)
                            token_input = gr.Textbox(
                                label="Optional Hugging Face Token (For high-speed serverless cloud quota)",
                                placeholder="hf_...",
                                type="password"
                            )

                    generate_btn = gr.Button("🎨 Synthesize Image", variant="primary", size="lg")

                with gr.Column(scale=5):
                    output_image = gr.Image(label="🖼️ Synthesized Output", type="pil")
                    output_diagnostics = gr.Markdown("### 📊 Status: Ready for generation")

            gr.Markdown("### 🖼️ Session Generation History")
            history_gallery = gr.Gallery(label="Recent Generations", columns=4, height=220)

            load_gpu_btn.click(
                fn=handle_load_local_gpu,
                inputs=[model_dropdown],
                outputs=[load_gpu_status]
            )

            generate_btn.click(
                fn=handle_text_to_image,
                inputs=[
                    prompt_input, neg_prompt_input, model_dropdown, scheduler_dropdown,
                    steps_slider, guidance_slider, width_slider, height_slider,
                    seed_input, token_input
                ],
                outputs=[output_image, output_diagnostics, history_gallery]
            )

        # -------------------------------------------------------------------
        # TAB 2: Conditional GAN (CGAN)
        # -------------------------------------------------------------------
        with gr.Tab("🎯 Conditional GAN (CGAN)"):
            gr.Markdown(r"""
            ### 📐 Conditional Generative Adversarial Networks
            In standard GANs, the Generator maps random noise $z \sim \mathcal{N}(0, I)$ to an arbitrary image.
            **Conditional GANs (CGANs)** condition both the Generator $G(z, c)$ and the Discriminator $D(x, c)$ on categorical or textual labels $c$.
            """)
            with gr.Row():
                with gr.Column(scale=5):
                    cgan_shape_select = gr.Dropdown(
                        choices=SHAPE_CLASSES,
                        value="circle",
                        label="Conditioning Label (Shape / Category)"
                    )
                    cgan_seed = gr.Number(-1, label="Latent Noise Seed (-1 for random)", precision=0)
                    cgan_variation = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Latent Noise Variance (σ)")
                    cgan_btn = gr.Button("🎲 Generate Conditional Form", variant="primary")

                with gr.Column(scale=5):
                    cgan_output = gr.Image(label="🖼️ CGAN Generated Visual", type="pil")
                    cgan_diag = gr.Markdown("### Status: Select a shape to generate")

            cgan_btn.click(
                fn=handle_cgan_generation,
                inputs=[cgan_shape_select, cgan_seed, cgan_variation],
                outputs=[cgan_output, cgan_diag]
            )

        # -------------------------------------------------------------------
        # TAB 3: Text Preprocessing & Embeddings (Transformers)
        # -------------------------------------------------------------------
        with gr.Tab("🔤 Text Preprocessing & Embeddings"):
            gr.Markdown("""
            ### 🧠 Text Tokenization & Continuous Latent Encoding
            Text-to-image models rely on **Transformer Text Encoders** (such as CLIP or BERT) to project discrete natural language words into continuous semantic vector representations.
            """)
            with gr.Row():
                with gr.Column(scale=6):
                    tok_prompt = gr.Textbox(
                        label="Enter Text Description",
                        value="a vibrant red flower blooming in soft morning sunshine with dew drops",
                        lines=2
                    )
                    tokenize_btn = gr.Button("⚡ Tokenize & Extract Embeddings", variant="primary")

                with gr.Column(scale=4):
                    tok_stats_md = gr.Markdown("Click above to analyze text representations.")

            tok_chart = gr.Image(label="📊 Token Vocabulary IDs & Attention Distribution", type="pil")
            tok_table = gr.Markdown()

            tokenize_btn.click(
                fn=handle_tokenization,
                inputs=[tok_prompt],
                outputs=[tok_chart, tok_table, tok_stats_md]
            )

            gr.Markdown("---")
            gr.Markdown("### 📐 Latent Space Cosine Similarity Comparator")
            with gr.Row():
                with gr.Column():
                    sim_prompt_1 = gr.Textbox(label="Prompt A", value="a crimson garden rose in sunlight")
                with gr.Column():
                    sim_prompt_2 = gr.Textbox(label="Prompt B", value="a vibrant red blooming flower in natural daylight")
            sim_btn = gr.Button("🔍 Compute Semantic Similarity")
            sim_output_md = gr.Markdown()

            sim_btn.click(
                fn=handle_prompt_similarity,
                inputs=[sim_prompt_1, sim_prompt_2],
                outputs=[sim_output_md]
            )

        # -------------------------------------------------------------------
        # TAB 4: Public Dataset Explorer & Statistics
        # -------------------------------------------------------------------
        with gr.Tab("📊 Dataset Explorer & Statistics"):
            gr.Markdown("""
            ### 📂 Public Text-to-Image Benchmark Dataset Inspection
            Analyze description lengths, vocabulary diversity, and image resolutions across benchmark datasets like **Oxford-102 Flowers** and **MS-COCO**.
            """)
            with gr.Row():
                with gr.Column(scale=4):
                    ds_choice = gr.Dropdown(
                        choices=dataset_analyzer.get_dataset_names(),
                        value="Oxford-102 Flowers",
                        label="Dataset Selection"
                    )
                    sample_slider = gr.Slider(0, 4, value=0, step=1, label="Sample Index Inspection")
                    inspect_btn = gr.Button("🔍 Inspect Dataset & Sample", variant="primary")

                with gr.Column(scale=6):
                    ds_stats_display = gr.Markdown("Click inspect to view dataset metrics.")

            with gr.Row():
                with gr.Column(scale=5):
                    ds_sample_img = gr.Image(label="🖼️ Paired Image Sample", type="pil")
                    ds_captions_box = gr.Textbox(label="📝 Associated Text Captions", lines=5)
                with gr.Column(scale=5):
                    ds_chart_img = gr.Image(label="📊 Dataset Statistical Distributions", type="pil")

            inspect_btn.click(
                fn=handle_dataset_inspection,
                inputs=[ds_choice, sample_slider],
                outputs=[ds_sample_img, ds_captions_box, ds_chart_img, ds_stats_display]
            )

        # -------------------------------------------------------------------
        # TAB 5: Attention-Enhanced GAN
        # -------------------------------------------------------------------
        with gr.Tab("👁️ Attention-Enhanced GAN"):
            gr.Markdown("""
            ### 🔍 Self-Attention & Cross-Attention in Generative Networks
            - **Self-Attention (SAGAN):** Captures long-range spatial dependencies across the entire canvas.
            - **Cross-Attention (AttnGAN):** Aligns visual feature regions with specific token representations in the text prompt.
            """)
            with gr.Row():
                with gr.Column(scale=5):
                    attn_prompt = gr.Textbox(
                        label="Prompt for Attention Synthesis",
                        value="vibrant golden flower petals with green stem",
                        lines=2
                    )
                    attn_type = gr.Radio(
                        choices=["Cross-Attention (Text-Image)", "Self-Attention (Spatial-Spatial)"],
                        value="Cross-Attention (Text-Image)",
                        label="Attention Strategy"
                    )
                    attn_seed = gr.Number(-1, label="Seed (-1 for random)", precision=0)
                    attn_btn = gr.Button("⚡ Generate with Attention Heatmaps", variant="primary")

                with gr.Column(scale=5):
                    attn_base_out = gr.Image(label="🖼️ Generated Visual", type="pil")
                    attn_meta_md = gr.Markdown()

            gr.Markdown("### 🗺️ Token-by-Token Spatial Attention Heatmap Overlays")
            attn_gallery = gr.Gallery(label="Attention Maps per Word", columns=4, height=260)

            attn_btn.click(
                fn=handle_attention_gan,
                inputs=[attn_prompt, attn_type, attn_seed],
                outputs=[attn_base_out, attn_gallery, attn_meta_md]
            )

        # -------------------------------------------------------------------
        # TAB 6: Diffusion Fine-Tuning & Domain Adaptation (LoRA)
        # -------------------------------------------------------------------
        with gr.Tab("🧪 Diffusion Fine-Tuning & LoRA"):
            gr.Markdown("""
            ### 🧬 Domain Adaptation & Parameter-Efficient Fine-Tuning (LoRA)
            Adapting a pre-trained foundation model to specific domains (e.g. Artwork, Medical Imaging) by training low-rank update matrices $W = W_0 + \\frac{\\alpha}{r}(B \\times A)$.
            """)
            with gr.Row():
                with gr.Column(scale=5):
                    domain_select = gr.Dropdown(
                        choices=domain_engine.get_domain_list(),
                        value="Medical Imagery (Radiology & MRI)",
                        label="Domain-Specific Visual Task"
                    )
                    domain_prompt = gr.Textbox(
                        label="Domain Prompt with Trigger Token",
                        value="frontal chest radiograph showing clear lung fields, medical_xray",
                        lines=2
                    )
                    lora_scale_slider = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="LoRA Adapter Scale (Blend)")

                    with gr.Accordion("⚙️ Training Hyperparameters", open=False):
                        lora_rank_input = gr.Slider(4, 64, value=16, step=4, label="LoRA Rank (r)")
                        lr_input = gr.Number(1e-4, label="Learning Rate")
                        epochs_input = gr.Slider(1, 100, value=20, step=1, label="Epochs")
                        domain_seed = gr.Number(-1, label="Seed", precision=0)

                    compare_btn = gr.Button("🔬 Run Domain Adaptation Comparison", variant="primary")

                with gr.Column(scale=5):
                    domain_report_md = gr.Markdown("Select domain and run comparison.")
                    with gr.Row():
                        domain_base_img = gr.Image(label="Pre-trained Base Model", type="pil")
                        domain_adapted_img = gr.Image(label="Domain-Adapted Model (LoRA)", type="pil")

            with gr.Accordion("📜 Export PyTorch Diffusers Training Specification (JSON)", open=False):
                config_json_display = gr.Code(language="json")

            compare_btn.click(
                fn=handle_domain_adaptation,
                inputs=[domain_prompt, domain_select, lora_scale_slider, lora_rank_input, lr_input, epochs_input, domain_seed],
                outputs=[domain_base_img, domain_adapted_img, domain_report_md, config_json_display]
            )

        # -------------------------------------------------------------------
        # TAB 7: Educational Theory & Guide
        # -------------------------------------------------------------------
        with gr.Tab("📚 Educational Guide & Architecture"):
            gr.Markdown(r"""
            ## 🧠 Generative AI & Text-to-Image Mathematical Foundations

            ### 1. Conditional GANs (CGAN)
            In a Conditional GAN, both the generator and discriminator receive side information $y$:
            $$\\min_G \\max_D V(D, G) = \\mathbb{E}_{x \\sim p_{data}}[\\log D(x|y)] + \\mathbb{E}_{z \\sim p_z}[\\log(1 - D(G(z|y)|y))]$$

            ### 2. Self-Attention & Cross-Attention
            - **Self-Attention:** Discovers non-local spatial relationships across image features $X$:
              $$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d}}\\right)V$$
            - **Cross-Attention:** Directs image features to attend to language tokens from the text encoder, anchoring colors, shapes, and positions to specific words.

            ### 3. Diffusion Denoising Process
            - **Forward Diffusion (Noising):** $q(x_t | x_{t-1}) = \\mathcal{N}(x_t; \\sqrt{1 - \\beta_t}x_{t-1}, \\beta_t I)$
            - **Reverse Diffusion (Denoising):** U-Net parameterized network $\\epsilon_\\theta(x_t, t, c)$ predicts the added Gaussian noise conditioned on text embedding $c$.
            - **Classifier-Free Guidance (CFG):**
              $$\\tilde{\\epsilon}_\\theta(x_t, c) = \\epsilon_\\theta(x_t, \\emptyset) + s \\cdot (\\epsilon_\\theta(x_t, c) - \\epsilon_\\theta(x_t, \\emptyset))$$
              where $s$ is the Guidance Scale slider. Higher $s$ strictly forces prompt alignment.

            ### 4. Low-Rank Adaptation (LoRA)
            Freezes foundation weights $W_0 \\in \\mathbb{R}^{d \\times k}$ and trains low-rank decomposition matrices:
            $$W = W_0 + \\frac{\\alpha}{r} (B \\times A), \\quad B \\in \\mathbb{R}^{d \\times r}, A \\in \\mathbb{R}^{r \\times k}$$
            Reduces trainable parameters by >95% while retaining full base generation quality.
            """)

# Launch application
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
        css=custom_css
    )
