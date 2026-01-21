# AHOCV Lab 3: Denoising Probabilistic Diffusion Models

This repository contains the implementation for Assignment #3 of the Advanced Hands-On Computer Vision course, focusing on text-to-image generation using diffusion models.

## Project Overview

This project demonstrates:
- Text-to-image generation using a CLIP-conditioned U-Net diffusion model
- Extraction of intermediate embeddings from the U-Net's bottleneck layer
- Evaluation using CLIP Score and Fréchet Inception Distance (FID)
- Dataset analysis with FiftyOne Brain (uniqueness & representativeness)
- Experiment tracking with Weights & Biases

## Links

| Resource | Link |
|----------|------|
| **WandB Dashboard** | [View Experiments](https://wandb.ai/finnfreiheit/denoising_probabilistic_diffusion_models) |
| **HuggingFace Dataset** | [diffusion-flowers-fiftyone](https://huggingface.co/datasets/FinnFreiheit/diffusion-flowers-fiftyone) |
| **GitHub Repository** | [FinnFreiheit/ahocv_lab3](https://github.com/FinnFreiheit/ahocv_lab3) |

## Project Structure

```
Lab_3/
├── Part1_ImageGeneration_EmbeddingExtraction.ipynb  # Image generation & embedding extraction
├── Part2_Evaluation_CLIP_FID.ipynb                  # CLIP Score & FID evaluation
├── Part3_FiftyOne_Analysis.ipynb                    # FiftyOne dataset analysis
├── Part4_WandB_Logging.ipynb                        # Weights & Biases logging
├── utils/                                           # Utility modules
│   ├── UNet_utils.py                                # U-Net model architecture
│   ├── ddpm_utils.py                                # DDPM sampling utilities
│   └── other_utils.py                               # Helper functions
├── generated_images/                                # Output images
└── Assignment #3 - Diffusion Models.md              # Assignment description
```

## Model Architecture

- **Model**: U-Net with CLIP conditioning
- **Timesteps (T)**: 400
- **Image Size**: 32x32
- **Down Channels**: (256, 256, 512)
- **CLIP Embedding Dimension**: 512

## Results

| Metric | Value |
|--------|-------|
| Mean CLIP Score | See WandB |
| FID Score | See WandB |

## Setup & Installation

```bash
pip install fiftyone wandb open-clip-torch einops ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

## Usage

Run the notebooks in order:
1. **Part 1**: Train model and generate images
2. **Part 2**: Calculate CLIP and FID scores
3. **Part 3**: Analyze with FiftyOne
4. **Part 4**: Log results to WandB

## Author

Finn Freiheit
