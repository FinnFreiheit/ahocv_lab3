# AHOCV Lab 3: Denoising Probabilistic Diffusion Models

This repository contains the implementation for Assignment #3 of the Advanced Hands-On Computer Vision course, focusing on text-to-image generation using diffusion models.


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

