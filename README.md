# DiT-CT: Diffusion Transformer for CT Artifacts Removal

The CT artifact removal project based on diffusion Transformer (DiT) includes two core processes: data generation and model training, supporting the removal of various clinical artifacts.

## Code Structure
DiTArtifactsRemoval/
│
├── data/                    # Data-related utilities (no real data included)
│   ├── diffct               # A high-performance, CUDA-accelerated library for circular orbits CT reconstruction, enabling advanced optimization and deep learning integration.
│   ├── pancreas_artifacts_simulation.py
│   ├── brain_artifacts_simulation.py
│   └── README.md            # Instructions for dataset preparation
│
├── trainers/                # Model training and inference modules
│   ├── diffusers/           # Neural network architectures (UNet, DiT, etc.)
│   ├── vae.py               # Variational Autoencoder implementation
│   ├── train.py             # Distributed training scripts for the main model Diffusion Transformer
│   ├── ddpm_ddp.py          # Distributed training script for conditional DDPM models
│   ├── pix2pix.py           # Pix2Pix model training and evaluation
│   └── inference_utils.py   # Utilities for inference, visualization, and evaluation
│
├── examples
│
├── README.md                # Project overview, installation instructions, usage examples
├── requirements.txt         # Python dependencies
├── .gitignore               # Files/folders to ignore (e.g., dataset, checkpoints, cache)
