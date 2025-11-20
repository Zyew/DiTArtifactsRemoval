# DiT-CT: Diffusion Transformer for CT Artifacts Removal
This project implements **CT artifact removal** using a **Diffusion Transformer (DiT)** framework. It supports removing various clinical artifacts from CT images and provides utilities for data simulation, model training, and inference.

## Features

- **Artifact Simulation**: Generate artifact images for different anatomical regions (brain, pancreas).
- **Latent Conditional Diffusion**: Perform conditional diffusion in the latent space for more efficient and robust artifact removal.
- **Inference Utilities**: Easy-to-use scripts for visualization, evaluation, and batch inference.


## Code Structure
```markdown
DiTArtifactsRemoval/
│
├── data/                                    # Data-related utilities
│   ├── diffct/                              # CUDA-accelerated CT reconstruction library
│   ├── pancreas_artifacts_simulation.py     # Simulate various CT artifacts for pancreas
│   ├── brain_artifacts_simulation.py        # Simulate motion artifacts for brain CT only
│   └── README.md                            # Dataset preparation instructions
│
├── trainers/                                # Model training and inference
│   ├── diffusers/                           # Neural network architectures (UNet, DiT, etc.)
│   ├── vae.py                               # Variational Autoencoder implementation
│   ├── train.py                             # Distributed training scripts for the main model Diffusion Transformer
│   ├── ddpm_ddp.py                          # Distributed training script for conditional DDPM models
│   ├── pix2pix.py                           # Pix2Pix model training and evaluation
│   └── inference_utils.py                   # Utilities for inference, visualization, and evaluation
│
├── examples
│
├── README.md                                # README
├── requirements.txt                         # Python dependencies
├── .gitignore                               # Files/folders to ignore (e.g., dataset, checkpoints, cache)
```

## Installation
This project requires two separate conda environments to avoid CUDA and dependency conflicts:

### DiffCT Environment for artifact simulation and CT data generation
```markdown
conda env create -f environment_diffct.yml
conda activate diffct
```
### DiT Environment for training and inference of artifact removal models
```markdown
conda env create -f environment_dit.yml
conda activate DiT
```

## Acknowledgements
This project leverages 
[DiTTransformer2DModel](https://huggingface.co/docs/diffusers/v0.35.1/en/api/models/dit_transformer2d#diffusers.DiTTransformer2DModel) for artifact removal, 
[DiffCT](https://github.com/sypsyp97/diffct) for differentiable CT artifact simulation and reconstruction.
