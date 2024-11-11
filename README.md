# MedMNIST Diffusion Foundation

This project implements a diffusion model for various datasets from the [MedMNIST collection](https://medmnist.com/). The code is structured to support training, sampling, and evaluating diffusion-based synthetic data generation for multiple medical imaging datasets. The project includes optional Weights & Biases logging for monitoring training metrics and supports easy integration with different MedMNIST datasets.

## Project Structure

- **`dataset.py`**: Contains dataset initialization functions to load MedMNIST datasets and create data loaders with specified batch sizes.
- **`model.py`**: Defines the diffusion model structure, including encoders and the main diffusion backbone.
- **`train.py`**: Contains the training loop for the diffusion model, including optional Weights & Biases logging.
- **`sample.py`**: Defines the `generate_samples` function to create synthetic data samples using reverse diffusion and displays results in a 3x7 grid.
- **`cli.py`**: Provides a command-line interface to run training and sampling for specified datasets.
- **`main.py`**: Runs training and sampling sequentially for all datasets, using default training parameters for testing.
- **`README.md`**: This file, providing an overview and instructions for the project.

## Installation

1. Create and activate a new conda environment
   ```bash
   conda create --name medmnist_diffusion_foundation_env
   conda activate medmnist_diffusion_foundation_env
   ```
2. Use conda to python, pip, ffmpeg (for animations), and cuda toolkit 12.4.1
   ```bash
   conda install python=3.12 pip ffmpeg
   conda install nvidia/label/cuda-12.4.1::cuda-toolkit
   ```
3. Install torch with support for cuda tool kit 12.4
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```
   
4. Install additional python packages
   ```bash
   pip3 install numpy matplotlib torch-ema pydicom nibabel diffusers wandb
   ```

5. Install the Generative Medical Imaging (gmi) python package
   ```bash
   pip3 install git+https://github.com/Generative-Medical-Imaging-Lab/gmi.git
   ```

6. Install the MedMNIST dataset, using the most recent version we have tested
   ```bash
   pip install git+https://github.com/MedMNIST/MedMNIST.git@8cce68f261f993bd0450edc0200498a0691362c2
   ```

## Usage

### Training and Sampling

To train and sample from all MedMNIST datasets using default testing parameters, run:

```bash
python main.py
```

This will train each dataset model for 10 epochs, with 100 iterations per epoch, and generate a set of diffusion samples for each.

### CLI for Specific Dataset

Use `cli.py` to train or sample from a specific dataset. For example, to train on `BloodMNIST`:

```bash
python cli.py --task train --dataset BloodMNIST --num_epochs 10
```

Or, to generate samples:

```bash
python cli.py --task sample --dataset BloodMNIST --num_steps 100 --output figures/bloodmnist_sample.png
```

## Optional Weights & Biases Logging

To enable Weights & Biases (WandB) logging during training, provide the project name as `wandb_project` when calling `train_medmnist_diffusion_foundation_model` or use the `--wandb_project` argument in `cli.py`. Ensure that WandB is properly set up in your environment.

## Notes

- **Batch Sizes**: By default, training uses `batch_size=32`, while sampling uses `batch_size=9` for optimal GPU utilization.
- **Testing Setup**: This setup trains models for a limited number of epochs (10) to verify functionality.

## Acknowledgments

This project leverages the MedMNIST dataset collection and incorporates diffusion model architectures compatible with PyTorch.

## License

Distributed under the MIT License. See `LICENSE` for more information.