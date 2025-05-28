# Music Generation Transformer Models

This project implements a modular training pipeline for music generation using various transformer-based models. It supports multiple model architectures and training strategies on a unified dataset extracted from NPZ files. The project uses Hugging Face’s Transformers Trainer along with Weights & Biases (WandB) for logging and monitoring.

## Features

- **Multiple Model Types:**
  - **Flattening Shared:**  
    A custom transformer model (shared codebook) that concatenates token embeddings with external positional embeddings (via a learnable projection) and conditions on instrument tokens.
  - **Flattening Separate:**  
    Similar to the shared variant, but uses a separate codebook for each audio channel. The base vocabulary is defined (e.g. 1024) and the total vocabulary size is computed as base_vocab_size × number_of_channels.
  - **GPT‑2 Fine-Tuning:**  
    A custom GPT‑2 model that concatenates token embeddings with external positional embeddings (followed by a projection), adds instrument conditioning, and supports both training from scratch and finetuning from pretrained weights.
  - **Delay Model:**  
    A transformer model that reorders 2D target track data (channels and time steps) into a 1D sequence via a delay pattern. It computes custom delay groups and uses these to generate an autoregressive mask. This allows the model to learn delay patterns within the audio signal.

- **Unified Dataset:**  
  The dataset is loaded from an aggregated NPZ file and converted into a unified format. A single `MusicDataset` class handles different processing modes:
  - `"shared"` for Flattening Shared and GPT‑2 processing,
  - `"separate"` for Flattening Separate processing,
  - `"delay"` for the Delay model (which includes computing delay groups).

- **Dynamic Evaluation and Logging:**  
  The training script dynamically computes evaluation steps (and saving steps) based on the percentage of data used. WandB callbacks log training metrics, generated audio samples, and spectrograms for monitoring.

- **Modular Architecture:**  
  The project is organized into separate modules:
  - **Main Training Script:** `train.py` handles argument parsing, model initialization, training, evaluation, and saving.
  - **Model Definitions:**  
    - `models_flattening.py` defines the Flattening Shared model.
    - `models_flattening_separate.py` defines the Flattening Separate model.
    - `models_gpt2.py` defines the GPT‑2 model variant.
    - `models_delay.py` defines the Delay model.
  - **Utility Modules:**  
    - `utils/models.py` provides helper functions for model initialization.
    - `utils/npz_loader.py` loads the NPZ dataset.
    - `utils/wandb_callbacks.py` contains WandB logging callbacks.
    - `utils/spectogram.py` is a helper for creating spectrogram plots.

## Directory Structure

```
project/
├── train.py                          # Main training script
├── models_flattening.py              # Flattening Shared model definition
├── models_flattening_separate.py     # Flattening Separate model definition
├── models_gpt2.py                    # GPT‑2 model definition
├── models_delay.py                   # Delay model definition
└── utils/
    ├── models.py                   # Helper functions for model initialization
    ├── npz_loader.py               # NPZ dataset loader
    ├── wandb_callbacks.py          # WandB logging callbacks
    └── spectogram.py               # Spectrogram plotting utility
```

## Requirements

- Python 3.7+
- PyTorch (CUDA-enabled version recommended)
- Hugging Face Transformers
- WandB
- scikit-learn
- numpy
- torchaudio
- matplotlib

## Usage

From the project root directory, run the unified training script (`train.py`) with command-line options to choose the model type, data subset, and instrument classes. Examples:

### Flattening Shared Model
```bash
python train.py --data_percent 1 --instrument_classes full_instrumental --model_type flattening_shared
```

### Flattening Separate Model
```bash
python train.py --data_percent 100 --instrument_classes full_instrumental drums --model_type flattening_separate
```

### GPT‑2 Fine-Tuning (from scratch)
```bash
python train.py --data_percent 100 --instrument_classes full_instrumental drums --model_type gpt2
```

### GPT‑2 Fine-Tuning (with pretrained weights)
```bash
python train.py --data_percent 100 --instrument_classes full_instrumental drums --model_type gpt2 --gpt2_pretrained
```

### Delay Model
```bash
python train.py --data_percent 100 --instrument_classes full_instrumental --model_type delay
```

## Logging and Saving

- **WandB Logging:**  
  Training metrics, generated audio samples, and spectrograms are logged to WandB. The run name is generated to reflect the data percentage, instrument classes, and model type (with an extra flag for GPT‑2 pretrained).

- **Model Saving:**  
  Models are saved at the end of training with a descriptive filename that includes the WandB run name.

## Notes

- The NPZ loader supports a `data_percent` parameter to control how many samples are loaded.
- The unified dataset class processes data differently based on the selected mode (`shared`, `separate`, or `delay`).
- For the delay model, custom processing is applied to flatten the track data using a delay pattern, and delay group indices are provided for mask generation.
- Training and evaluation steps are dynamically computed based on dataset size.
- Feel free to adjust configurations (e.g., batch size, learning rate, number of epochs) in `train.py` as needed.

## License

This project is provided for educational and research purposes @JKU Computational Perception Lab
