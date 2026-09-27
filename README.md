# RemoteCLIP on UC Merced

This project evaluates **RemoteCLIP** for satellite-image scene classification on the **UC Merced Land Use** dataset. It compares two approaches:

- **Linear probing:** use RemoteCLIP as a frozen image encoder, then train a linear classifier on the extracted image features.
- **Zero-shot classification:** compare image embeddings with text embeddings for 21 class prompts without training a classifier.

Both notebooks use the Hugging Face dataset [`rifahnanjiba2/UC_Merced`](https://huggingface.co/datasets/rifahnanjiba2/UC_Merced) and the RemoteCLIP ViT-B/32 checkpoint from [`chendelong/RemoteCLIP`](https://huggingface.co/chendelong/RemoteCLIP).

## Notebooks

| Notebook | Description |
| --- | --- |
| [`01_remote_clip_linear_probe.ipynb`](01_remote_clip_linear_probe.ipynb) | Downloads RemoteCLIP, extracts normalized image features, trains a linear classifier for 20 epochs, plots training metrics, and reports test accuracy. |
| [`02_remoteclip_zero_shot.ipynb`](02_remoteclip_zero_shot.ipynb) | Builds 21 satellite-image text prompts, predicts classes from image-text similarity, and reports zero-shot test accuracy. |

## Dataset split

The notebooks start from the dataset's `train` split and create the same deterministic stratified split:

- 70% training
- 15% validation
- 15% test

The split uses `seed=42` and the `label` column for stratification. UC Merced contains 21 land-use classes, including agricultural land, beaches, forests, harbors, residential areas, runways, storage tanks, and tennis courts.

## Requirements

- Python 3.9 or newer
- Jupyter or Google Colab
- PyTorch
- A CUDA-capable GPU is recommended for practical runtime

The notebooks install their additional dependencies with:

```bash
pip install -q open_clip_torch huggingface_hub datasets
```

`matplotlib` is also used by the linear-probe notebook for plots. In a fresh environment, install it with:

```bash
pip install matplotlib
```

## Running the notebooks

### Google Colab

Open either notebook in Colab using its **Open in Colab** badge, then select a GPU runtime when running the linear-probe notebook.

### Local Jupyter

From the project directory:

```bash
jupyter notebook
```

Run the cells from top to bottom. The first run downloads the UC Merced dataset and the RemoteCLIP checkpoint from Hugging Face.

The linear-probe notebook currently sets its device to CUDA directly, so it requires a CUDA-enabled PyTorch installation and an available GPU. The zero-shot notebook selects CUDA when available and otherwise falls back to CPU, although CPU inference can be slow because it evaluates every test image individually.

## Method overview

### Linear probe

1. Load and stratify the UC Merced data.
2. Load the pretrained RemoteCLIP ViT-B/32 checkpoint.
3. Extract and L2-normalize image embeddings with the RemoteCLIP preprocessing pipeline.
4. Train a `torch.nn.Linear` classifier while keeping RemoteCLIP frozen.
5. Track training and validation loss/accuracy and evaluate on the held-out test split.

### Zero-shot evaluation

1. Load and stratify the UC Merced data.
2. Encode one text prompt for each of the 21 classes.
3. Encode each test image and L2-normalize both image and text embeddings.
4. Compute image-text similarities and apply softmax to obtain class probabilities.
5. Compare the highest-scoring prompt with the ground-truth label.

## Reproducibility notes

- Dataset splits use `seed=42`.
- The model checkpoint is downloaded from Hugging Face and cached locally by `huggingface_hub`.
- Results can vary with library versions, PyTorch/CUDA versions, and hardware.
- The notebooks are currently distributed as experiments and do not write model checkpoints or metric files to disk.
