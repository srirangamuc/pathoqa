# PathoQA: Pathology Visual Question Answering

**PathoQA** is a specialized multimodal Visual Question Answering (VQA) system designed for the medical domain, specifically pathology. It leverages state-of-the-art vision and language models to understand and answer specific questions about pathology images.

## Key Features

* **Multimodal Fusion Architecture**: Integrates a **T5 (Text-to-Text Transfer Transformer)** language model with a **CLIP (Contrastive Language-Image Pre-Training)** vision encoder.
* **Domain-Specific Encoders**:
  * **Vision**: Defaults to `jamessyx/pathgenclip-vit-large-patch14-hf`, a CLIP model fine-tuned on pathology images.
  * **Text**: Supports custom domain-adapted T5 models (e.g., fine-tuned on medical corpora).
* **Robust Evaluation**: Includes a comprehensive evaluation suite tracking:
  * **Binary Accuracy** (for Yes/No questions).
  * **Open-Ended Metrics**: Exact Match, BLEU scores, and **SBERT** (Sentence-BERT) semantic similarity.
* **Training Safeguards**: Implements automatic checkpointing (`last_checkpoint.pth`) to resume training and saves the best model based on overall performance metrics.

## Project Structure

```text
.
├── src/                    # Core source code
│   ├── config.py           # Configuration and argument parsing
│   ├── dataset.py          # Data loading and processing
│   ├── evaluator.py        # Metric computation (BLEU, SBERT, etc.)
│   ├── model.py            # PathVQA Fusion Model architecture
│   └── utils.py            # Helper functions
├── train.py                # Main training script
├── evaluate.py             # Model evaluation script
├── experiment_t5_vqa.ipynb # Jupyter notebooks for experimentation
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/srirangamuc/pathoqa.git
   cd pathoqa
   ```
2. **Install dependencies**:
   Ensure you have Python 3.8+ installed.

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

To start training the model, use the `train.py` script. You can configure hyperparameters via command-line arguments.

```bash
python train.py --epochs 10 --batch_size 8 --output_dir ./results
```

**Common Arguments:**

| Argument         | Default                        | Description                                     |
| :--------------- | :----------------------------- | :---------------------------------------------- |
| `--t5_path`    | `./final_pathology_model_v2` | Path to the T5 model or HuggingFace checkpoint. |
| `--clip_path`  | `jamessyx/pathgenclip...`    | Path to the CLIP vision encoder.                |
| `--image_size` | `336`                        | Input image size (pixels).                      |
| `--lr`         | `1e-4`                       | Learning rate.                                  |
| `--device`     | `cuda` (if available)        | Training device (cuda/cpu).                     |

### Evaluation

Use the `evaluate.py` script to test a trained model (this assumes the script exists and follows similar patterns, or you can use the notebook for interactive eval).

```bash
python evaluate.py --output_dir ./results
```

## Model Architecture

The `PathVQA_Fusion_Model` consists of three main components:

1. **Vision Encoder**: Extracts high-level visual features from pathology slides.
2. **Text Encoder (T5 Encoder)**: Processes the input question.
3. **Fusion Module**: A cross-attention mechanism that fuses visual and textual embeddings before passing them to the T5 Decoder
   to generate the natural language answer.
