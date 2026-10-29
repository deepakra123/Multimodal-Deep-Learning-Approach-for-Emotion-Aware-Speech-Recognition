# Multimodal Deep Learning Approach for Emotion-Aware Speech Recognition

This repository contains a notebook-first speech recognition project built around OpenAI Whisper, PEFT, and LoRA. The workflow focuses on efficient fine-tuning and evaluation on the LibriSpeech ASR dataset, with the repo organized so the notebook, outputs, and reusable assets have clear homes.

## Demo

[Open the Streamlit demo](http://127.0.0.1:8501/)

## Overview

The project demonstrates an end-to-end speech recognition pipeline that transcribes audio to text while keeping training cost low through parameter-efficient fine-tuning. It is designed for experimentation in Google Colab or a local Jupyter environment.

## Repository Structure

```text
.
├── notebooks/
│   └── Multimodal_Emotion_Aware_Speech_Recognition.ipynb
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── outputs/
│   └── figures/
├── src/
└── README.md
```

## What Each Folder Is For

- `notebooks/`: main exploratory and training notebook.
- `data/raw/`: original dataset files.
- `data/processed/`: cleaned or resampled data prepared for training.
- `models/`: saved checkpoints and final model artifacts.
- `outputs/figures/`: charts, plots, confusion matrices, and evaluation visuals.
- `src/`: reusable Python helpers if the notebook is split into modules later.

## Project Highlights

- Whisper Base English model as the starting point.
- LibriSpeech ASR dataset for training and evaluation.
- LoRA and PEFT for efficient fine-tuning.
- Metrics including WER, CER, BLEU, accuracy, precision, recall, and F1-score.
- Visualization support for confusion matrices and ROC-style analysis.

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Datasets
- Accelerate
- PEFT
- BitsAndBytes
- Torchaudio
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn
- JiWER
- Evaluate

## Getting Started

1. Clone the repository.
2. Open `notebooks/Multimodal_Emotion_Aware_Speech_Recognition.ipynb` in Jupyter or Google Colab.
3. Install the required packages listed in the notebook.
4. Run the notebook cells in order.

## Run the interface

Install the requirements and launch the Streamlit interface:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The interface accepts WAV, MP3, FLAC, M4A, and OGG files. It transcribes speech with
Whisper and provides text-based emotion signals. The repository does not include a
trained audio-emotion checkpoint, so the emotion view is explicitly transcript-based.

For Colab, use the notebook's built-in badge or open it directly from the `notebooks/` folder path.

## Workflow

1. Install dependencies.
2. Authenticate with Hugging Face if needed.
3. Load a LibriSpeech sample or the full dataset.
4. Resample audio and prepare Whisper input features.
5. Load the Whisper Base model.
6. Attach LoRA adapters and fine-tune.
7. Evaluate with transcription and classification metrics.
8. Save checkpoints and visualizations.

## Evaluation

The notebook reports the following metrics:

- Word Error Rate (WER)
- Character Error Rate (CER)
- BLEU score
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix and ROC-style plots

## Notes

- The repository is intentionally structured for future modularization, even though the current implementation lives primarily in one notebook.
- Empty folders are included to reserve space for datasets, models, and outputs.
- If you later split the notebook into scripts, place shared code under `src/` and keep generated artifacts out of version control.
