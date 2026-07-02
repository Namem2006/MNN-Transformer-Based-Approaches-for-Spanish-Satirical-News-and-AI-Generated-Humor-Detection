# Transformer-Based Approaches for Spanish Satirical News and AI-Generated Humor Detection

This repository contains the cleaned project materials for our **HAHA @ IberLEF 2026** submission on two binary text classification tasks:

- **Subtask 1 — Satirical News Detection**: classify a Spanish news headline as **`satirical`** or **`real`**.
- **Subtask 2 — AI-Generated Humor Detection**: classify a joke as **`machine`**-generated or **`human`**-written.

The repository is organized for GitHub submission and Kaggle execution, with a simple structure centered around two folders:

- `dataset/` → official shared-task TSV files used by the notebooks.
- `notebooks/` → three main pipelines used in our experiments.

> **Important cleanup note**  
> The old combined notebook was removed. The interactive demo is now integrated directly into **Pipeline 1 (BETO)** as the final **Gradio** cell.

---

## Repository structure

```text
.
├── dataset/
│   ├── task1_trial.tsv
│   ├── task1_dev_gold.tsv
│   ├── task1_test.tsv
│   ├── task2_trial.tsv
│   ├── task2_dev_gold.tsv
│   ├── task2_test.tsv
│   └── task3_test.tsv
│
├── notebooks/
│   ├── pipeline1_beto_humor_detection.ipynb
│   ├── pipeline2_xlmr_humor_detection.ipynb
│   └── pipeline3_multibranch_stacking.ipynb
│
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

---

## Dataset

For both subtasks, training is done on **`trial + dev_gold`**, and predictions are generated for the official **test** files.

| Subtask | Train files | Train size | Positive class | Test file | Test size |
|---|---|---:|---|---|---:|
| Task 1 — Satirical News Detection | `task1_trial.tsv` + `task1_dev_gold.tsv` | 564 | `satirical` | `task1_test.tsv` | 600 |
| Task 2 — AI-Generated Humor Detection | `task2_trial.tsv` + `task2_dev_gold.tsv` | 362 | `machine` | `task2_test.tsv` | 550 |

### Label mapping

```text
Task 1: real = 0, satirical = 1
Task 2: human = 0, machine = 1
```

---

## Notebook overview

| Notebook | Role | What it does |
|---|---|---|
| `pipeline1_beto_humor_detection.ipynb` | **Pipeline 1** | Uses **BETO** for **both Task 1 and Task 2**. This notebook also includes the final **Gradio demo UI**. |
| `pipeline2_xlmr_humor_detection.ipynb` | **Pipeline 2** | Uses **XLM-RoBERTa-base** for **both Task 1 and Task 2**, following the same workflow as Pipeline 1. |
| `pipeline3_multibranch_stacking.ipynb` | **Pipeline 3** | Uses a **multi-branch system**: **Task 1 = XLM-R + TF-IDF**, **Task 2 = XLM-R + perplexity/stylometry** with stacking and threshold tuning. |

### Correct pipeline logic

This repository follows the logic below:

```text
Pipeline 1 = BETO workflow for Task 1 + Task 2
Pipeline 2 = XLM-RoBERTa-base workflow for Task 1 + Task 2
Pipeline 3 = extended multi-branch system
```

So **Pipeline 1 and Pipeline 2 are the same workflow**, but use different encoders.

---

## Method summary

## 1) Shared Transformer workflow (Pipeline 1 and Pipeline 2)

Pipeline 1 and Pipeline 2 share the same overall process:

```text
TSV data
→ text cleaning
→ input construction
→ tokenization
→ Transformer fine-tuning
→ 5-fold stratified cross-validation
→ best checkpoint selection by positive-class F1
→ soft-voting ensemble across folds
→ test prediction TSV + submission ZIP
```

### Input construction

- **Task 1**: `headline + context`
- **Task 2**: `headline + [SEP] + [LONG]/[SHORT] + joke`

For **Task 2**, the notebook inserts:

- `[LONG]` if the joke has **at least 18 words**
- `[SHORT]` otherwise

This follows our observation that machine-generated jokes tend to be longer than human-written jokes.

### Encoders

- **Pipeline 1** uses **BETO** (`dccuchile/bert-base-spanish-wwm-cased`)
- **Pipeline 2** uses **XLM-RoBERTa-base** (`xlm-roberta-base`)

## 2) Multi-branch workflow (Pipeline 3)

Pipeline 3 extends beyond a single Transformer branch.

### Task 1

- XLM-R Transformer branch
- TF-IDF + Logistic Regression branch
- Stacking with a meta-learner
- Threshold tuning on OOF predictions

### Task 2

- XLM-R Transformer branch
- Perplexity / stylometry feature branch
- Stacking with a meta-learner
- Threshold tuning on OOF predictions

Stylometric features include signals such as word count, character count, punctuation ratio, uppercase ratio, digit ratio, type-token ratio, and optional Spanish GPT-2 perplexity features.

---

## Report-aligned results

The table below summarizes the main OOF F1 numbers referenced in our project materials.

| Method | Task 1 F1 | Task 2 F1 |
|---|---:|---:|
| Pipeline 1 — BETO | **0.8040** | **0.8625** |
| Pipeline 2 — XLM-RoBERTa-base | 0.7500 | 0.8460 |
| Pipeline 3 — XLM-R + TF-IDF / perplexity-stylometry | 0.7946 | 0.8454 |
| Official baseline | 0.6250 | 0.6600 |

> In the written report, the main paired setup highlights **BETO for Task 1** and **XLM-R for Task 2**.  
> In this GitHub repository, however, **Pipeline 1** and **Pipeline 2** are each kept complete for **both tasks** so they are easier to inspect and reproduce.

---

## How to run on Kaggle

1. Create a new **Kaggle Notebook**.
2. Upload this repository, or upload the `dataset/` folder as a Kaggle Dataset.
3. Open one notebook from `notebooks/`.
4. In the notebook settings:
   - Enable **GPU (T4)**.
   - Turn **Internet ON** to allow model download from Hugging Face.
5. Run the notebook from top to bottom.

### Dataset auto-discovery

The notebooks automatically search for the dataset in:

```text
./dataset
../dataset
/kaggle/input/**
```

If automatic discovery fails, set `DATA_DIR` manually in the configuration cell.

---

## Outputs

Each notebook writes outputs to:

- `/kaggle/working` on Kaggle
- `./outputs` when running locally

Typical output structure:

```text
/kaggle/working/
├── pipeline1_beto_task1/
│   ├── fold_1/
│   ├── fold_2/
│   └── ...
├── pipeline1_beto_task2/
│   ├── fold_1/
│   ├── fold_2/
│   └── ...
├── task1.tsv
├── task2.tsv
└── pipeline1_beto_submission.zip
```

Generated checkpoints and submission outputs are ignored by `.gitignore`.

---

## Pipeline 1 demo (Gradio UI)

The final cell of `pipeline1_beto_humor_detection.ipynb` launches a **Gradio interface** after training.

### What the demo supports

- choose **Task 1** or **Task 2**,
- load **easy / hard preset examples**,
- test **custom Spanish inputs**,
- display the **predicted label**,
- display the **class probabilities**,
- show the **assembled model input** used for inference.

### Important usage note

The Gradio demo uses the **BETO fold checkpoints trained earlier in the same notebook**.

So the correct usage is:

```text
Run training first → fold checkpoints are saved → launch Gradio demo → test examples/custom inputs
```

If you launch the demo before training, inference will fail because no checkpoints exist yet.

---

## Dependencies

Install dependencies with:

```bash
pip install -r requirements.txt
```

Main libraries used in the project:

- PyTorch
- Transformers
- scikit-learn
- pandas / numpy
- tqdm
- sentencepiece
- Gradio (used by the final demo cell in Pipeline 1)

---

## Reproducibility notes

- Random seed: **42**
- Cross-validation: **5-fold StratifiedKFold**
- Evaluation metric: **binary F1 of the official positive class**
- Mixed precision is enabled when CUDA is available
- XLM-R notebooks use memory-saving options such as gradient checkpointing when needed
- Each fold checkpoint is saved to disk and GPU memory is cleared after fold inference/training steps

---

## References

- Cañete et al. (2020). *Spanish Pre-Trained BERT Model and Evaluation Data (BETO)*.
- Conneau et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale (XLM-R)*.
- Loshchilov & Hutter (2019). *Decoupled Weight Decay Regularization (AdamW)*.
- HAHA @ IberLEF 2026 shared-task materials.
