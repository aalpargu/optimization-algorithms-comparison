# Optimization algorithms on Turkish QA embeddings

A course experiment that connects Turkish question-answer data generation with a small optimization problem. I used a local Turkish language model to generate examples, filtered similar questions with sentence embeddings, then compared gradient descent, stochastic gradient descent and Adam implemented in NumPy.

[Türkçe](README.tr.md) · [Notebook](optimization.ipynb)

## Experiment

The notebook has three main stages:

1. **Generate examples.** Turkish-Gemma-9B is loaded in GGUF form through `llama-cpp-python`. The generator samples eight topics and produces a question with a compatible answer and an incompatible answer.
2. **Filter and embed.** `ytu-ce-cosmos/turkish-e5-large` supplies sentence embeddings. Cosine similarity is used to reject near-duplicate questions, with a configured threshold of 0.82.
3. **Compare optimizers.** A classifier built on the embeddings is trained with NumPy versions of GD, SGD and Adam. The comparison repeats training from five initializations for 100 epochs and records the optimization curves.

The generator is configured for **100 question/positive-answer/negative-answer triplets**, split into 50 training and 50 test triplets. A triplet can produce two labelled question-answer examples; this should not be described as 200 distinct questions.

## Read or run

Open `optimization.ipynb` in Jupyter or Colab. The notebook contains its setup cells and recorded output. It uses Python, NumPy, scikit-learn, Matplotlib, Sentence Transformers, Hugging Face Hub and llama-cpp-python. Downloading the 9B model requires substantial storage and memory; the generation cells are separate from the optimization cells.

The model referenced by the notebook is [Turkish-Gemma-9b-T1-GGUF](https://huggingface.co/ytu-ce-cosmos/Turkish-Gemma-9b-T1-GGUF). The embedding model is [turkish-e5-large](https://huggingface.co/ytu-ce-cosmos/turkish-e5-large).

## Reproduction limit

This repository currently contains the notebook, without its generated dataset JSON files or a pinned environment. The generation section writes `dataset_v2.json`, while the later embedding/training sections load `dataset_v3.json` and `dataset_v3_with_embeddings.json`. Running every cell unchanged from an empty directory will therefore not reproduce the recorded experiment.

To run a new experiment, make the generation and downstream filenames consistent, generate a fresh dataset and its embeddings, and record that run separately. The existing cell outputs are historical results; they were not regenerated for the documentation update.

## Scope

This is a controlled optimizer comparison on a small synthetic dataset. It does not fine-tune the language model, establish a general ranking of optimizers, or measure performance on an independent real-world Turkish QA benchmark.
