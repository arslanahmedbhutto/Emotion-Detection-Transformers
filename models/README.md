# Trained models

The fine-tuned weights are **not stored in git**. Each file is 250–500 MB, which is over GitHub's 100 MB limit. The small config and tokenizer files are tracked. Put the weight files back into this layout:

```
models/
├── distilbert/   config.json, vocab.txt, tokenizer_config.json, special_tokens_map.json, tf_model.h5        (~256 MB)
├── xlnet/        config.json, spiece.model, tokenizer_config.json, special_tokens_map.json, tf_model.h5     (~449 MB)
├── roberta/      config.json, vocab.json, merges.txt, tokenizer_config.json, special_tokens_map.json, tf_model.h5 (~478 MB)
└── bigbird/      config.json, spiece.model, tokenizer_config.json, special_tokens_map.json, model.safetensors (~490 MB)
```

| Folder | Framework | Weights file | Notebook |
|--------|-----------|--------------|----------|
| `distilbert/` | TensorFlow | `tf_model.h5` | `notebooks/01_DistilBERT.ipynb` |
| `xlnet/`      | TensorFlow | `tf_model.h5` | `notebooks/02_XLNet.ipynb` |
| `roberta/`    | TensorFlow | `tf_model.h5` | `notebooks/03_RoBERTa.ipynb` |
| `bigbird/`    | PyTorch    | `model.safetensors` | `notebooks/04_BigBird.ipynb` |

## Ways to get the weights

- **Download (recommended):** the models are on the Hugging Face Hub. From the repo root, run:

  ```bash
  python download_models.py            # all four models (~1.7 GB)
  python download_models.py roberta    # or just one
  ```

  | Model | Hugging Face |
  |-------|--------------|
  | DistilBERT | [arslanahmed3/emotion-distilbert](https://huggingface.co/arslanahmed3/emotion-distilbert) |
  | XLNet      | [arslanahmed3/emotion-xlnet](https://huggingface.co/arslanahmed3/emotion-xlnet) |
  | RoBERTa    | [arslanahmed3/emotion-roberta](https://huggingface.co/arslanahmed3/emotion-roberta) |
  | BigBird    | [arslanahmed3/emotion-bigbird](https://huggingface.co/arslanahmed3/emotion-bigbird) |

  You can also load them directly in code with `from_pretrained("arslanahmed3/emotion-roberta")`. See each model card for an example.

- **Retrain:** run the notebooks in `notebooks/` and save each model into the matching folder with `model.save_pretrained(...)` and `tokenizer.save_pretrained(...)`.

`early_experiments/` (git-ignored) holds the earlier 5-epoch DistilBERT and XLNet checkpoints from `notebooks/early_experiments/`.
