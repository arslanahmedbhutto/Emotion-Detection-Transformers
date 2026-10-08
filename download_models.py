"""
Download the fine-tuned models from the Hugging Face Hub into models/.

Usage:
    python download_models.py              # all four models
    python download_models.py roberta      # just one (distilbert, xlnet, roberta, bigbird)
"""

import os
import sys
from huggingface_hub import snapshot_download

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

MODELS = {
    'distilbert': 'arslanahmed3/emotion-distilbert',
    'xlnet': 'arslanahmed3/emotion-xlnet',
    'roberta': 'arslanahmed3/emotion-roberta',
    'bigbird': 'arslanahmed3/emotion-bigbird',
}


def main():
    names = sys.argv[1:] or list(MODELS)
    for name in names:
        if name not in MODELS:
            print(f"Unknown model '{name}'. Choose from: {', '.join(MODELS)}")
            continue
        target = os.path.join(BASE_DIR, 'models', name)
        print(f"Downloading {MODELS[name]} -> {target}")
        snapshot_download(repo_id=MODELS[name], local_dir=target, ignore_patterns=['README.md', '.gitattributes'])
    print("Done.")


if __name__ == '__main__':
    main()
