"""
Hugging Face Hub settings for slow / blocked networks (timeouts, optional mirror).

Set before loading sentence-transformers, e.g. in PowerShell:
  $env:HF_ENDPOINT = "https://hf-mirror.com"
  $env:HF_HUB_DOWNLOAD_TIMEOUT = "600"
"""

from __future__ import annotations

import os


def configure_hf_hub() -> None:
    """Raise download/connect timeouts; optional mirror for regions where huggingface.co is slow."""
    # Longer waits for large model files (seconds)
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
    # Optional mirror (user sets HF_ENDPOINT); do not override if already set
    # Example for China: https://hf-mirror.com


def load_sentence_transformer(model_name: str, *, max_attempts: int = 3):
    """
    Load SentenceTransformer with retries (handles transient WinError 10060 / timeouts).
    """
    import time

    from sentence_transformers import SentenceTransformer

    configure_hf_hub()
    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return SentenceTransformer(model_name)
        except Exception as e:
            last_err = e
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)
    assert last_err is not None
    raise last_err
