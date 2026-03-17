import argparse
import os
import shutil
from huggingface_hub import snapshot_download


def check_free_space(path: str, min_gb: int) -> None:
    os.makedirs(path, exist_ok=True)
    free_bytes = shutil.disk_usage(path).free
    free_gb = free_bytes / (1024 ** 3)
    if free_gb < min_gb:
        raise RuntimeError(
            f"Not enough disk space at '{path}'. Free: {free_gb:.2f} GB, required: >= {min_gb} GB."
        )


def default_model_dir() -> str:
    if os.path.isdir("/kaggle/temp"):
        return "/kaggle/temp/model"
    return "./model"


def default_cache_dir() -> str:
    if os.path.isdir("/kaggle/temp"):
        return "/kaggle/temp/hf_cache"
    return "./.hf_cache"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen2-7B-Instruct")
    parser.add_argument("--save-dir", default=os.environ.get("MODEL_DIR", default_model_dir()))
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME", default_cache_dir()))
    parser.add_argument("--min-free-gb", type=int, default=10)
    args = parser.parse_args()

    save_dir = os.path.abspath(args.save_dir)
    cache_dir = os.path.abspath(args.cache_dir)

    check_free_space(os.path.dirname(save_dir) or ".", args.min_free_gb)
    check_free_space(cache_dir, max(2, args.min_free_gb // 5))

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Downloading {args.model_id} -> {save_dir}")
    print(f"Using cache dir: {cache_dir}")

    snapshot_download(
        repo_id=args.model_id,
        local_dir=save_dir,
        cache_dir=cache_dir,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "tokenizer*",
            "vocab*",
            "merges.txt",
            "*.model",
            "*.tiktoken",
        ],
    )

    config_path = os.path.join(save_dir, "config.json")
    if not os.path.exists(config_path):
        raise RuntimeError(f"Download finished but missing config file: {config_path}")

    print(f"Done! Model files are in: {save_dir}")


if __name__ == "__main__":
    main()
