from jsonargparse import CLI
from typing import List
from pathlib import Path
import warnings

from transformers import AutoModelForCausalLM, AutoTokenizer


def main(output: Path, names: List[str] = None, cache_dir: str = None):
    """Save all models in names in the specified output directory"""
    output.mkdir(exist_ok=True)
    for name in names:
        output_path = output / name.split("/")[-1]
        try:
            tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir, trust_remote_code=True)
            tokenizer.save_pretrained(output_path)
        except Exception as e:
            warnings.warn(f"Ignoring tokenizer exception: {e}")
        model = AutoModelForCausalLM.from_pretrained(name, cache_dir=cache_dir, trust_remote_code=True)
        model.save_pretrained(output_path)


if __name__ == "__main__":
    CLI(main, description=main.__doc__)