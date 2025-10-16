from jsonargparse import CLI
from pathlib import Path
from typing import List
import yaml
import os

from iso639 import Lang


def main(languages: List[str], original_config: str):
    original_config = Path(original_config)
    with open(original_config, 'rt') as file:
        config = yaml.safe_load(file)
    trad_template = config["template"]
    output_dir = Path(config["output_path"]).parent
    for i in range(len(languages)):
        for j in range(len(languages)):
            if i == j:
                continue
            output_path = output_dir/f"{languages[i]}-{languages[j]}_output.json"
            if output_path.exists():
                continue
            config["template"] = trad_template.format(src=Lang(languages[i]).name, tgt=Lang(languages[j]).name, text="{text}")
            config["output_path"] = str(output_path)
            config["input_key"] = languages[i]
            new_config_path = output_dir/f"{languages[i]}-{languages[j]}_config.yaml"
            with open(new_config_path, "wt") as file:
                yaml.safe_dump(config, file)
            print(new_config_path, "\n")
            print(config, "\n\n")
            os.system(f"sbatch 1-v100-32g_20h-any.sh python -m text2text.prompt --config={new_config_path}")


if __name__ == '__main__':
    CLI(main)