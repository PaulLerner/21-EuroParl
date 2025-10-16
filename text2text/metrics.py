from jsonargparse import CLI
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List
import json
import pandas as pd
from tqdm import tqdm

from sacrebleu.metrics import BLEU
from comet import load_from_checkpoint, download_model

from .utils import get_split, load_dataset


@dataclass
class BleuKwargs:
    lowercase: bool = False
    force: bool = False
    tokenize: Optional[str] = None
    smooth_method: str = 'exp'
    smooth_value: Optional[float] = None
    max_ngram_order: int = 4
    effective_order: bool = False
    trg_lang: str = ''


def compute_bleu(bleu, predictions, references, sentence_level: bool = False):
    if sentence_level:
        bleus, bps = [], []
        for hypothesis, *reference in zip(predictions, *references):
            bleu_score = bleu.sentence_score(hypothesis, reference)
            bleus.append(bleu_score.score)
            bps.append(bleu_score.bp)
        metrics = {
            "BLEU": sum(bleus)/len(bleus),
            "BP": sum(bps)/len(bps),
            "BLEUs": bleus,
            "BPs": bps
        }
    else:
        bleu_score = bleu.corpus_score(predictions, references)
        metrics = {
            "BLEU": bleu_score.score,
            "BP": bleu_score.bp
        }
    metrics.update({
        "sacrebleu": str(bleu.get_signature()),
        "sentence_level": sentence_level
    })
    return metrics


def main(output_dir: str, data_path: str, languages: List[str], bleu_kwargs: BleuKwargs = BleuKwargs(), 
         split: str = "test", sentence_level: bool = False, batch_size: int = 64):
    # boilerplate needed because of https://github.com/Unbabel/COMET/issues/250
    output_dir = Path(output_dir)
    data_path = Path(data_path)
    bleu = BLEU(**asdict(bleu_kwargs))
    wmt22_comet_da = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
    dataset = load_dataset(data_path)
    dataset = get_split(dataset, split)
    all_metrics = []
    for tgt in tqdm(languages):
        references = [list(dataset[tgt])]
        for src in languages:
            if src == tgt:
                continue
            output_path = output_dir/f"{src}-{tgt}_output.json"
            if not output_path.exists():
                continue
            with open(output_path, 'rt') as file:
                predictions = json.load(file)
            metrics = compute_bleu(bleu, predictions, references, sentence_level=sentence_level)
            metrics.update({"src": src, "tgt": tgt})
            comet_data = [{"src": s, "mt": prediction, "ref": reference} for s, prediction, reference in zip(dataset[src], predictions, references[0])]
            wmt22_comet_da_metrics = wmt22_comet_da.predict(comet_data, batch_size=batch_size, gpus=1, progress_bar=False)
            metrics.update({"wmt22-comet-da": wmt22_comet_da_metrics.system_score, "wmt22-comet-das": wmt22_comet_da_metrics.scores})
            all_metrics.append(metrics)
    with open(output_dir/"metrics.json", "wt") as file:
        json.dump(all_metrics, file)
    all_metrics = pd.DataFrame([{k: v for k, v in metrics.items() if isinstance(v, (float, str))} for metrics in all_metrics])
    print(all_metrics.to_markdown(floatfmt=".3f"))


if __name__ == '__main__':
    CLI(main)
