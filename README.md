<a target="_blank" href="https://colab.research.google.com/github/PaulLerner/21-EuroParl/blob/main/analyze.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# 21-EuroParl
Dataset and code for the paper "Assessing the Political Fairness of Multilingual LLMs: A Case Study based on a 21-way Multiparallel EuroParl Dataset" (Lerner and Yvon, 2025)

## Dataset

Get the dataset from https://huggingface.co/datasets/PaulLerner/21-EuroParl

```py
from datasets import load_dataset
dataset = load_dataset("PaulLerner/21-EuroParl")
```

See [below](#alignment) how the dataset was aligned.

## Experiments

Reproduce our experiments as instructed below or analyze our results from https://huggingface.co/datasets/PaulLerner/21-EuroParl-Pred/

```bash
git clone https://huggingface.co/datasets/PaulLerner/21-EuroParl-Pred
```

### Installation
```bash
git clone https://github.com/PaulLerner/21-EuroParl
cd 21-EuroParl
uv sync
```

### Prompt
Use `python -m text2text.prompt -h` to see CLI arguments. Otherwise, use our configs from `21-EuroParl-Pred`.

```bash
python -m text2text.prompt --config=/path/to/config.yaml
```

This prompts a model on a language pair. Obviously, you're free to run a for loop to launch one SLURM job per language pair (see for example [the trad.py script](21-EuroParl/trad.py)).

To print the default config, use
```yaml
$ python -m text2text.prompt --print_config
output_path: null
data_path: null
llm_arguments:
  model: null
  tokenizer: null
  tokenizer_mode: auto
  skip_tokenizer_init: false
  trust_remote_code: true
  tensor_parallel_size: 1
  dtype: auto
  quantization: null
  revision: null
  tokenizer_revision: null
  seed: 0
  gpu_memory_utilization: 0.9
  swap_space: 4
  cpu_offload_gb: 0
  enforce_eager: false
  max_seq_len_to_capture: 8192
  disable_custom_all_reduce: false
  disable_async_output_proc: false
  mm_processor_kwargs: null
  max_model_len: null
sampling_arguments:
  n: 1
  best_of: null
  presence_penalty: 0.0
  frequency_penalty: 0.0
  repetition_penalty: 1.0
  temperature: 1.0
  top_p: 1.0
  top_k: 0
  min_p: 0.0
  seed: null
  stop: null
  stop_token_ids: null
  ignore_eos: false
  max_tokens: 16
  min_tokens: 0
  logprobs: null
  prompt_logprobs: null
  detokenize: true
  skip_special_tokens: true
  spaces_between_special_tokens: true
  logits_processors: null
  include_stop_str_in_output: false
  truncate_prompt_tokens: null
input_key: text
template: '{text}'
chat: true
join: false
shuffle: false
split: test
```

### Metrics
Likewise, use `-h` to see arguments but it's much more convenient to use YAML.

```bash
python -m text2text.metrics --config=/path/to/config.yaml
```

### Analyze
You can analyze our results following [the notebook](analyze.ipynb)

## Alignment
Note: you don't need to do any of that, see [above](#dataset) to download the aligned dataset

You can also start from the alignments if you want to extend our dataset (e.g. include one-to-many alignments, change the threshold etc.): https://huggingface.co/datasets/PaulLerner/21-EuroParl-Align

- Get the [LinkedEP knowledge graph](https://ssh.datastations.nl/dataset.xhtml?persistentId=doi:10.17026/dans-x62-ew3m) ([van Aggelen et al., 2016](https://journals.sagepub.com/doi/full/10.3233/SW-160227))
- Run `python data/rdf.py /path/to/linkedep` to query the RDF (`/path/to/linkedep` should hold the .ttl files downloaded above)
- Filter out missing data using fasttext LIDs: [`data/lid.ipynb`](data/lid.ipynb)
- See https://github.com/PaulLerner/bertalign to align (the output should be like https://huggingface.co/datasets/PaulLerner/21-EuroParl-Align)
- Merge bialignments into multialignments: [`data/merge_align.ipynb`](data/merge_align.ipynb)

Done! The result should be like https://huggingface.co/datasets/PaulLerner/21-EuroParl




## Citation

TODO
