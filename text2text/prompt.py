import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Any, Dict, Union
from jsonargparse import CLI

from vllm import LLM, SamplingParams

from .utils import get_split, load_dataset


@dataclass
class LlmArguments:
    model: str
    tokenizer: Optional[str] = None
    tokenizer_mode: str = "auto"
    skip_tokenizer_init: bool = False
    trust_remote_code: bool = True
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    quantization: Optional[str] = None
    revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    seed: int = 0
    gpu_memory_utilization: float = 0.9
    swap_space: float = 4
    cpu_offload_gb: float = 0
    enforce_eager: Optional[bool] = False
    max_seq_len_to_capture: int = 8192
    disable_custom_all_reduce: bool = False
    disable_async_output_proc: bool = False
    mm_processor_kwargs: Optional[Dict[str, Any]] = None
    max_model_len: int = None


@dataclass
class SamplingArguments:
    n: int = 1
    best_of: Optional[int] = None
    _real_n: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    seed: Optional[int] = None
    stop: Optional[Union[str, list[str]]] = None
    stop_token_ids: Optional[list[int]] = None
    ignore_eos: bool = False
    max_tokens: Optional[int] = 16
    min_tokens: int = 0
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None
    detokenize: bool = True
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    logits_processors: Optional[Any] = None
    include_stop_str_in_output: bool = False
    truncate_prompt_tokens: Optional[int] = None


def prompt(output_path: str, data_path: str, llm_arguments: LlmArguments,
           sampling_arguments: SamplingArguments = SamplingArguments(), 
           input_key: str = "text", template: str = "{text}", chat: bool = True, 
           join: bool = False, shuffle: bool = False, split: str = "test"
    ):
    output_path = Path(output_path)
    data_path = Path(data_path)
    if shuffle:
        assert join, "shuffle has no effect if not join"
        #TODO
        raise NotImplementedError()
    dataset = load_dataset(data_path)
    dataset = get_split(dataset, split)
    texts = dataset[input_key]
    if join: 
        text = "\n\n".join(texts)
        prompts = [template.format(text=text)]
    else:
        prompts = [template.format(text=text) for text in texts]
    llm = LLM(**asdict(llm_arguments))
    sampling_params = SamplingParams(**asdict(sampling_arguments))

    if chat:
        prompts = [[{"role": "user", "content": p}] for p in prompts]
        # HACK: disable thinking
        outputs = llm.chat(prompts, sampling_params, chat_template_kwargs=dict(enable_thinking=False))
    else:
        outputs = llm.generate(prompts, sampling_params)

    outputs_texts = [output.outputs[0].text for output in outputs]
    with open(output_path, "wt") as file:
        json.dump(outputs_texts, file)


def main():
    CLI(prompt, description=__doc__)


if __name__ == '__main__':
    main()
