import json
from typing import List, Dict, Any, Optional, Iterable

def _apply_chat_template_if_available(tokenizer, messages: List[Dict[str, str]]) -> Optional[str]:
    # Use tokenizer's chat template if defined, else return None
    tmpl = getattr(tokenizer, "chat_template", None)
    if tmpl:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return None
    return None

# --------- transformers backend ---------
def infer_transformers(model_path: str, messages_list: List[List[Dict[str, str]]], 
                       max_new_tokens: int = 256, temperature: float = 0.0, top_p: float = 1.0,
                       dtype: str = "auto", device_map: str = "auto") -> List[str]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=(torch.float16 if dtype in ("fp16","half") else ("auto" if dtype=="auto" else torch.bfloat16)),
        device_map=device_map,
        trust_remote_code=True,
    )
    outs = []
    for messages in messages_list:
        prompt = _apply_chat_template_if_available(tokenizer, messages)
        if prompt is None:
            # Basic fallback
            prompt = ""
            for m in messages:
                prompt += f"<|{m['role']}|> {m['content'].strip()}\n"
            prompt += "<|assistant|>"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        outs.append(text.strip())
    return outs

# --------- vLLM backend ---------
def infer_vllm(model_path: str, messages_list: List[List[Dict[str, str]]],
               max_new_tokens: int = 256, temperature: float = 0.0, top_p: float = 1.0,
               tensor_parallel_size: int = 1, trust_remote_code: bool = True, batch_size: int = 16) -> List[str]:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size, trust_remote_code=trust_remote_code)
    prompts = []
    for messages in messages_list:
        prompt = _apply_chat_template_if_available(tokenizer, messages)
        if prompt is None:
            # degrade to a simple concatenation
            prompt = ""
            for m in messages:
                prompt += f"<|{m['role']}|> {m['content'].strip()}\n"
            prompt += "<|assistant|>"
        prompts.append(prompt)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
    )
    rets = llm.generate(prompts, sampling_params, use_tqdm=True)
    return [out.outputs[0].text.strip() for out in rets]

# --------- llama.cpp backend ---------
def infer_llama_cpp(model_path: str, messages_list: List[List[Dict[str, str]]],
                    max_new_tokens: int = 256, temperature: float = 0.0, top_p: float = 1.0,
                    n_ctx: int = 4096, n_gpu_layers: int = 0) -> List[str]:
    from llama_cpp import Llama
    llm = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, logits_all=False)
    outs = []
    for messages in messages_list:
        res = llm.create_chat_completion(messages=messages, temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)
        text = res["choices"][0]["message"]["content"].strip()
        outs.append(text)
    return outs
