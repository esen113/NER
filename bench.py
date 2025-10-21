import argparse, os, json, time, csv
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

from prompting import build_messages
from utils import robust_json_parse, to_tuple_list, read_dataset, relation_vocab_of_dataset
from metrics import micro_prf, headtail_micro_prf, entity_micro_prf, relation_bag_micro_prf, per_relation_f1, macro_from_per_class

def run_one_model_on_dataset(
    backend: str,
    model_name: str,
    model_path: str,
    dataset_path: str,
    save_dir: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    batch_size: int = 8,
    dtype: str = "auto",
    device_map: str = "auto",
    tensor_parallel_size: int = 1,
    n_ctx: int = 4096,
    n_gpu_layers: int = 0,
    max_samples: int = 0,
    use_rel_vocab: bool = True,
) -> Dict[str, Any]:
    ds = read_dataset(dataset_path)
    if max_samples and max_samples > 0:
        ds = ds[:max_samples]

    rel_vocab = relation_vocab_of_dataset(ds) if use_rel_vocab else []

    # Build messages
    messages_list = [build_messages(item["text"], rel_vocab) for item in ds]

    # Inference
    if backend == "transformers":
        from backends import infer_transformers
        outputs = infer_transformers(model_path, messages_list, max_new_tokens, temperature, top_p, dtype, device_map)
    elif backend == "vllm":
        from backends import infer_vllm
        outputs = infer_vllm(model_path, messages_list, max_new_tokens, temperature, top_p, tensor_parallel_size, True, batch_size)
    elif backend == "llama_cpp":
        from backends import infer_llama_cpp
        outputs = infer_llama_cpp(model_path, messages_list, max_new_tokens, temperature, top_p, n_ctx, n_gpu_layers)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    preds = []
    for out in outputs:
        obj = robust_json_parse(out)
        triples = to_tuple_list(obj)
        preds.append(triples)

    gold = [item["triple_list"] for item in ds]

    # Metrics
    triple_em = micro_prf(preds, gold)
    ht = headtail_micro_prf(preds, gold)
    ent = entity_micro_prf(preds, gold)
    rel = relation_bag_micro_prf(preds, gold)
    per_rel = per_relation_f1(preds, gold)
    macro_rel = macro_from_per_class(per_rel)

    # Save artifacts
    model_dir = os.path.join(save_dir, os.path.basename(dataset_path).replace(".json",""), model_name)
    os.makedirs(model_dir, exist_ok=True)
    # Save predictions
    with open(os.path.join(model_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for item, out, tri in zip(ds, outputs, preds):
            rec = {"text": item["text"], "pred_triples": tri, "gold_triples": item["triple_list"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # Save metrics
    metrics = {
        "backend": backend,
        "model_name": model_name,
        "dataset": os.path.basename(dataset_path),
        "num_samples": len(ds),
        "triple_em_micro": triple_em,
        "head_tail_micro": ht,
        "entity_micro": ent,
        "relation_bag_micro": rel,
        "per_relation_f1": per_rel,
        "per_relation_macro": macro_rel,
    }
    with open(os.path.join(model_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", required=True, help="Path to JSON dataset; can appear multiple times")
    ap.add_argument("--backend", choices=["transformers", "vllm", "llama_cpp"], required=True)
    ap.add_argument("--model-name", action="append", required=True, help="Logical model name (for saving)")
    ap.add_argument("--model-path", action="append", required=True, help="Local path; must match order of --model-name")
    ap.add_argument("--save-dir", default="./runs")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=8, help="For vLLM")
    ap.add_argument("--dtype", default="auto", help="transformers dtype: auto/fp16/bf16")
    ap.add_argument("--device-map", default="auto", help="transformers device_map")
    ap.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM TP size")
    ap.add_argument("--n-ctx", type=int, default=4096, help="llama.cpp context")
    ap.add_argument("--n-gpu-layers", type=int, default=0, help="llama.cpp gpu layers")
    ap.add_argument("--max-samples", type=int, default=0, help="debug subset; 0 = all")
    ap.add_argument("--no-rel-vocab", action="store_true", help="Do NOT inject relation vocab into prompts")
    args = ap.parse_args()

    if len(args.model_name) != len(args.model_path):
        raise SystemExit("model-name and model-path counts must match")

    os.makedirs(args.save_dir, exist_ok=True)

    summary_rows = []
    for ds_path in args.dataset:
        for name, path in zip(args.model_name, args.model_path):
            print(f"\n=== Running {name} on {ds_path} ({args.backend}) ===")
            metrics = run_one_model_on_dataset(
                backend=args.backend,
                model_name=name, model_path=path,
                dataset_path=ds_path, save_dir=args.save_dir,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p,
                batch_size=args.batch_size, dtype=args.dtype, device_map=args.device_map,
                tensor_parallel_size=args.tensor_parallel_size, n_ctx=args.n_ctx, n_gpu_layers=args.n_gpu_layers,
                max_samples=args.max_samples,
                use_rel_vocab=(not args.no_rel_vocab),
            )
            row = {
                "dataset": metrics["dataset"],
                "backend": metrics["backend"],
                "model_name": metrics["model_name"],
                "num_samples": metrics["num_samples"],
                "triple_em_f1": metrics["triple_em_micro"]["f1"],
                "triple_em_p": metrics["triple_em_micro"]["precision"],
                "triple_em_r": metrics["triple_em_micro"]["recall"],
                "head_tail_f1": metrics["head_tail_micro"]["f1"],
                "entity_f1": metrics["entity_micro"]["f1"],
                "relation_bag_f1": metrics["relation_bag_micro"]["f1"],
                "relation_macro_f1": metrics["per_relation_macro"]["f1"],
            }
            summary_rows.append(row)

    # write summary
    summary_csv = os.path.join(args.save_dir, "summary.csv")
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nWrote summary to: {summary_csv}")

if __name__ == "__main__":
    main()
