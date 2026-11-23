import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import dotenv_values
from openai import OpenAI
import tiktoken


REPO_ROOT = Path(os.environ.get("CODERAG_BENCH_ROOT", Path.cwd())).resolve()
GENERATION_DIR = REPO_ROOT / "generation"
DEFAULT_DATA = REPO_ROOT / "retrieval/results/repoeval-function-2k-bm25-filetop15-top5-sample60-10perrepo-with-metadata.jsonl"
DEFAULT_CODEGIST_ROOT = Path(os.environ.get("CODEGIST_ROOT", Path(__file__).resolve().parents[1]))
DEFAULT_DOC_METADATA = REPO_ROOT / "tmp/repoeval_bm25_filetop5_noleak_doc_instances/metadata.json"

os.chdir(GENERATION_DIR)
sys.path.insert(0, str(GENERATION_DIR))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_repoeval_api_codegist import import_codegist, sketch_source  # noqa: E402
from run_repoeval_api_currentfunc import build_messages, clean_completion  # noqa: E402


def env_value(values, key):
    value = os.environ.get(key) or values.get(key)
    if value is None:
        raise ValueError(f"Missing {key} in .env")
    return value.strip().strip('"').strip("'")


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_json(path):
    if path and path.exists():
        return json.load(path.open("r", encoding="utf-8"))
    return []


def save_json(path, data):
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False)


def comment_text(text):
    return "\n".join("# " + line if line else "#" for line in text.rstrip("\n").splitlines())


def load_doc_index(metadata_path):
    payload = json.load(metadata_path.open("r", encoding="utf-8"))
    by_row = {}
    for item in payload["docs"]:
        by_row.setdefault(item["row_idx"], []).append(item)
    for row_docs in by_row.values():
        row_docs.sort(key=lambda item: item["rank"])
    return by_row


def build_file_context(record, mode, cg=None, stats=None, row_docs=None, compressed_docs_dir=None):
    lines = ["# Here are some relevant full Python files from the repo:", "# --------------------------------------------------"]
    docs = row_docs[:5] if row_docs is not None else record.get("docs", [])[:5]
    for idx, doc in enumerate(docs):
        title = doc["title"]
        lines.append("# the below Python file can be found in:")
        lines.append(f"# {title}")
        lines.append("# --------------------------------------------------")
        if compressed_docs_dir is not None:
            path = compressed_docs_dir / doc["relative_path"]
            if not path.exists():
                raise FileNotFoundError(f"Missing compressed doc: {path}")
            rendered = path.read_text(encoding="utf-8", errors="replace").rstrip("\n")
        elif mode == "raw":
            text = doc.get("text", "")
            rendered = text.rstrip("\n")
        else:
            text = doc.get("text", "")
            rendered = sketch_source(
                cg,
                text,
                f"{record['metadata']['task_id']}::file_doc{idx}",
                stats,
                add_header=False,
            ).rstrip("\n")
        lines.append(comment_text(rendered))
        lines.append("# --------------------------------------------------")
    return "\n".join(lines) + "\n"


def select_records(records, limit_start, limit):
    end = None if limit is None else limit_start + limit
    return records[limit_start:end]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default=str(DEFAULT_DATA))
    parser.add_argument("--mode", choices=["raw", "sketch"], required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--raw_output_path")
    parser.add_argument("--prompt_debug_path")
    parser.add_argument("--usage_path")
    parser.add_argument("--stats_path")
    parser.add_argument("--env_file", default=str(REPO_ROOT / ".env"))
    parser.add_argument("--codegist_root", default=str(DEFAULT_CODEGIST_ROOT))
    parser.add_argument("--compressed_docs_dir")
    parser.add_argument("--doc_metadata", default=str(DEFAULT_DOC_METADATA))
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit_start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    env = dotenv_values(args.env_file)
    model = env_value(env, "LLM_MODEL")
    client = OpenAI(
        api_key=env_value(env, "LLM_API_KEY"),
        base_url=env_value(env, "LLM_MODEL_URL"),
        timeout=240,
        max_retries=2,
    )

    enc = tiktoken.get_encoding("o200k_base")

    records = select_records(load_jsonl(resolve_path(args.data_file)), args.limit_start, args.limit)
    output_path = resolve_path(args.output_path)
    raw_output_path = resolve_path(args.raw_output_path) if args.raw_output_path else None
    prompt_debug_path = resolve_path(args.prompt_debug_path) if args.prompt_debug_path else None
    usage_path = resolve_path(args.usage_path) if args.usage_path else None
    stats_path = resolve_path(args.stats_path) if args.stats_path else None

    generations = load_json(output_path)
    raw_generations = load_json(raw_output_path) if raw_output_path else []
    debug_prompts = load_json(prompt_debug_path) if prompt_debug_path else []
    usage_records = load_json(usage_path) if usage_path else []

    if raw_output_path and raw_generations and len(generations) != len(raw_generations):
        raise ValueError("Existing cropped and raw output lengths differ.")
    if usage_path and usage_records and len(generations) != len(usage_records):
        raise ValueError("Existing generation and usage output lengths differ.")

    compressed_docs_dir = resolve_path(args.compressed_docs_dir) if args.compressed_docs_dir else None
    doc_index = load_doc_index(resolve_path(args.doc_metadata)) if compressed_docs_dir else None
    cg = import_codegist(args.codegist_root) if args.mode == "sketch" and compressed_docs_dir is None else None
    stats = {
        "mode": args.mode,
        "precompressed_docs_dir": str(compressed_docs_dir) if compressed_docs_dir else None,
        "doc_metadata": str(resolve_path(args.doc_metadata)) if compressed_docs_dir else None,
        "codegist": {"compressed": 0, "repaired": 0, "fallback_original": 0, "empty": 0, "errors": []},
    }

    for idx in range(len(generations), len(records)):
        record = records[idx]
        global_idx = args.limit_start + idx
        context_stats = stats["codegist"] if args.mode == "sketch" else None
        row_docs = doc_index.get(global_idx) if doc_index is not None else None
        if doc_index is not None and not row_docs:
            raise KeyError(f"No precompressed docs found for row_idx={global_idx}")
        retrieved_context = build_file_context(
            record,
            args.mode,
            cg=cg,
            stats=context_stats,
            row_docs=row_docs,
            compressed_docs_dir=compressed_docs_dir,
        )
        prompt_mode = "docs" if args.mode == "sketch" else "baseline"
        messages = build_messages(retrieved_context, record["prompt"], mode=prompt_mode)

        for attempt in range(1, args.retries + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                text = response.choices[0].message.content or ""
                break
            except Exception as exc:
                print(
                    f"request failed for idx={global_idx} task={record['metadata']['task_id']} "
                    f"attempt={attempt}: {type(exc).__name__}: {exc}"
                )
                if attempt == args.retries:
                    raise
                time.sleep(5 * attempt)

        generation = clean_completion(text, record["prompt"])
        generations.append([generation])
        if raw_output_path:
            raw_generations.append([text])

        api_usage = response.usage.model_dump() if getattr(response, "usage", None) else None
        prompt_tokens = sum(
            len(enc.encode(message["content"], disallowed_special=()))
            for message in messages
        )
        completion_tokens = len(enc.encode(text, disallowed_special=()))
        if usage_path:
            usage_records.append(
                {
                    "idx": global_idx,
                    "task_id": record["metadata"]["task_id"],
                    "mode": args.mode,
                    "api_usage": api_usage,
                    "local_prompt_tokens": prompt_tokens,
                    "local_completion_tokens": completion_tokens,
                    "local_total_tokens": prompt_tokens + completion_tokens,
                    "raw_chars": len(text),
                    "cropped_chars": len(generation),
                }
            )
        if prompt_debug_path:
            debug_prompts.append(
                {
                    "idx": global_idx,
                    "task_id": record["metadata"]["task_id"],
                    "mode": args.mode,
                    "retrieved_context": retrieved_context,
                    "infile_prompt": record["prompt"],
                    "messages": messages,
                }
            )

        print(
            f"generated {len(generations)}/{len(records)} "
            f"(idx={global_idx}, task={record['metadata']['task_id']}, "
            f"prompt_tokens={prompt_tokens}, raw_chars={len(text)}, cropped_chars={len(generation)})",
            flush=True,
        )

        if len(generations) % args.save_every == 0:
            save_json(output_path, generations)
            save_json(raw_output_path, raw_generations)
            save_json(prompt_debug_path, debug_prompts)
            save_json(usage_path, usage_records)
            save_json(stats_path, stats)
            print(f"saved {len(generations)} generations", flush=True)
        time.sleep(0.1)

    save_json(output_path, generations)
    save_json(raw_output_path, raw_generations)
    save_json(prompt_debug_path, debug_prompts)
    save_json(usage_path, usage_records)
    save_json(stats_path, stats)
    print(f"done: saved {len(generations)} generations to {output_path}")


if __name__ == "__main__":
    main()
