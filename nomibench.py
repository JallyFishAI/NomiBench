#!/usr/bin/env python3
"""
NomiBench - a lightweight, model-agnostic LLM benchmark.

Tests a model across five categories:
    knowledge   - General Knowledge
    instruction - Instruction Following
    code        - Code (auto-executed & unit-tested)
    logic       - Logic
    math        - Math

Works with:
    - GGUF models (via llama-cpp-python)   e.g. --model="model.gguf"
    - HuggingFace safetensors model folders (via transformers)
                                             e.g. --model="./MyModel/"

Usage:
    python nomibench.py --model="Model.gguf"
    python nomibench.py --model="./my-model-folder"
    python nomibench.py --model="Model.gguf" --test=code
    python nomibench.py --model="Model.gguf" --test=code,knowledge
    python nomibench.py --model="Model.gguf" --time-budget=1800 --output=results.json

See README.md for full documentation.
"""

import argparse
import json
import os
import sys
import time

from nomibench_data import QUESTION_BANK, CATEGORY_ORDER, CATEGORY_LABELS, DEFAULT_MAX_TOKENS
from nomibench_eval import evaluate

VERSION = "1.0.0"

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


# --------------------------------------------------------------------------
# Model backends
# --------------------------------------------------------------------------

class GGUFBackend:
    """Backend for .gguf models via llama-cpp-python."""

    def __init__(self, model_path, n_ctx=4096, n_gpu_layers=-1, seed=42, verbose=False):
        try:
            from llama_cpp import Llama
        except ImportError:
            sys.exit(
                "ERROR: llama-cpp-python is not installed.\n"
                "Install it with:  pip install llama-cpp-python\n"
                "(For GPU acceleration, see the llama-cpp-python README for build flags.)"
            )
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            verbose=verbose,
        )
        self.name = os.path.basename(model_path)
        self.kind = "GGUF"

    def generate(self, prompt, max_tokens=256, temperature=0.2):
        try:
            out = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return out["choices"][0]["message"]["content"].strip()
        except Exception:
            out = self.llm(prompt, max_tokens=max_tokens, temperature=temperature)
            return out["choices"][0]["text"].strip()


class HFBackend:
    """Backend for HuggingFace model folders (safetensors) via transformers."""

    def __init__(self, model_path, seed=42, verbose=False):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError:
            sys.exit(
                "ERROR: transformers/torch are not installed.\n"
                "Install them with:  pip install torch transformers accelerate"
            )
        self.torch = torch
        if not verbose:
            try:
                from transformers.utils import logging as hf_logging
                hf_logging.set_verbosity_error()
            except Exception:
                pass

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
        self.model.eval()
        self.name = os.path.basename(os.path.normpath(model_path))
        self.kind = "HF/safetensors"
        torch.manual_seed(seed)

    def generate(self, prompt, max_tokens=256, temperature=0.2):
        torch = self.torch
        if getattr(self.tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(self.model.device)
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 0.01),
            pad_token_id=self.tokenizer.eos_token_id or self.tokenizer.pad_token_id,
        )
        with torch.no_grad():
            output = self.model.generate(input_ids, **gen_kwargs)
        text = self.tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
        return text.strip()


def load_backend(model_path, args):
    if os.path.isfile(model_path) and model_path.lower().endswith(".gguf"):
        return GGUFBackend(model_path, n_ctx=args.n_ctx, n_gpu_layers=args.gpu_layers,
                            seed=args.seed, verbose=args.verbose)
    if os.path.isdir(model_path):
        return HFBackend(model_path, seed=args.seed, verbose=args.verbose)
    sys.exit(
        f"ERROR: could not recognize --model '{model_path}'.\n"
        "Provide either a .gguf file or a directory containing a safetensors model."
    )


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------

def build_prompt(category, question):
    """Instruction-following prompts are used verbatim (that's what we're
    testing). Other categories get a light nudge to make final-answer
    extraction reliable."""
    prompt = question["prompt"]
    if category == "instruction":
        return prompt
    if question["type"] == "code":
        return prompt
    if question["type"] in ("numeric", "multiple_choice", "single_letter", "yes_no"):
        return prompt + "\n\nThink if needed, but clearly state your final answer, e.g. 'Answer: <your answer>'."
    return prompt + "\n\nGive a brief, direct answer."


# --------------------------------------------------------------------------
# Terminal output
# --------------------------------------------------------------------------

def colorize(text, color, enabled):
    return f"{color}{text}{RESET}" if enabled else text


def print_banner(model_name, model_kind, categories, args):
    use_color = not args.no_color
    b = BOLD if use_color else ""
    r = RESET if use_color else ""
    print(f"{b}{'=' * 72}{r}")
    print(f"{b}NomiBench v{VERSION}{r}")
    print(f"{'=' * 72}")
    print(f"Model      : {model_name}  ({model_kind})")
    print(f"Categories : {', '.join(CATEGORY_LABELS[c] for c in categories)}")
    print(f"Time budget: {args.time_budget / 60:.0f} minutes")
    print(f"Temperature: {args.temperature}")
    print(f"{'=' * 72}\n")


def print_category_header(category, n_questions, use_color):
    label = CATEGORY_LABELS[category]
    print(f"\n{colorize(f'--- {label} ({n_questions} questions) ---', CYAN + BOLD, use_color)}")


def print_live(category, i, total, question, correct, elapsed, detail, use_color, verbose, response):
    status = colorize("CORRECT  ", GREEN + BOLD, use_color) if correct else colorize("INCORRECT", RED + BOLD, use_color)
    short_q = question["prompt"].replace("\n", " ")
    if len(short_q) > 65:
        short_q = short_q[:62] + "..."
    print(f"  [{i:>2}/{total:<2}] {status} | {elapsed:5.1f}s | {short_q}")
    if verbose:
        short_resp = response.replace("\n", " ")
        if len(short_resp) > 200:
            short_resp = short_resp[:200] + "..."
        print(f"           {DIM}model answer: {short_resp}{RESET}")
        print(f"           {DIM}{detail}{RESET}")


def print_summary(results, total_time, stopped_early, use_color):
    b = BOLD if use_color else ""
    r = RESET if use_color else ""
    print(f"\n{b}{'=' * 72}{r}")
    print(f"{b}NOMIBENCH RESULTS SUMMARY{r}")
    print(f"{'=' * 72}")
    grand_correct = 0
    grand_total = 0
    for category in CATEGORY_ORDER:
        if category not in results:
            continue
        items = results[category]
        correct = sum(1 for r in items if r["correct"])
        total = len(items)
        if total == 0:
            continue
        pct = 100 * correct / total
        grand_correct += correct
        grand_total += total
        bar = _bar(pct)
        color = GREEN if pct >= 70 else (YELLOW if pct >= 40 else RED)
        label = CATEGORY_LABELS[category]
        print(f"{label:<24} {correct:>3}/{total:<3}  {colorize(f'{pct:5.1f}%', color, use_color)}  {bar}")
    print(f"{'-' * 72}")
    overall_pct = 100 * grand_correct / grand_total if grand_total else 0
    color = GREEN if overall_pct >= 70 else (YELLOW if overall_pct >= 40 else RED)
    print(f"{b}{'OVERALL':<24} {grand_correct:>3}/{grand_total:<3}  "
          f"{colorize(f'{overall_pct:5.1f}%', color, use_color)}  {_bar(overall_pct)}{r}")
    print(f"\nTotal time: {total_time / 60:.1f} minutes")
    if stopped_early:
        print(colorize("Note: benchmark stopped early - time budget exceeded. "
                        "Remaining questions were skipped.", YELLOW, use_color))
    print(f"{'=' * 72}\n")


def _bar(pct, width=20):
    filled = round(width * pct / 100)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def run_benchmark(backend, categories, args):
    results = {}
    start_time = time.time()
    stopped_early = False
    use_color = not args.no_color

    for category in categories:
        questions = QUESTION_BANK[category]
        if args.num_questions:
            questions = questions[: args.num_questions]
        if not questions:
            continue

        print_category_header(category, len(questions), use_color)
        cat_results = []

        for i, q in enumerate(questions, 1):
            elapsed_total = time.time() - start_time
            if elapsed_total > args.time_budget:
                stopped_early = True
                skipped = len(questions) - i + 1
                print(colorize(f"  Time budget exceeded - skipping remaining "
                                f"{skipped} question(s) in this category.", YELLOW, use_color))
                break

            max_tokens = args.max_tokens or q.get("max_tokens") or DEFAULT_MAX_TOKENS[category]
            prompt = build_prompt(category, q)

            t0 = time.time()
            try:
                response = backend.generate(prompt, max_tokens=max_tokens, temperature=args.temperature)
            except Exception as e:
                response = ""
                correct, detail = False, f"generation error: {e}"
            else:
                correct, detail = evaluate(q, response)
            dt = time.time() - t0

            cat_results.append({
                "id": q["id"],
                "prompt": q["prompt"],
                "response": response,
                "correct": correct,
                "detail": detail,
                "time_s": round(dt, 2),
            })
            print_live(category, i, len(questions), q, correct, dt, detail, use_color, args.verbose, response)

            if stopped_early:
                break

        results[category] = cat_results
        if stopped_early:
            break

    total_time = time.time() - start_time
    return results, total_time, stopped_early


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_test_arg(raw):
    if not raw or raw.strip().lower() == "all":
        return list(CATEGORY_ORDER)
    requested = [t.strip().lower() for t in raw.split(",") if t.strip()]
    unknown = [t for t in requested if t not in CATEGORY_ORDER]
    if unknown:
        valid = ", ".join(CATEGORY_ORDER)
        sys.exit(f"ERROR: unknown --test value(s): {', '.join(unknown)}. Valid options: {valid}, all")
    # preserve canonical order regardless of user's order
    return [c for c in CATEGORY_ORDER if c in requested]


def build_parser():
    p = argparse.ArgumentParser(
        prog="nomibench.py",
        description="NomiBench - a lightweight, model-agnostic LLM benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True,
                    help="Path to a .gguf file, or a directory containing a safetensors model.")
    p.add_argument("--test", default="all",
                    help="Comma-separated categories to run: knowledge,instruction,code,logic,math or 'all'.")
    p.add_argument("--num-questions", type=int, default=0,
                    help="Limit the number of questions per category (0 = use full bank, 15/category).")
    p.add_argument("--max-tokens", type=int, default=0,
                    help="Override max generation tokens for every question (0 = use per-question defaults).")
    p.add_argument("--temperature", type=float, default=0.2,
                    help="Sampling temperature.")
    p.add_argument("--time-budget", type=float, default=3600,
                    help="Hard cap on total benchmark runtime in seconds. Remaining questions are "
                         "skipped once exceeded.")
    p.add_argument("--n-ctx", type=int, default=4096,
                    help="Context window size (GGUF backend only).")
    p.add_argument("--gpu-layers", type=int, default=-1,
                    help="Number of layers to offload to GPU (GGUF backend only). -1 = all.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--output", default="", help="Path to save detailed JSON results.")
    p.add_argument("--no-color", action="store_true", help="Disable colored terminal output.")
    p.add_argument("--verbose", action="store_true", help="Show full model responses and grading detail.")
    return p


def main():
    args = build_parser().parse_args()
    categories = parse_test_arg(args.test)

    print(f"Loading model from '{args.model}' ...")
    backend = load_backend(args.model, args)
    print_banner(backend.name, backend.kind, categories, args)

    results, total_time, stopped_early = run_benchmark(backend, categories, args)
    print_summary(results, total_time, stopped_early, not args.no_color)

    if args.output:
        payload = {
            "model": backend.name,
            "backend": backend.kind,
            "categories_tested": categories,
            "total_time_seconds": round(total_time, 2),
            "stopped_early": stopped_early,
            "results": results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Detailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
