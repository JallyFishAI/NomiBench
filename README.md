# NomiBench

A lightweight, model-agnostic benchmark for local LLMs. It tests a model
across five categories, grades every answer automatically (no human or
"LLM-judge" needed), and prints live pass/fail results in the terminal as
it goes — finishing with a full score breakdown.

```
============================================================
NomiBench v1.0.0
============================================================
Model      : my-model-Q4_K_M.gguf  (GGUF)
Categories : General Knowledge, Instruction Following, Code, Logic, Math
Time budget: 60 minutes
Temperature: 0.2
============================================================

--- General Knowledge (15 questions) ---
  [ 1/15] CORRECT   |  1.2s | What is the capital of Australia?
  [ 2/15] INCORRECT |  0.9s | What is the chemical symbol for gold?
  ...

============================================================
NOMIBENCH RESULTS SUMMARY
============================================================
General Knowledge        13/15   86.7%  [#################---]
Instruction Following    11/15   73.3%  [###############-----]
Code                     12/15   80.0%  [################----]
Logic                    10/15   66.7%  [#############-------]
Math                     14/15   93.3%  [##################--]
------------------------------------------------------------
OVERALL                  60/75   80.0%  [################----]

Total time: 8.4 minutes
============================================================
```

## What it tests

| Category                | What it measures | # Questions |
|--------------------------|-------------------|:-----------:|
| **General Knowledge**    | Facts about science, geography, history, literature | 15 |
| **Instruction Following**| Whether the model precisely obeys formatting/constraint instructions (exact word counts, output formats, JSON, letter restrictions, etc.) | 15 |
| **Code**                 | Python coding tasks — the model's code is **actually executed and unit-tested** in a sandboxed subprocess | 15 |
| **Logic**                | Puzzles, sequences, syllogisms, riddles | 15 |
| **Math**                 | Arithmetic, algebra, geometry, word problems | 15 |

75 questions total. Every answer is graded **deterministically** by code —
there's no LLM-as-judge step, so results are 100% reproducible.

## Requirements

- Python 3.9+
- One of:
  - **GGUF models**: `pip install llama-cpp-python`
  - **safetensors / HuggingFace model folders**: `pip install torch transformers accelerate`

```bash
pip install -r requirements.txt
```

(You only need to install the dependency for the model format you're actually testing.)

## Usage

Test a GGUF model:

```bash
python nomibench.py --model="Model.gguf"
```

Test a HuggingFace-format model (a folder containing `.safetensors`, `config.json`, tokenizer files, etc.):

```bash
python nomibench.py --model="./MyModelFolder"
```

Run only specific categories:

```bash
python nomibench.py --model="Model.gguf" --test=code
python nomibench.py --model="Model.gguf" --test=code,knowledge
```

Valid category names: `knowledge`, `instruction`, `code`, `logic`, `math`, or `all` (default).

Save detailed results (every prompt, response, and grading detail) to a JSON file:

```bash
python nomibench.py --model="Model.gguf" --output=results.json
```

Show full model responses live in the terminal (useful for debugging a model that's scoring low):

```bash
python nomibench.py --model="Model.gguf" --verbose
```

## All options

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | Path to a `.gguf` file or a directory with a safetensors model |
| `--test` | `all` | Comma-separated categories to run |
| `--num-questions` | `0` (= all 15/category) | Limit questions per category, e.g. for a quick smoke test |
| `--max-tokens` | `0` (= per-question default) | Force the same max generation length for every question |
| `--temperature` | `0.2` | Sampling temperature |
| `--time-budget` | `3600` (1 hour) | Hard cap in seconds. If exceeded mid-run, remaining questions are skipped and results reported for what completed |
| `--n-ctx` | `4096` | Context window size (GGUF backend only) |
| `--gpu-layers` | `-1` (all) | Layers to offload to GPU (GGUF backend only) |
| `--seed` | `42` | Random seed |
| `--output` | *(none)* | Path to save a detailed JSON results file |
| `--no-color` | off | Disable ANSI colors (use this if your terminal doesn't render them well) |
| `--verbose` | off | Print full model answers + grading detail for every question |

Quick smoke test (5 questions/category, ~1 minute):

```bash
python nomibench.py --model="Model.gguf" --num-questions=5
```

## About the time budget

The benchmark is sized to comfortably finish **well under an hour** at
~80 tokens/s: 75 questions with an average response length of roughly
200 tokens is about 15,000 generated tokens total, i.e. **~3–5 minutes**
of generation time, plus model loading and prompt-processing overhead.
Realistic total runtime is usually a few minutes to ~15 minutes.

`--time-budget` (default 3600s = 1 hour) is a **hard safety cap**: the
runner checks elapsed time before every question, and if the budget is
exceeded it stops immediately, prints how many questions were skipped,
and still reports full results for everything that did complete. This
means the benchmark can never run away past the 1-hour ceiling even on
very slow hardware or very verbose models — lower it with
`--time-budget=1800` etc. if you want a stricter cap.

## How grading works

Every question has an explicit, automatic grading method — no subjective
scoring:

- **Exact / contains match** — normalized string comparison (case/punctuation-insensitive)
- **Numeric** — the model's final number is extracted (preferring text after "Answer:") and compared with a small tolerance
- **Multiple choice / single letter** — the chosen option letter is extracted and compared
- **Yes/No** — extracted and compared
- **Code** — the model's Python function is extracted from its response and **actually executed** in an isolated subprocess against 2–4 unit tests per task, with a 10-second timeout. A task passes only if *all* its test cases pass.
- **Instruction following (format checks)** — each question is verified programmatically against its specific constraint: exact word count, all-caps, no-letter-'e', valid JSON with specific field values, starts/ends with a given word, exact sentence/line count, numbered list format, etc.

For non-instruction-following categories, NomiBench appends a small nudge
to the prompt (e.g. *"clearly state your final answer, e.g. 'Answer:
<your answer>'"*) so that automatic answer-extraction is reliable across
very different models and prompting styles. Instruction-following prompts
are sent **completely unmodified**, since obeying the instruction exactly
as given is the entire point of that category.

## Model compatibility

NomiBench auto-detects the backend from `--model`:

- Path ends in `.gguf` and is a file → loaded via `llama-cpp-python`, using
  chat-completion mode (falls back to raw completion if the model has no
  chat template).
- Path is a directory → loaded via `transformers`
  (`AutoModelForCausalLM` + `AutoTokenizer`), with `device_map="auto"` so it
  will use GPU automatically if available. Uses the model's chat template
  if the tokenizer defines one.

This covers essentially any local instruct/chat model in either format —
Llama, Mistral, Qwen, Gemma, Phi, etc.

## Extending the benchmark

All questions live in `nomibench_data.py` as plain Python dicts — add new
ones to any category's list following the existing pattern. Supported
`type` values (see `nomibench_eval.py` for grading logic) are:

`exact`, `numeric`, `multiple_choice`, `single_letter`, `yes_no`,
`contains_any`, `contains_all`, `code`, `format_check`.

For `format_check` (instruction following) questions, `check` selects one
of the functions in `FORMAT_CHECKS` in `nomibench_eval.py`
(`exact_match`, `word_count_exact`, `all_caps`, `no_letter`, `valid_json`,
`starts_with`, `ends_with`, `sentence_count`, `line_count`,
`numbered_list`, `single_word_contains`, `yes_no`) — add a new function
and register it there to create new constraint types.

## Files

```
nomibench.py        - CLI entry point, model backends, runner, terminal output
nomibench_data.py    - the question bank (75 questions, 5 categories)
nomibench_eval.py    - all automatic grading logic
requirements.txt     - optional dependencies (install what you need)
README.md            - this file
```

## Troubleshooting

- **`llama-cpp-python is not installed`** → `pip install llama-cpp-python`
  (for GPU support, install with the appropriate build flags for your
  hardware — see the [llama-cpp-python docs](https://github.com/abetlen/llama-cpp-python)).
- **`transformers/torch are not installed`** → `pip install torch transformers accelerate`.
- **Model folder not recognized** → make sure the path is a directory
  containing `config.json`, tokenizer files, and `*.safetensors` weight
  files (not a path to an individual `.safetensors` file).
- **Low code scores despite a good model** → run with `--verbose --test=code`
  to see exactly what the model output and why grading failed; often it's
  the model wrapping code in extra prose without a fenced ` ```python `
  block, or slightly wrong function signatures/names.
- **Colors look broken on Windows** → add `--no-color`, or
  `pip install colorama` and modern Windows Terminal/PowerShell will
  usually render ANSI colors fine anyway.
