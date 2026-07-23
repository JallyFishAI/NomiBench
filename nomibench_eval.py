"""
nomibench_eval.py
Grading logic for NomiBench. Every function here takes the raw text a model
produced and returns (bool correct, str detail_message).

All grading is fully automatic / deterministic — no second model is used as
a judge, so results are 100% reproducible across runs and across machines.
"""

import re
import json
import subprocess
import sys


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _normalize(text):
    """Lowercase, strip punctuation (keep slashes for fractions), collapse
    whitespace. Used for loose exact-match comparisons."""
    text = text.strip().lower()
    text = re.sub(r"[^\w\s/]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _words(text):
    return re.findall(r"[A-Za-z0-9']+", text)


NUM_PATTERN = re.compile(r"-?\d[\d,]*\.?\d*")


def _extract_numbers(text):
    nums = []
    for m in NUM_PATTERN.findall(text):
        cleaned = m.replace(",", "")
        if cleaned in ("", "-", "."):
            continue
        try:
            nums.append(float(cleaned))
        except ValueError:
            continue
    return nums


def _within_tol(candidate, answer, tolerance):
    if answer == 0:
        return abs(candidate) < 1e-6
    allowed = max(abs(answer) * tolerance, 0.5)
    return abs(candidate - answer) <= allowed


# --------------------------------------------------------------------------
# Basic type evaluators
# --------------------------------------------------------------------------

def eval_exact(response, answer):
    norm_resp = _normalize(response)
    norm_ans = _normalize(answer)
    tokens = norm_resp.split()
    correct = (norm_ans == norm_resp) or (norm_ans in tokens) or (
        len(norm_ans) > 2 and norm_ans in norm_resp
    )
    return correct, f"expected '{answer}'"


def eval_numeric(response, answer, tolerance=0.02):
    answer = float(answer)
    m = re.search(r"answer[^0-9\-]{0,15}(-?\d[\d,]*\.?\d*)", response, re.IGNORECASE)
    if m:
        try:
            candidate = float(m.group(1).replace(",", ""))
            if _within_tol(candidate, answer, tolerance):
                return True, f"expected {answer}, got {candidate}"
        except ValueError:
            pass
    nums = _extract_numbers(response)
    for n in reversed(nums):
        if _within_tol(n, answer, tolerance):
            return True, f"expected {answer}, found {n} in response"
    got = nums[-1] if nums else None
    return False, f"expected {answer}, got {got}"


def eval_multiple_choice(response, answer):
    answer = answer.strip().upper()
    m = re.search(r"answer[^A-Da-d]{0,10}([A-D])\b", response, re.IGNORECASE)
    if m:
        candidate = m.group(1).upper()
        return candidate == answer, f"expected {answer}, got {candidate}"
    letters = re.findall(r"\b([A-D])\b", response.upper())
    if letters:
        candidate = letters[-1]
        return candidate == answer, f"expected {answer}, got {candidate} (inferred)"
    return False, f"expected {answer}, no option letter found"


def eval_single_letter(response, answer):
    answer = answer.strip().upper()
    m = re.search(r"answer[^A-Za-z]{0,10}([A-Za-z])\b", response, re.IGNORECASE)
    if m:
        candidate = m.group(1).upper()
        return candidate == answer, f"expected {answer}, got {candidate}"
    letters = re.findall(r"\b([A-Za-z])\b", response)
    if letters:
        candidate = letters[-1].upper()
        return candidate == answer, f"expected {answer}, got {candidate} (inferred)"
    return False, f"expected {answer}, no single letter found"


def eval_yes_no(response, answer):
    answer = answer.strip().lower()
    m = re.search(r"answer[^A-Za-z]{0,10}(yes|no)\b", response, re.IGNORECASE)
    if m:
        candidate = m.group(1).lower()
        return candidate == answer, f"expected {answer}, got {candidate}"
    matches = re.findall(r"\b(yes|no)\b", response, re.IGNORECASE)
    if matches:
        candidate = matches[-1].lower()
        return candidate == answer, f"expected {answer}, got {candidate} (inferred)"
    return False, f"expected {answer}, no yes/no found"


def eval_contains_any(response, keywords):
    norm = response.lower()
    found = [k for k in keywords if k.lower() in norm]
    return (len(found) > 0), f"expected any of {keywords}, found {found}"


def eval_contains_all(response, keywords):
    norm = response.lower()
    missing = [k for k in keywords if k.lower() not in norm]
    return (len(missing) == 0), f"expected all of {keywords}, missing {missing}"


# --------------------------------------------------------------------------
# Code evaluator — actually executes the candidate function in a subprocess
# --------------------------------------------------------------------------

def _extract_code(response):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1)
    return response


def eval_code(response, question):
    code = _extract_code(response)
    func_name = question["function_name"]
    tests = question["tests"]

    harness = code + "\n\n"
    harness += "import json as _json\n"
    harness += f"_tests = {tests!r}\n"
    harness += "_results = []\n"
    harness += "for _tc in _tests:\n"
    harness += "    try:\n"
    harness += f"        _res = {func_name}(*_tc['args'])\n"
    harness += "        _results.append(_res == _tc['expected'])\n"
    harness += "    except Exception:\n"
    harness += "        _results.append(False)\n"
    harness += "print('NOMIBENCH_RESULT:' + _json.dumps(_results))\n"

    try:
        proc = subprocess.run(
            [sys.executable, "-c", harness],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return False, "code execution timed out (>10s)"

    m = re.search(r"NOMIBENCH_RESULT:(\[.*\])", proc.stdout)
    if not m:
        err = (proc.stderr or "no output").strip().splitlines()[-1:] or [""]
        return False, f"execution failed: {err[0][:150]}"

    try:
        results_list = json.loads(m.group(1))
    except json.JSONDecodeError:
        return False, "could not parse test results"

    passed = sum(1 for r in results_list if r)
    total = len(results_list)
    return (total > 0 and passed == total), f"{passed}/{total} test cases passed"


# --------------------------------------------------------------------------
# Instruction-following ("format_check") evaluators
# --------------------------------------------------------------------------

def check_exact_match(response, params):
    return _normalize(response) == _normalize(params["target"])


def check_word_count_exact(response, params):
    return len(_words(response)) == params["n"]


def check_all_caps(response, params):
    letters = [c for c in response if c.isalpha()]
    return len(letters) > 0 and all(c.isupper() for c in letters)


def check_no_letter(response, params):
    return params["letter"].lower() not in response.lower()


def check_valid_json(response, params):
    m = re.search(r"\{.*\}", response, re.DOTALL)
    text = m.group(0) if m else response
    try:
        data = json.loads(text)
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    expected = params.get("expected", {})
    for k, v in expected.items():
        if str(data.get(k, "")).strip().lower() != str(v).strip().lower():
            return False
    return True


def check_starts_with(response, params):
    return response.strip().lower().startswith(params["prefix"].lower())


def check_ends_with(response, params):
    cleaned = response.strip().rstrip(".!? \n\t")
    return cleaned.lower().endswith(params["suffix"].lower())


def check_sentence_count(response, params):
    sentences = [s for s in re.split(r"[.!?]+", response.strip()) if s.strip()]
    return len(sentences) == params["n"]


def check_line_count(response, params):
    lines = [l for l in response.strip().splitlines() if l.strip()]
    return len(lines) == params["n"]


def check_numbered_list(response, params):
    n = params["n"]
    lines = [l.strip() for l in response.strip().splitlines() if l.strip()]
    if len(lines) != n:
        return False
    for i, line in enumerate(lines, 1):
        if not re.match(rf"^{i}[.)]", line):
            return False
    return True


def check_single_word_contains(response, params):
    words = _words(response)
    return len(words) <= 2 and params["word"].lower() in response.lower()


def check_yes_no(response, params):
    expected = params["expected"].lower()
    norm = _normalize(response)
    return norm == expected or norm.startswith(expected)


FORMAT_CHECKS = {
    "exact_match": check_exact_match,
    "word_count_exact": check_word_count_exact,
    "all_caps": check_all_caps,
    "no_letter": check_no_letter,
    "valid_json": check_valid_json,
    "starts_with": check_starts_with,
    "ends_with": check_ends_with,
    "sentence_count": check_sentence_count,
    "line_count": check_line_count,
    "numbered_list": check_numbered_list,
    "single_word_contains": check_single_word_contains,
    "yes_no": check_yes_no,
}


def eval_format(response, question):
    check_name = question["check"]
    params = question.get("params", {})
    fn = FORMAT_CHECKS[check_name]
    ok = fn(response, params)
    return ok, f"check '{check_name}' {'passed' if ok else 'failed'}"


# --------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------

def evaluate(question, response):
    qtype = question["type"]
    if qtype == "exact":
        return eval_exact(response, question["answer"])
    if qtype == "numeric":
        return eval_numeric(response, question["answer"], question.get("tolerance", 0.02))
    if qtype == "multiple_choice":
        return eval_multiple_choice(response, question["answer"])
    if qtype == "single_letter":
        return eval_single_letter(response, question["answer"])
    if qtype == "yes_no":
        return eval_yes_no(response, question["answer"])
    if qtype == "contains_any":
        return eval_contains_any(response, question["answer"])
    if qtype == "contains_all":
        return eval_contains_all(response, question["answer"])
    if qtype == "code":
        return eval_code(response, question)
    if qtype == "format_check":
        return eval_format(response, question)
    raise ValueError(f"Unknown question type: {qtype}")
