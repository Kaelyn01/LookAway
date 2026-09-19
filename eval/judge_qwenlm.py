import argparse
import json
import os
import re
import sys
import threading
import time

from tqdm import tqdm

MCQ_BENCHMARKS = [
    "hrbench-4k",
    "hrbench-8k",
    "vstar",
    "mme-realworld",
    "mme-realworld-cn",
    "mme-realworld-lite",
    "mmstar",
    "cv-bench",
]

POPE_BENCHMARKS = [
    "pope",
    "pope_adv",
    "pope_pop",
    "pope_random",
]

MMVP_BENCHMARKS = [
    "mmvp",
]

PROMPT_TEMPLATE = (
    "Your task is to judge whether the response expresses the same meaning "
    "as the answer of a question.\n"
    "The question is: {question}\n"
    "The answer is: {gt}\n"
    "The response is: {response}\n"
    "Please check and compare them and then judge. "
    "If the response is correct, your output should be Yes. "
    "Otherwise, your output should be No. Directly give me your output."
)


def extract_first_option(text):
    if not text:
        return ""
    match = re.search(r"\(([A-Z])\)", text)
    if match:
        return match.group(1)
    match = re.search(
        r"(?:^|\b(?:answer|option|choice)\s*(?:is\s+)?[:\-]?\s*)([A-F])(?:$|[.\)\]\s])",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()
    return ""


def extract_mcq_option(answer):
    if not isinstance(answer, str) or not answer:
        return ""
    text = answer.strip()
    pattern = r"^[ (\[]*([A-F])(?:(?=$)|[\.\)\]]|(?:[\:\-]\s+))"
    match = re.match(pattern, text)
    if match:
        return match.group(1)
    return ""


def pope_extract(text):
    if not isinstance(text, str) or not text:
        return ""
    t = text.lstrip("*").strip()
    if t.lower().startswith("answer"):
        t = t.split("answer", 1)[1].lstrip(":").lstrip("*").strip()
    t_lower = t.lower()
    if t_lower.startswith("yes"):
        return "yes"
    if t_lower.startswith("no"):
        return "no"
    last = t.rstrip(".").rstrip("*").strip().split()[-1] if t.strip() else ""
    last = last.lstrip("*").rstrip("*").lower()
    if last in ("yes", "no"):
        return last
    return text.strip()


def mmvp_extract(text):
    if not isinstance(text, str) or not text:
        return ""
    t = text.strip().lower()
    match = re.search(r"\(([ab])\)", t)
    if match:
        return f"({match.group(1)})"
    match = re.search(r"\b([ab])\b", t)
    if match:
        return f"({match.group(1)})"
    return ""


def first_letter_match(gt, answer):
    gt_val = extract_mcq_option(gt)
    pred_val = extract_first_option(answer)
    return bool(gt_val and pred_val and gt_val == pred_val)


def extract_answer(model_answer_raw):
    text = model_answer_raw.strip()
    think_end = text.rfind("</think>")
    if think_end != -1:
        text = text[think_end + len("</think>") :].strip()
    start = text.rfind("<answer>")
    end = text.find("</answer>", start + len("<answer>")) if start != -1 else -1
    if start != -1 and end != -1 and end > start:
        return text[start + len("<answer>") : end].strip()
    answer_match = re.search(r"(?i)answer\s*:\s*(.*)", text, flags=re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    return text


def judge_via_api(prompts, api_base, api_key, judge_model, judge_max_tokens, parallel_workers=32):
    from openai import OpenAI

    thread_local = threading.local()

    def get_client():
        c = getattr(thread_local, "client", None)
        if c is None:
            c = OpenAI(api_key=api_key, base_url=api_base, timeout=600)
            thread_local.client = c
        return c

    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = [""] * len(prompts)

    def call_one(idx, prompt):
        client = get_client()
        last_error = None
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=judge_max_tokens,
                )
                return idx, (resp.choices[0].message.content or "").strip()
            except Exception as exc:
                last_error = type(exc).__name__
                if attempt < 2:
                    time.sleep(1.0)
        return idx, f"[JUDGE_ERROR] {last_error}"

    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        futures = [executor.submit(call_one, i, p) for i, p in enumerate(prompts)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="LLM Judge"):
            idx, text = future.result()
            results[idx] = text

    return results


def judge_via_vllm(prompts, judge_model_path, judge_max_tokens):
    import torch._dynamo

    torch._dynamo.config.suppress_errors = True

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    sampling_params = SamplingParams(max_tokens=judge_max_tokens, temperature=0)
    llm = LLM(model=judge_model_path, tensor_parallel_size=1, gpu_memory_utilization=0.9)
    tokenizer = AutoTokenizer.from_pretrained(judge_model_path)

    chat_prompts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        chat_prompts.append(text)

    outputs = llm.generate(chat_prompts, sampling_params)
    return [output.outputs[0].text.strip() for output in outputs]


def main():
    parser = argparse.ArgumentParser(description="LLM-based judge for benchmark evaluation")
    parser.add_argument("--benchmark", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--judge_model_path", default=None, type=str, help="Local model path for vLLM-based judging")
    parser.add_argument("--api_base", default=None, type=str, help="OpenAI-compatible API base URL for judging")
    parser.add_argument("--api_key", default=None, type=str, help=argparse.SUPPRESS)
    parser.add_argument("--judge_model", default=None, type=str, help="Model name for API-based judging")
    parser.add_argument("--judge_max_tokens", default=2048, type=int)
    parser.add_argument("--answer_dir", default="model_answer", type=str)
    args = parser.parse_args()
    api_key = args.api_key or os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY", "EMPTY")

    if not args.api_base and not args.judge_model_path:
        print(
            "ERROR: Either --api_base (for API-based judging) or --judge_model_path "
            "(for local vLLM judging) must be provided.",
            file=sys.stderr,
        )
        sys.exit(1)

    answer_path = os.path.join(args.answer_dir, args.benchmark, f"{args.model}_answer.jsonl")
    save_path = f"judge/{args.benchmark}/{args.model}_answer.jsonl"
    os.makedirs(f"judge/{args.benchmark}", exist_ok=True)
    is_mcq = args.benchmark in MCQ_BENCHMARKS
    is_pope = args.benchmark in POPE_BENCHMARKS
    is_mmvp = args.benchmark in MMVP_BENCHMARKS

    data_list = []
    with open(answer_path, encoding="utf-8") as f:
        for line in f:
            data_list.append(json.loads(line))

    try:
        from mathruler.grader import grade_answer

        has_mathruler = True
    except ImportError:
        has_mathruler = False

    to_llm_indices = []
    prompt_lists = []

    for i, item in enumerate(tqdm(data_list, desc="Rule-based grading")):
        question = item["query"].replace("<image>", "")
        model_answer_raw = item["model_answer"]
        extracted_answer = extract_answer(model_answer_raw)
        gt = item["response"]
        item["extracted_answer"] = extracted_answer

        is_correct = False

        if is_pope:
            pred = pope_extract(extracted_answer)
            if pred == gt.strip().lower():
                is_correct = True
                item["judge"] = "Yes"
                item["judge_source"] = "pope_exact"

        if not is_correct and is_mmvp:
            pred = mmvp_extract(extracted_answer)
            gt_norm = gt.strip().lower()
            if pred and pred == gt_norm:
                is_correct = True
                item["judge"] = "Yes"
                item["judge_source"] = "mmvp_option"

        if not is_correct and has_mathruler:
            try:
                is_correct = grade_answer(gt, extracted_answer)
            except Exception:
                is_correct = False

        is_letter_correct = False
        if not is_correct and is_mcq:
            try:
                is_letter_correct = first_letter_match(gt, extracted_answer)
            except Exception:
                is_letter_correct = False

        if is_correct and "judge" not in item:
            item["judge"] = "Yes"
            item["judge_source"] = "mathruler"
        elif is_letter_correct:
            item["judge"] = "Yes"
            item["judge_source"] = "first letter"
        else:
            prompt = PROMPT_TEMPLATE.format(gt=gt, response=extracted_answer, question=question)
            to_llm_indices.append(i)
            prompt_lists.append(prompt)

    if prompt_lists:
        print(f"Calling LLM judge for {len(prompt_lists)} remaining cases...")
        if args.api_base:
            judge_model_name = args.judge_model or "default"
            results = judge_via_api(
                prompt_lists, args.api_base, api_key, judge_model_name, args.judge_max_tokens
            )
        else:
            results = judge_via_vllm(prompt_lists, args.judge_model_path, args.judge_max_tokens)

        for idx_in_llm, response_text in enumerate(results):
            original_idx = to_llm_indices[idx_in_llm]
            data_list[original_idx]["judge"] = response_text
            data_list[original_idx]["judge_source"] = "llm"

    print(f"Total: {len(data_list)}, LLM used: {len(prompt_lists)}")

    with open(save_path, "w", encoding="utf-8") as out_file:
        json.dump(data_list, out_file, ensure_ascii=False, indent=4)
    print(f"Saved judge results to: {save_path}")
    failed = [item for item in data_list if str(item.get("judge", "")).startswith("[JUDGE_ERROR]")]
    if failed:
        raise RuntimeError(f"LLM judge failed for {len(failed)} cases; results were saved for inspection and retry")


if __name__ == "__main__":
    main()
