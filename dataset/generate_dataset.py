"""
Teacher-distillation dataset generator.

For each gold-standard ExperimentReport JSON (produced by
extract_gold_json.py from real, correctly-structured example documents),
this script asks a strong "teacher" LLM to reverse-generate several
plausible MESSY raw-content inputs — the kind of rough, unstructured notes
a student might actually type up before organizing them into the final
report. Each (raw_content, gold_json) pair becomes one training example:
the fine-tuned model's job is to learn the raw_content -> gold_json
direction.

We deliberately do NOT ask the teacher to invent the target JSON. The
target JSON already exists (extracted from real, guideline-compliant
documents) and is strictly more trustworthy than anything a teacher model
would hallucinate. The teacher's job here is narrow and well-suited to an
LLM: paraphrase/de-structure fluent prose back into rough notes, in several
different styles, which is a much easier and more reliable generative task
than "invent a correct structured document from scratch."

The Compilation Guidelines are injected into the SYSTEM prompt of every
training example (not baked into weights only) — this is the same prompt
template that will be used at real inference time later, so the model
practices the exact conditioning it will see in production, and guideline
changes down the line just mean editing this template + regenerating,
not retraining from zero.

Usage:
    export ANTHROPIC_API_KEY=sk-...
    python generate_dataset.py --variants-per-doc 4

    # Test the full pipeline (file I/O, retries, jsonl assembly) without
    # spending API credits:
    python generate_dataset.py --dry-run --variants-per-doc 2
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "schema"))
from schema import ExperimentReport  # noqa: E402


STYLE_PROMPTS = [
    (
        "telegraphic_fragments",
        "Write it as terse, disconnected bullet-point fragments and half-sentences "
        "— the kind of rushed notes someone jots down right after a lab session, "
        "with abbreviations and no connecting prose.",
    ),
    (
        "verbose_rambling",
        "Write it as an over-long, rambling stream-of-consciousness paragraph that "
        "circles back on itself, includes tangents and filler phrases, but still "
        "contains all the same underlying facts.",
    ),
    (
        "casual_informal",
        "Write it in a casual, informal tone as if texting a classmate a summary — "
        "contractions, mild slang, no section structure at all.",
    ),
    (
        "reasonably_organized",
        "Write it as a rough but reasonably organized draft — mostly prose, loosely "
        "grouped by topic, but missing formal headings, numbering, and polish.",
    ),
]

SYSTEM_PROMPT_FOR_TEACHER = """You are helping build a training dataset. You will be given the FINAL, \
correctly structured content of a student lab-experiment report (as JSON). \
Your job is to reverse-engineer what the student's rough, unpolished RAW \
NOTES might have looked like before they wrote up this final version — \
notes that contain the same underlying facts and ideas, but in a messier, \
unstructured form.

Rules:
- Preserve every factual claim, technical detail, and named entity from the source JSON. Do not invent new facts, and do not drop any.
- Do NOT reproduce the final document's heading structure, numbering, or polished phrasing — that's exactly what needs to be missing from raw notes.
- Do NOT mention that this is a reconstruction or reference the JSON.
- Output ONLY the raw notes text, nothing else — no preamble, no markdown fences.
"""


def build_teacher_user_prompt(gold: dict, style_instruction: str) -> str:
    # We intentionally exclude image_ref paths and figure_number bookkeeping
    # fields from what we show the teacher — those are formatter-internal
    # details irrelevant to "what would a student's raw notes say".
    content_summary = json.dumps(gold, indent=2)
    return (
        f"Final structured report content (JSON):\n```json\n{content_summary}\n```\n\n"
        f"Style instruction: {style_instruction}\n\n"
        f"Now write the student's rough raw notes."
    )


def call_teacher(client, model: str, gold: dict, style_instruction: str, max_retries: int = 3) -> str:
    prompt = build_teacher_user_prompt(gold, style_instruction)
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=1500,
                system=SYSTEM_PROMPT_FOR_TEACHER,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(
                block.text for block in resp.content if block.type == "text"
            ).strip()
        except Exception as e:  # noqa: BLE001 - broad on purpose, this is a retry loop
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Teacher call failed after {max_retries} attempts: {last_err}")


def mock_teacher(gold: dict, style_instruction: str) -> str:
    """Deterministic stand-in for --dry-run: exercises every downstream code
    path (file writes, jsonl assembly) without calling any API."""
    fragments = []
    fragments.append(f"[DRY-RUN MOCK, style={style_instruction[:30]}...]")
    fragments.append(gold["particulars"]["aim"])
    for s in gold["sections"]:
        for block in s["content"]:
            if block["type"] == "paragraph":
                fragments.append(block["text"][:60] + "...")
            elif block["type"] == "bullet_list":
                for item in block["items"]:
                    fragments.append(f"- {item.get('lead_in','')}: {item['text'][:40]}...")
    random.shuffle(fragments[1:])  # scramble order like real rough notes would be
    return "\n".join(fragments)


def strip_asset_paths_for_target(gold: dict) -> dict:
    """The gold JSON we train the model to PRODUCE should not contain
    machine-local filesystem paths (image_ref) — those are specific to
    wherever extraction happened to run and won't exist at inference time.
    Replace with a placeholder token the formatter/inference pipeline is
    expected to fill in separately."""
    gold = json.loads(json.dumps(gold))  # deep copy
    for s in gold["sections"]:
        for block in s["content"]:
            if block.get("type") == "figure":
                block["image_ref"] = "<ASSET_PLACEHOLDER>"
    return gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-json-dir", default="target_json")
    ap.add_argument("--raw-content-out", default="raw_content")
    ap.add_argument("--jsonl-out", default="training_pairs.jsonl")
    ap.add_argument("--variants-per-doc", type=int, default=4,
                     help=f"Max {len(STYLE_PROMPTS)} distinct styles available.")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--dry-run", action="store_true",
                     help="Skip real API calls; use a deterministic mock teacher instead.")
    args = ap.parse_args()

    args.variants_per_doc = min(args.variants_per_doc, len(STYLE_PROMPTS))

    gold_paths = sorted(glob.glob(os.path.join(args.target_json_dir, "*.json")))
    if not gold_paths:
        print(f"No gold JSON files found in {args.target_json_dir}/. "
              f"Run extract_gold_json.py first.", file=sys.stderr)
        sys.exit(1)

    client = None
    if not args.dry_run:
        import anthropic
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ANTHROPIC_API_KEY not set. Use --dry-run to test the pipeline "
                  "without an API key.", file=sys.stderr)
            sys.exit(1)
        client = anthropic.Anthropic()

    os.makedirs(args.raw_content_out, exist_ok=True)

    n_written = 0
    with open(args.jsonl_out, "w") as jsonl_f:
        for gold_path in gold_paths:
            with open(gold_path) as f:
                gold = json.load(f)

            # Validate the gold example itself before using it as a training
            # target — never train on something that wouldn't pass the same
            # gate production inference output has to pass.
            ExperimentReport.model_validate(gold)

            doc_id = os.path.splitext(os.path.basename(gold_path))[0]
            target_gold = strip_asset_paths_for_target(gold)

            styles = random.sample(STYLE_PROMPTS, args.variants_per_doc)
            for style_name, style_instruction in styles:
                if args.dry_run:
                    raw_content = mock_teacher(gold, style_instruction)
                else:
                    raw_content = call_teacher(client, args.model, gold, style_instruction)

                raw_path = os.path.join(args.raw_content_out, f"{doc_id}__{style_name}.txt")
                with open(raw_path, "w") as f:
                    f.write(raw_content)

                training_example = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT_FOR_INFERENCE_PLACEHOLDER},
                        {"role": "user", "content": raw_content},
                        {"role": "assistant", "content": json.dumps(target_gold)},
                    ],
                    "_meta": {"source_doc": doc_id, "style": style_name},
                }
                jsonl_f.write(json.dumps(training_example) + "\n")
                n_written += 1
                print(f"  {doc_id} [{style_name}] -> {raw_path}")

    print(f"\nDone. {n_written} training pairs written to {args.jsonl_out}")


# NOTE: this references a system prompt that must exactly match what
# inference/generate_report.py uses at real inference time (RAG/guideline
# injection is only effective if the model is trained on the SAME framing
# it will see in production). Defined in a shared module once
# inference/generate_report.py exists; using a local placeholder for now
# so this script is independently runnable and testable today.
SYSTEM_PROMPT_FOR_INFERENCE_PLACEHOLDER = """You convert a student's raw, unstructured experiment notes into a \
structured JSON report. Output must be a single JSON object conforming \
exactly to the ExperimentReport schema (particulars + numbered sections \
2,3,4... each containing paragraph/bullet_list/figure/table content blocks). \
Every mandatory particulars field must be present: case_study_title, aim, \
problem_statement, author, section, roll_number, date_of_compilation. \
Output ONLY the JSON object, no other text."""


if __name__ == "__main__":
    main()
