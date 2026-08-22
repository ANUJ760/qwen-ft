"""
Teacher-distillation dataset generator.

For each gold-standard ExperimentReport JSON (produced by
extract_gold_json.py from real, correctly-structured example documents),
this script asks a strong Gemini teacher LLM to reverse-generate several
plausible MESSY raw-content inputs — the kind of rough, unstructured notes
a student might actually type up before organizing them into the final
report.

Each (raw_content, gold_json) pair becomes one training example:
the fine-tuned model's job is to learn the raw_content -> gold_json
direction.

We deliberately do NOT ask the teacher to invent the target JSON. The
target JSON already exists (extracted from real, guideline-compliant
documents) and is strictly more trustworthy than anything a teacher model
would hallucinate.

The teacher's job is narrow:
paraphrase/de-structure fluent prose back into rough notes in several
different styles.

The Compilation Guidelines are injected into the SYSTEM prompt of every
training example. This is the same prompt template that will be used at
real inference time later, so guideline changes down the line mean editing
this template + regenerating the dataset rather than retraining from zero.

Usage:
    export GEMINI_API_KEY=...
    python generate_dataset.py --variants-per-doc 4

Test the full pipeline without using API credits:

    python generate_dataset.py --dry-run --variants-per-doc 2
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import random
import sys
import time


ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(ROOT),
)
sys.path.insert(
    0,
    str(ROOT / "schema"),
)

from schema import ExperimentReport  # noqa: E402
from common.prompts import INFERENCE_SYSTEM_PROMPT, TEACHER_SYSTEM_PROMPT  # noqa: E402




STYLE_PROMPTS = [
    (
        "telegraphic_fragments",
        "Write it as terse, disconnected bullet-point fragments and "
        "half-sentences — the kind of rushed notes someone jots down right "
        "after a lab session, with abbreviations and no connecting prose.",
    ),
    (
        "verbose_rambling",
        "Write it as an over-long, rambling stream-of-consciousness "
        "paragraph that circles back on itself, includes tangents and filler "
        "phrases, but still contains all the same underlying facts.",
    ),
    (
        "casual_informal",
        "Write it in a casual, informal tone as if texting a classmate a "
        "summary — contractions, mild slang, no section structure at all.",
    ),
    (
        "reasonably_organized",
        "Write it as a rough but reasonably organized draft — mostly prose, "
        "loosely grouped by topic, but missing formal headings, numbering, "
        "and polish.",
    ),
]




SYSTEM_PROMPT_FOR_TEACHER = """
You are helping build a training dataset.

You will be given the FINAL, correctly structured content of a student
lab-experiment report as JSON.

Your job is to reverse-engineer what the student's rough, unpolished RAW
NOTES might have looked like before they wrote the final report.

The raw notes should contain the same underlying facts and ideas, but in a
messier and less structured form.

Rules:

- Preserve every factual claim, technical detail, and named entity from
  the source JSON.
- Do not invent new facts.
- Do not drop important facts.
- Do NOT reproduce the final document's heading structure.
- Do NOT reproduce section numbering.
- Do NOT copy polished phrasing from the final document unnecessarily.
- Do NOT mention that this is a reconstruction.
- Do NOT reference the JSON.
- Do NOT create a polished report.
- Output ONLY the raw notes text.
- Do not output JSON.
- Do not output Markdown fences.
- Do not add a preamble or explanation.
"""



SYSTEM_PROMPT_FOR_INFERENCE = """
You convert a student's raw, unstructured experiment notes into a
structured JSON ExperimentReport.

The output MUST be a single valid JSON object conforming to the
ExperimentReport schema.

The JSON object must contain:

- title
- particulars
- sections
- submission_meta

The particulars object must contain:

- case_study_title
- aim
- problem_statement
- author
- section
- roll_number
- date_of_compilation

Body sections must:

- start at section number 2
- be numbered sequentially: 2, 3, 4, ...
- contain at least one content block

Allowed content block types are:

- paragraph
- bullet_list
- figure
- table

The submission_meta object must contain:

- experiment_number
- semester_prefix
- division
- roll_number
- part_suffix

For figure blocks, use:

"<ASSET_PLACEHOLDER>"

for image_ref unless an asset reference is explicitly available.

Output ONLY the JSON object.

Do not use Markdown code fences.
Do not provide explanations.
Do not provide text before or after the JSON.
"""

SYSTEM_PROMPT_FOR_TEACHER = TEACHER_SYSTEM_PROMPT
SYSTEM_PROMPT_FOR_INFERENCE = INFERENCE_SYSTEM_PROMPT




def build_teacher_user_prompt(
    gold: dict,
    style_instruction: str,
) -> str:
    """
    Convert the gold ExperimentReport JSON into the teacher's input.

    The teacher sees the correct final structure and reconstructs plausible
    messy notes from it.
    """

    content_summary = json.dumps(
        gold,
        indent=2,
        ensure_ascii=False,
    )

    return (
        "Final structured report content (JSON):\n"
        "```json\n"
        f"{content_summary}\n"
        "```\n\n"
        f"Style instruction: {style_instruction}\n\n"
        "Now write the student's rough raw notes."
    )




def call_teacher(
    client,
    model: str,
    gold: dict,
    style_instruction: str,
    max_retries: int = 3,
) -> str:
    """
    Ask Gemini to generate one messy raw-content version.

    The teacher only generates the INPUT side of the training pair.
    The gold JSON remains the trusted TARGET.
    """

    prompt = build_teacher_user_prompt(
        gold,
        style_instruction,
    )

    last_err = None

    for attempt in range(max_retries):

        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "system_instruction": SYSTEM_PROMPT_FOR_TEACHER,
                    "temperature": 0.8,
                    "max_output_tokens": 1500,
                },
            )

            if not response.text:
                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            return response.text.strip()

        except Exception as e:
            last_err = e

            print(
                f"Teacher attempt {attempt + 1}/{max_retries} failed: {e}",
                file=sys.stderr,
            )

            time.sleep(2 ** attempt)

    raise RuntimeError(
        f"Gemini teacher call failed after "
        f"{max_retries} attempts: {last_err}"
    )




def mock_teacher(
    gold: dict,
    style_instruction: str,
) -> str:
    """
    Deterministic stand-in for --dry-run.

    Exercises file writing and JSONL assembly without calling Gemini.
    """

    fragments = []

    fragments.append(
        f"[DRY-RUN MOCK, style={style_instruction[:30]}...]"
    )

    fragments.append(
        gold["particulars"]["aim"]
    )

    for section in gold["sections"]:

        for block in section["content"]:

            if block["type"] == "paragraph":

                fragments.append(
                    block["text"][:60] + "..."
                )

            elif block["type"] == "bullet_list":

                for item in block["items"]:

                    lead_in = item.get("lead_in", "")

                    fragments.append(
                        f"- {lead_in}: "
                        f"{item['text'][:40]}..."
                    )

    tail = fragments[1:]
    random.shuffle(tail)
    fragments[1:] = tail

    return "\n".join(fragments)




def strip_asset_paths_for_target(
    gold: dict,
) -> dict:
    """
    Remove machine-specific image paths from the training target.

    The model cannot know the actual local path at inference time, so figure
    references are represented using <ASSET_PLACEHOLDER>.
    """

    gold = json.loads(
        json.dumps(gold)
    )

    for section in gold["sections"]:

        for block in section["content"]:

            if block.get("type") == "figure":

                block["image_ref"] = "<ASSET_PLACEHOLDER>"

    return gold




def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target-json-dir",
        type=Path,
        default=ROOT / "dataset" / "target_json",
        help="Directory containing validated gold ExperimentReport JSON files.",
    )

    parser.add_argument(
        "--raw-content-out",
        type=Path,
        default=ROOT / "dataset" / "raw_content",
        help="Directory where generated raw-content files are stored.",
    )

    parser.add_argument(
        "--jsonl-out",
        type=Path,
        default=ROOT / "dataset" / "training_pairs.jsonl",
        help="Output JSONL file for fine-tuning.",
    )

    parser.add_argument(
        "--variants-per-doc",
        type=int,
        default=4,
        help=(
            f"Number of synthetic raw-content variants per document. "
            f"Maximum available styles: {len(STYLE_PROMPTS)}."
        ),
    )

    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model used as the teacher.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Skip Gemini API calls and use the deterministic mock teacher."
        ),
    )

    parser.add_argument(
        "--force-raw",
        action="store_true",
        help=(
            "Regenerate raw-content variants even when matching files "
            "already exist."
        ),
    )

    args = parser.parse_args()

   
    args.variants_per_doc = min(
        args.variants_per_doc,
        len(STYLE_PROMPTS),
    )


    target_json_dir = args.target_json_dir
    raw_content_out = args.raw_content_out
    jsonl_out = args.jsonl_out

    if not target_json_dir.is_absolute():
        target_json_dir = ROOT / target_json_dir

    if not raw_content_out.is_absolute():
        raw_content_out = ROOT / raw_content_out

    if not jsonl_out.is_absolute():
        jsonl_out = ROOT / jsonl_out

    gold_paths = sorted(
        glob.glob(
            str(target_json_dir / "*.json")
        )
    )

    if not gold_paths:

        print(
            f"No gold JSON files found in "
            f"{target_json_dir}/. "
            f"Run extract_gold_json.py first.",
            file=sys.stderr,
        )

        sys.exit(1)


    client = None

    if not args.dry_run:

        from google import genai

        if not os.environ.get("GEMINI_API_KEY"):

            print(
                "GEMINI_API_KEY not set. "
                "Set it before running without --dry-run.",
                file=sys.stderr,
            )

            sys.exit(1)

        client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"]
        )

 

    raw_content_out.mkdir(
        parents=True,
        exist_ok=True,
    )

    jsonl_out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    n_written = 0

    with jsonl_out.open(
        "w",
        encoding="utf-8",
    ) as jsonl_file:

        for gold_path in gold_paths:

          

            with open(
                gold_path,
                "r",
                encoding="utf-8",
            ) as f:

                gold = json.load(f)

 
            ExperimentReport.model_validate(
                gold
            )

  

            doc_id = os.path.splitext(
                os.path.basename(gold_path)
            )[0]

      

            target_gold = strip_asset_paths_for_target(
                gold
            )



            styles = random.sample(
                STYLE_PROMPTS,
                args.variants_per_doc,
            )

        

            for style_name, style_instruction in styles:

                raw_path = (
                    raw_content_out
                    / f"{doc_id}__{style_name}.txt"
                )

                if raw_path.exists() and not args.force_raw:

                    raw_content = raw_path.read_text(
                        encoding="utf-8"
                    )

                    print(
                        f"  {doc_id} "
                        f"[{style_name}] "
                        f"reused -> {raw_path}"
                    )

                else:

                    if args.dry_run:

                        raw_content = mock_teacher(
                            gold,
                            style_instruction,
                        )

                    else:

                        raw_content = call_teacher(
                            client,
                            args.model,
                            gold,
                            style_instruction,
                        )

                    raw_path.write_text(
                        raw_content,
                        encoding="utf-8",
                    )

                    print(
                        f"  {doc_id} "
                        f"[{style_name}] "
                        f"-> {raw_path}"
                    )


                training_example = {
                    "messages": [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT_FOR_INFERENCE,
                        },
                        {
                            "role": "user",
                            "content": raw_content,
                        },
                        {
                            "role": "assistant",
                            "content": json.dumps(
                                target_gold,
                                ensure_ascii=False,
                            ),
                        },
                    ],
                    "_meta": {
                        "source_doc": doc_id,
                        "style": style_name,
                    },
                }

             

                jsonl_file.write(
                    json.dumps(
                        training_example,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                n_written += 1

    print(
        f"\nDone. {n_written} training pairs written "
        f"to {jsonl_out}"
    )




if __name__ == "__main__":
    main()
