from pathlib import Path
import argparse
import json
import sys

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel




ROOT = Path(__file__).resolve().parent.parent

BASE_MODEL = "Qwen/Qwen3-4B"

ADAPTER_DIR = (
    ROOT / "finetune" / "checkpoints" / "experiment-report"
)

MAX_SEQ_LENGTH = 4096




sys.path.insert(
    0,
    str(ROOT / "schema"),
)

from schema import ExperimentReport




SYSTEM_PROMPT = """
You convert a student's raw, unstructured experiment notes into a
structured JSON ExperimentReport.

The output MUST be a single valid JSON object conforming exactly to the
ExperimentReport schema.

The top-level object MUST contain exactly these conceptual fields:

{
  "title": "...",
  "particulars": {...},
  "sections": [...],
  "submission_meta": {...}
}

The "particulars" object MUST contain:

{
  "case_study_title": "...",
  "aim": "...",
  "problem_statement": "...",
  "author": "...",
  "section": "...",
  "roll_number": "...",
  "date_of_compilation": "YYYY-MM-DD"
}

The "sections" array contains Section objects.

Each Section MUST have:

{
  "number": 2,
  "title": "...",
  "content": [...]
}

IMPORTANT:
Use the key "number", NOT "section_number".

Sections MUST start at 2 and then increase sequentially:

2, 3, 4, 5, ...

The "content" array contains one or more content blocks.

Allowed content block types:

1. Paragraph:

{
  "type": "paragraph",
  "text": "..."
}

2. Bullet list:

{
  "type": "bullet_list",
  "items": [
    {
      "lead_in": null,
      "text": "..."
    }
  ]
}

3. Figure:

{
  "type": "figure",
  "image_ref": "<ASSET_PLACEHOLDER>",
  "caption": "..."
}

4. Table:

{
  "type": "table",
  "header": ["..."],
  "rows": [
    ["...", "..."]
  ]
}

The "submission_meta" object MUST contain:

{
  "experiment_number": 1,
  "semester_prefix": "...",
  "division": "...",
  "roll_number": "...",
  "part_suffix": null
}

Do NOT invent information.

If a required value cannot be determined from the input, use:

"<MISSING>"

instead of making up a value.

For example:

"author": "<MISSING>"

Do not invent:

- names
- roll numbers
- dates
- experiment numbers
- divisions
- semester values
- technical facts

Preserve factual information from the user's raw content.

For figure blocks, use:

"<ASSET_PLACEHOLDER>"

for image_ref unless an actual asset reference is explicitly available.

Output ONLY the JSON object.

Do NOT output:

- <think> blocks
- explanations
- Markdown
- Markdown code fences
- commentary
- text before the JSON
- text after the JSON
"""




def load_model():

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Loading LoRA adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_DIR,
    )

    model.eval()

    return model, tokenizer




def clean_generated_output(
    generated_text: str,
) -> str:

    generated_text = generated_text.strip()



    if "</think>" in generated_text:

        generated_text = generated_text.split(
            "</think>",
            1,
        )[1].strip()



    if generated_text.startswith("```json"):

        generated_text = generated_text[
            len("```json"):
        ].strip()

    elif generated_text.startswith("```"):

        generated_text = generated_text[
            len("```"):
        ].strip()

    if generated_text.endswith("```"):

        generated_text = generated_text[
            :-3
        ].strip()



    start = generated_text.find("{")

    end = generated_text.rfind("}")

    if start == -1 or end == -1 or end <= start:

        raise ValueError(
            "Model did not produce a JSON object.\n\n"
            f"Model output:\n{generated_text}"
        )

    generated_text = generated_text[
        start:end + 1
    ]

    return generated_text.strip()




def generate_report(
    model,
    tokenizer,
    raw_content: str,
):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": raw_content,
        },
    ]


    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,

            max_new_tokens=3000,

            do_sample=False,

            pad_token_id=tokenizer.pad_token_id,

            eos_token_id=tokenizer.eos_token_id,
        )

   
    generated_tokens = outputs[
        0
    ][
        inputs["input_ids"].shape[1]:
    ]

    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

  

    generated_text = clean_generated_output(
        generated_text
    )

    return generated_text




def validate_report(
    result: str,
):

    try:

        parsed = json.loads(result)

    except json.JSONDecodeError as e:

        raise ValueError(
            f"Model output is not valid JSON: {e}\n\n"
            f"Output:\n{result}"
        )

    try:

        report = ExperimentReport.model_validate(
            parsed
        )

    except Exception as e:

        raise ValueError(
            "Generated JSON does not conform to "
            "ExperimentReport schema.\n\n"
            f"Validation error:\n{e}\n\n"
            f"Generated JSON:\n"
            f"{json.dumps(parsed, indent=2, ensure_ascii=False)}"
        )

    return report




def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to raw-content text file.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path for generated JSON.",
    )

    args = parser.parse_args()



    input_path = Path(
        args.input
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    raw_content = input_path.read_text(
        encoding="utf-8"
    )



    model, tokenizer = load_model()

    print()
    print("=" * 60)
    print("Generating ExperimentReport")
    print("=" * 60)
    print()



    result = generate_report(
        model,
        tokenizer,
        raw_content,
    )

    print(result)



    print()
    print("=" * 60)
    print("Validating generated JSON")
    print("=" * 60)
    print()

    report = validate_report(
        result
    )

    print("✓ Valid JSON")
    print("✓ Valid ExperimentReport schema")

   

    if args.output:

        output_path = Path(
            args.output
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                report.model_dump(mode="json"),
                f,
                indent=2,
                ensure_ascii=False,
            )

        print()
        print(
            f"JSON saved to: {output_path}"
        )




if __name__ == "__main__":
    main()