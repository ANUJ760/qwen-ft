from pathlib import Path
import argparse
import json
import os
import re
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

from schema import ExperimentReport
from common.prompts import build_gold_extraction_prompt


DEFAULT_MODEL = "gemini-3.6-flash"

FILENAME_RE = re.compile(
    r"^E(?P<experiment>\d+)"
    r"_"
    r"(?P<semester>[A-Za-z0-9]+)"
    r"(?P<division>[A-Za-z]+\d+)"
    r"_"
    r"(?P<roll>\d+)"
    r"(?P<suffix>[A-Za-z])?"
    r"\.(?P<extension>docx|pdf)$",
    re.IGNORECASE,
)


SYSTEM_PROMPT = """
You are the teacher model for a document-compilation dataset.

Your task is to inspect a practical/experiment document and convert its
actual content into a structured ExperimentReport JSON object.

You MUST preserve information from the source document.

You MUST NOT invent:
- student names
- roll numbers
- dates
- experiment numbers
- sections
- technical facts
- content that is not supported by the source

If a required textual field cannot be determined from the document, use
"<MISSING>".

The output MUST conform to the ExperimentReport schema supplied below.

TOP LEVEL:

{
  "title": "...",
  "particulars": {...},
  "sections": [...],
  "submission_meta": {...}
}

PARTICULARS:

{
  "case_study_title": "...",
  "aim": "...",
  "problem_statement": "...",
  "author": "...",
  "section": "...",
  "roll_number": "...",
  "date_of_compilation": "YYYY-MM-DD"
}

BODY SECTIONS:

Every section MUST have:

{
  "number": 2,
  "title": "...",
  "content": [...]
}

Use the key "number", NOT "section_number".

Sections must start at 2 and increase sequentially:

2, 3, 4, ...

CONTENT BLOCKS:

Paragraph:

{
  "type": "paragraph",
  "text": "..."
}

Bullet list:

{
  "type": "bullet_list",
  "items": [
    {
      "lead_in": null,
      "text": "..."
    }
  ]
}

Figure:

{
  "type": "figure",
  "image_ref": "<ASSET_PLACEHOLDER>",
  "caption": "...",
  "figure_number": 1
}

Table:

{
  "type": "table",
  "header": ["..."],
  "rows": [
    ["...", "..."]
  ]
}

For figures, use "<ASSET_PLACEHOLDER>" for image_ref.

For tables, preserve the actual table information from the source.

For bullet lists, do not create empty items.

SUBMISSION METADATA:

{
  "experiment_number": 1,
  "semester_prefix": "...",
  "division": "...",
  "roll_number": "...",
  "part_suffix": null
}

The submission metadata is supplied separately from the document filename.
Use the supplied metadata exactly.

IMPORTANT:

Return ONLY the JSON object.

Do not return:
- Markdown
- code fences
- explanations
- analysis
- <think> blocks
- text before the JSON
- text after the JSON
"""


def parse_filename(path: Path):

    match = FILENAME_RE.match(
        path.name
    )

    if not match:
        raise ValueError(
            f"Invalid filename: {path.name}\n"
            f"Expected format such as:\n"
            f"E02_5A2_33.pdf\n"
            f"E03_5A2_33.docx\n"
            f"E03_5A2_33a.pdf"
        )

    return {
        "experiment_number": int(
            match.group("experiment")
        ),
        "semester_prefix": match.group(
            "semester"
        ),
        "division": match.group(
            "division"
        ),
        "roll_number": match.group(
            "roll"
        ),
        "part_suffix": match.group(
            "suffix"
        ),
    }


def load_schema():
    schema = (
        ExperimentReport
        .model_json_schema()
    )

    return json.dumps(
        schema,
        indent=2,
    )


def build_prompt(metadata):

    schema = load_schema()

    return build_gold_extraction_prompt(
        metadata,
        schema,
    )

def output_path_for(metadata, output_dir):
    experiment = (
        metadata["experiment_number"]
    )

    suffix = (
        metadata["part_suffix"]
        or ""
    )

    return (
        output_dir
        / f"E{experiment:02d}{suffix}.json"
    )


def clean_json(text):

    text = text.strip()

    if "</think>" in text:

        text = text.split(
            "</think>",
            1,
        )[1].strip()

    if text.startswith(
        "```json"
    ):

        text = text[
            len("```json"):
        ].strip()

    elif text.startswith(
        "```"
    ):

        text = text[
            len("```"):
        ].strip()

    if text.endswith(
        "```"
    ):

        text = text[:-3].strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "Gemini did not return a JSON object."
        )

    return text[
        start:end + 1
    ]


def generate_gold(
    client,
    model_name,
    document_path,
    metadata,
):
    from google.genai import types

    prompt = build_prompt(
        metadata
    )

    print(
        f"  Sending {document_path.name} to Gemini..."
    )

    uploaded_file = client.files.upload(
        file=document_path,
    )

    try:

        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text=prompt
                        ),
                        types.Part.from_uri(
                            file_uri=uploaded_file.uri,
                            mime_type=uploaded_file.mime_type,
                        ),
                    ],
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
            ),
        )

        if not response.text:
            raise ValueError(
                "Gemini returned an empty response."
            )

        cleaned = clean_json(
            response.text
        )

        return json.loads(
            cleaned
        )

    finally:

        try:
            client.files.delete(
                name=uploaded_file.name
            )
        except Exception:
            pass


def validate_gold(data):

    report = (
        ExperimentReport
        .model_validate(data)
    )

    return report


def process_document(
    client,
    model_name,
    document_path,
    output_dir,
    force=False,
):

    print()
    print(
        "=" * 70
    )
    print(
        f"Processing: {document_path.name}"
    )
    print(
        "=" * 70
    )

    metadata = parse_filename(
        document_path
    )

    output_path = output_path_for(
        metadata,
        output_dir,
    )

    if output_path.exists() and not force:

        print(
            f"  Skipping existing gold JSON: {output_path}"
        )

        return "skipped"

    print(
        f"  Experiment : "
        f"E{metadata['experiment_number']:02d}"
    )

    print(
        f"  Semester   : "
        f"{metadata['semester_prefix']}"
    )

    print(
        f"  Division   : "
        f"{metadata['division']}"
    )

    print(
        f"  Roll       : "
        f"{metadata['roll_number']}"
    )

    print(
        f"  Part suffix: "
        f"{metadata['part_suffix']}"
    )

    data = generate_gold(
        client,
        model_name,
        document_path,
        metadata,
    )

    report = validate_gold(
        data
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report.model_dump(
                mode="json"
            ),
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"  ✓ Valid ExperimentReport"
    )

    print(
        f"  ✓ Saved: {output_path}"
    )

    return "generated"


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=ROOT / "examples",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "dataset" / "target_json",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate gold JSON even when the output file already exists.",
    )

    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help=(
            "Optional filename filters. Each value can be an exact filename "
            "or a substring such as E05."
        ),
    )

    args = parser.parse_args()

    examples_dir = (
        args.examples_dir
    )

    if not examples_dir.is_absolute():

        examples_dir = (
            ROOT / examples_dir
        )

    output_dir = (
        args.output_dir
    )

    if not output_dir.is_absolute():

        output_dir = (
            ROOT / output_dir
        )

    files = sorted(
        [
            *examples_dir.glob(
                "*.pdf"
            ),
            *examples_dir.glob(
                "*.PDF"
            ),
            *examples_dir.glob(
                "*.docx"
            ),
            *examples_dir.glob(
                "*.DOCX"
            ),
        ]
    )

    if not files:

        print(
            f"No PDF or DOCX files found in "
            f"{examples_dir}"
        )

        return

    if args.only:

        filters = args.only

        files = [
            path
            for path in files
            if any(
                item == path.name or item in path.name
                for item in filters
            )
        ]

        if not files:

            print(
                "No files matched --only filters."
            )

            return

    print(
        f"Found {len(files)} documents."
    )

    successful = []
    skipped = []
    failed = []
    client = None

    for document_path in files:

        try:
            metadata = parse_filename(
                document_path
            )

            needs_generation = (
                args.force
                or not output_path_for(
                    metadata,
                    output_dir,
                ).exists()
            )

            if needs_generation and client is None:

                from google import genai

                api_key = os.getenv(
                    "GEMINI_API_KEY"
                )

                if not api_key:

                    raise RuntimeError(
                        "GEMINI_API_KEY environment variable "
                        "is not set."
                    )

                client = genai.Client(
                    api_key=api_key
                )

            success = process_document(
                client,
                args.model,
                document_path,
                output_dir,
                force=args.force,
            )

            if success == "generated":
                successful.append(
                    document_path.name
                )
                time.sleep(1)

            elif success == "skipped":
                skipped.append(
                    document_path.name
                )

        except Exception as e:

            print(
                f"  ✗ ERROR: {e}"
            )

            failed.append(
                document_path.name
            )

    print()
    print(
        "=" * 70
    )
    print(
        "EXTRACTION SUMMARY"
    )
    print(
        "=" * 70
    )

    print(
        f"Total      : {len(files)}"
    )

    print(
        f"Successful : {len(successful)}"
    )

    print(
        f"Skipped    : {len(skipped)}"
    )

    print(
        f"Failed     : {len(failed)}"
    )

    if failed:

        print()
        print(
            "Failed files:"
        )

        for name in failed:

            print(
                f"  ✗ {name}"
            )


if __name__ == "__main__":
    main()
