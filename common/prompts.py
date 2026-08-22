from __future__ import annotations

import json


INFERENCE_SYSTEM_PROMPT = """
You convert a student's raw, unstructured experiment notes into a
structured JSON ExperimentReport.

The output MUST be a single valid JSON object conforming exactly to the
ExperimentReport schema.

The top-level object MUST contain:

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

Each section MUST have:

{
  "number": 2,
  "title": "...",
  "content": [...]
}

Use the key "number", NOT "section_number".

Sections MUST start at 2 and increase sequentially:

2, 3, 4, ...

Allowed content block types are:

- paragraph
- bullet_list
- figure
- table

The "submission_meta" object MUST contain:

{
  "experiment_number": 1,
  "semester_prefix": "...",
  "division": "...",
  "roll_number": "...",
  "part_suffix": null
}

Do NOT invent information.

If a required value cannot be determined from the input, use "<MISSING>"
instead of making up a value.

For figure blocks, use "<ASSET_PLACEHOLDER>" for image_ref unless an actual
asset reference is explicitly available.

Output ONLY the JSON object.

Do NOT output Markdown, code fences, explanations, <think> blocks, commentary,
or text before or after the JSON.
"""


TEACHER_SYSTEM_PROMPT = """
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


GOLD_EXTRACTION_SYSTEM_PROMPT = """
You are the teacher model for a document-compilation dataset.

Your task is to inspect a practical/experiment document and convert its
actual content into a structured ExperimentReport JSON object.

You MUST preserve information from the source document.

You MUST NOT invent student names, roll numbers, dates, experiment numbers,
sections, technical facts, or content unsupported by the source.

If a required textual field cannot be determined from the document, use
"<MISSING>".

The output MUST conform to the ExperimentReport schema supplied below.

Use the supplied submission metadata exactly.

Return ONLY the JSON object. Do not return Markdown, code fences,
explanations, analysis, <think> blocks, or text before or after the JSON.
"""


def build_gold_extraction_prompt(metadata: dict, schema_json: str) -> str:
    metadata_json = json.dumps(metadata, indent=2)

    return f"""
{GOLD_EXTRACTION_SYSTEM_PROMPT}

SUBMISSION METADATA:

{metadata_json}

JSON Schema:

{schema_json}

Generate the ExperimentReport for the attached document.
"""
