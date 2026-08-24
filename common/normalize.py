"""Deterministic post-processing shared by gold extraction and inference.

Both the Gemini teacher (dataset/extract_gold_json.py) and the fine-tuned
student model (inference/generate_report.py) emit JSON that is *almost*
schema-conforming. Rather than re-prompting (extra API cost, non-
deterministic), the pipeline repairs known deviation classes locally:

1. Trailing junk around the JSON object (stray closing brace, duplicated
   object, commentary) - repaired by extracting the FIRST balanced JSON
   object instead of slicing from the first '{' to the last '}', which
   preserves the junk and makes json.loads fail with "Extra data".
2. submission_meta drifting from the authoritative filename metadata and
   particulars.section / particulars.roll_number disagreeing with it -
   repaired by enforce_submission_consistency() before Pydantic validation.
3. Minor structural deviations in body sections (numbering from 1 instead
   of 2, bullet items as plain strings, table 'headers' key) - repaired by
   normalize_report_data().
"""

from __future__ import annotations

import copy
import json
from typing import Optional


def extract_first_json_object(text: str) -> str:
    """Return the first complete, balanced JSON object embedded in *text*,
    with prematurely-evicted top-level keys spliced back in.

    ``json.loads`` rejects outputs such as '{"a": 1}}' or '{"a": 1} {"a": 2}'
    with "Extra data". Language models frequently emit those shapes.
    ``json.JSONDecoder.raw_decode`` parses exactly one value starting at a
    given index and reports where it ended, so we probe each candidate '{'
    position until one yields a complete object. Braces inside string
    literals are handled correctly because raw_decode is a real tokenizer.

    Observed failure shape handled here: the model closes the top-level
    object early (right after "sections") and then continues with
    ', "submission_meta": {...}}'. The first balanced object alone is then
    an *incomplete* payload, so any following ', "key": value' pairs are
    merged back into it before returning.

    Raises ValueError when *text* contains no complete JSON object.
    """
    decoder = json.JSONDecoder()

    index = text.find("{")
    while index != -1:
        try:
            _, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            index = text.find("{", index + 1)
            continue

        first = text[index:end]

        merged = _merge_stray_top_level_pairs(decoder, text, end, first)
        return merged if merged is not None else first

    raise ValueError(
        "No complete JSON object found in output.\n\n"
        f"Output:\n{text}"
    )


def _merge_stray_top_level_pairs(decoder, text, first_end, first_text):
    """Splice ', "key": value' pairs that follow a prematurely closed
    JSON object back into that object.

    Returns the merged object text, or None when the leftover text after
    *first_end* does not match the stray-pairs shape (whitespace, prose,
    duplicated objects etc.), in which case the caller keeps the plain
    first object.
    """
    merged_pairs = []
    pos = _skip_whitespace(text, first_end)

    while pos < len(text):
        char = text[pos]

        # Tolerate several comma-separated pairs: ', "a": 1, "b": 2 }'.
        if char == ",":
            pos = _skip_whitespace(text, pos + 1)
            continue

        if char != '"':
            break

        # Parse the key (a JSON string), expect ':', parse the value.
        # NOTE: raw_decode does not skip leading whitespace, hence the
        # explicit _skip_whitespace calls.
        try:
            key, key_end = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            break

        colon_pos = _skip_whitespace(text, key_end)
        if colon_pos >= len(text) or text[colon_pos] != ":":
            break

        value_start = _skip_whitespace(text, colon_pos + 1)
        try:
            _, value_end = decoder.raw_decode(text, value_start)
        except json.JSONDecodeError:
            break

        merged_pairs.append(
            json.dumps(key, ensure_ascii=False)
            + ": "
            + text[value_start:value_end]
        )
        pos = _skip_whitespace(text, value_end)

    if not merged_pairs:
        return None

    inner = first_text[1:-1].strip()
    if inner:
        inner += ", "
    return "{" + inner + ", ".join(merged_pairs) + "}"


_WHITESPACE = " \t\r\n"


def _skip_whitespace(text, pos):
    while pos < len(text) and text[pos] in _WHITESPACE:
        pos += 1
    return pos


def enforce_submission_consistency(
    data: dict,
    metadata: Optional[dict] = None,
) -> dict:
    """Force submission_meta (and its mirrors in particulars) into a
    self-consistent, schema-valid state.

    ExperimentReport validates cross-field invariants:
      particulars.section == submission_meta.division
      particulars.roll_number == submission_meta.roll_number
    Models drift on both (e.g. emitting the class "section" number or a
    department name instead of the division code).

    - When *metadata* (parsed from the authoritative document filename) is
      supplied, submission_meta is overwritten with it outright; the
      filename is ground truth for these fields.
    - particulars.section / particulars.roll_number are then synced from
      submission_meta so the cross-field validators pass deterministically.
    """
    # Deep copy so callers' payloads are never mutated by the repair.
    data = copy.deepcopy(data)

    submission_meta = data.get("submission_meta")
    if not isinstance(submission_meta, dict):
        submission_meta = {}
        data["submission_meta"] = submission_meta

    if metadata:
        submission_meta.update(
            {
                "experiment_number": metadata["experiment_number"],
                "semester_prefix": metadata["semester_prefix"],
                "division": metadata["division"],
                "roll_number": metadata["roll_number"],
                "part_suffix": metadata["part_suffix"],
            }
        )

    particulars = data.get("particulars")
    if isinstance(particulars, dict):

        division = submission_meta.get("division")
        if division:
            particulars["section"] = division

        roll_number = submission_meta.get("roll_number")
        if roll_number:
            particulars["roll_number"] = roll_number

    return data


def normalize_report_data(
    data: dict,
) -> dict:
    """Deterministically coerce the fine-tuned model's JSON into
    schema-conforming shapes.

    The model frequently produces minor structural deviations such as:
      - bullet_list items as plain strings (schema requires
        {'lead_in', 'text'} objects)
      - table 'headers' key instead of 'header'
      - body sections numbered from 1 instead of 2
      - particulars.section not matching submission_meta.division

    These are formatting differences, not content differences, so we
    normalize the structure rather than re-prompting the model.
    """
    data = dict(data)

    # 1. Sections must start at 2 and be sequential.
    sections = data.get("sections")
    if isinstance(sections, list):
        normalized_sections = []
        for index, section in enumerate(sections, start=2):
            section = dict(section) if isinstance(section, dict) else section
            if isinstance(section, dict):
                section["number"] = index

                content = section.get("content")
                if isinstance(content, list):
                    normalized_content = []
                    for block in content:
                        block = dict(block) if isinstance(block, dict) else block
                        if not isinstance(block, dict):
                            normalized_content.append(block)
                            continue

                        block_type = block.get("type")

                        # 1a. bullet_list items as plain strings -> objects
                        if block_type == "bullet_list":
                            items = block.get("items", [])
                            normalized_items = []
                            for item in items:
                                if isinstance(item, str):
                                    normalized_items.append(
                                        {
                                            "lead_in": None,
                                            "text": item,
                                        }
                                    )
                                else:
                                    normalized_items.append(item)
                            block["items"] = normalized_items

                        # 1b. table 'headers' -> 'header'
                        elif block_type == "table":
                            if "headers" in block and "header" not in block:
                                block["header"] = block.pop("headers")

                        normalized_content.append(block)
                    section["content"] = normalized_content

            normalized_sections.append(section)
        data["sections"] = normalized_sections

    # 2. particulars.section / roll_number must match submission_meta.
    data = enforce_submission_consistency(data)

    return data
