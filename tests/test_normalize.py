from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "schema"))
sys.path.insert(0, str(ROOT / "dataset"))

from common.normalize import (
    enforce_submission_consistency,
    extract_first_json_object,
    normalize_report_data,
)
from schema import ExperimentReport


def valid_report_data():
    """Minimal schema-valid payload (mirrors tests/test_schema_contract.py)."""
    return {
        "title": "Experiment 2",
        "particulars": {
            "case_study_title": "Online Vehicle Rental System",
            "aim": "Draw a UML model.",
            "problem_statement": "Model the system.",
            "author": "Test Student",
            "section": "A2",
            "roll_number": "33",
            "date_of_compilation": "2026-08-22",
        },
        "sections": [
            {
                "number": 2,
                "title": "Short Description",
                "content": [
                    {
                        "type": "paragraph",
                        "text": "The system supports vehicle rentals.",
                    }
                ],
            }
        ],
        "submission_meta": {
            "experiment_number": 2,
            "semester_prefix": "5",
            "division": "A2",
            "roll_number": "33",
            "part_suffix": None,
        },
    }


def online_voting_report_payload():
    """Replica of the fine-tuned model's output from the failing pipeline
    run (particulars.section "2", invented division "CSE")."""
    return {
        "title": "Online Voting System",
        "particulars": {
            "case_study_title": "Online Voting System",
            "aim": (
                "Design an online voting system where registered students "
                "can securely cast votes."
            ),
            "problem_statement": (
                "Students need registration and login. Admin manages "
                "candidates and election details."
            ),
            "author": "Student",
            "section": "2",
            "roll_number": "123456789",
            "date_of_compilation": "2023-10-15",
        },
        "sections": [
            {
                "number": 2,
                "title": "Introduction",
                "content": [
                    {
                        "type": "paragraph",
                        "text": "This report outlines the design.",
                    },
                    {
                        "type": "bullet_list",
                        "items": [
                            {"text": "Students can register."},
                        ],
                    },
                ],
            },
            {
                "number": 3,
                "title": "System Architecture",
                "content": [
                    {
                        "type": "paragraph",
                        "text": "Client-server architecture.",
                    },
                ],
            },
        ],
        "submission_meta": {
            "experiment_number": 1,
            "semester_prefix": "2023",
            "division": "CSE",
            "roll_number": "123456789",
            "part_suffix": None,
        },
    }


def broken_premature_close_payload():
    """Builds the EXACT malformed shape observed in the failing pipeline
    run: the model closes the top-level object right after "sections" and
    then continues with ', "submission_meta": {...}}'. Returns
    (payload, broken)."""
    payload = json.dumps(online_voting_report_payload())
    index = payload.index('"submission_meta"')
    head = payload[:index].rstrip().rstrip(",")
    return payload, head + "}, " + payload[index:]


class ExtractFirstJsonObjectTests(unittest.TestCase):
    def test_plain_object_unchanged(self):
        text = '{"a": 1, "b": {"c": 2}}'
        self.assertEqual(extract_first_json_object(text), text)

    def test_stray_trailing_brace_is_dropped(self):
        # Exact failure shape from the pipeline run: a complete object
        # followed by an extra closing brace made json.loads raise
        # "Extra data".
        text = '{"a": {"b": 1}}}'
        self.assertEqual(extract_first_json_object(text), '{"a": {"b": 1}}')

    def test_duplicate_object_keeps_first(self):
        first = json.dumps({"title": "x"})
        text = first + "\n" + json.dumps({"title": "y"})
        self.assertEqual(
            json.loads(extract_first_json_object(text)),
            {"title": "x"},
        )

    def test_leading_commentary_is_dropped(self):
        text = 'Sure! Here is the JSON:\n{"ok": true}'
        self.assertEqual(
            json.loads(extract_first_json_object(text)),
            {"ok": True},
        )

    def test_braces_inside_strings_are_preserved(self):
        text = '{"text": "set } or { brace"} and some trailing junk'
        self.assertEqual(
            json.loads(extract_first_json_object(text))["text"],
            "set } or { brace",
        )

    def test_missing_or_malformed_object_raises_value_error(self):
        with self.assertRaises(ValueError):
            extract_first_json_object("no braces here")
        with self.assertRaises(ValueError):
            extract_first_json_object('{ "unbalanced": ')

    def test_premature_close_with_trailing_submission_meta_is_merged(self):
        # Regression for the actual pipeline crash: the model emitted
        # '{..., "sections": [...]}, "submission_meta": {...}}' - the
        # top-level object was closed early and submission_meta followed
        # as a stray pair, producing "Extra data ... char 2057".
        payload, broken = broken_premature_close_payload()

        self.assertNotEqual(json.loads(payload), None)
        with self.assertRaises(json.JSONDecodeError):
            json.loads(broken)

        extracted = extract_first_json_object(broken)
        self.assertEqual(json.loads(extracted), json.loads(payload))

    def test_multiple_stray_pairs_are_merged(self):
        broken = '{"a": 1}, "b": 2, "c": {"d": [1, 2]}}'
        self.assertEqual(
            json.loads(extract_first_json_object(broken)),
            {"a": 1, "b": 2, "c": {"d": [1, 2]}},
        )


class EnforceSubmissionConsistencyTests(unittest.TestCase):
    def test_filename_metadata_overrides_drifted_submission_meta(self):
        data = {
            "particulars": {"section": "2", "roll_number": "123456789"},
            "submission_meta": {
                "experiment_number": 99,
                "semester_prefix": "2023",
                "division": "CSE",
                "roll_number": "123456789",
                "part_suffix": None,
            },
        }
        metadata = {
            "experiment_number": 3,
            "semester_prefix": "5",
            "division": "A2",
            "roll_number": "38",
            "part_suffix": None,
        }

        fixed = enforce_submission_consistency(data, metadata)

        self.assertEqual(fixed["submission_meta"]["division"], "A2")
        self.assertEqual(fixed["submission_meta"]["roll_number"], "38")
        self.assertEqual(fixed["submission_meta"]["experiment_number"], 3)
        self.assertEqual(fixed["submission_meta"]["semester_prefix"], "5")
        self.assertEqual(fixed["particulars"]["section"], "A2")
        self.assertEqual(fixed["particulars"]["roll_number"], "38")

    def test_particulars_synced_without_metadata(self):
        data = {
            "particulars": {"section": "2", "roll_number": "33"},
            "submission_meta": {"division": "B1", "roll_number": "77"},
        }

        fixed = enforce_submission_consistency(data)

        self.assertEqual(fixed["particulars"]["section"], "B1")
        self.assertEqual(fixed["particulars"]["roll_number"], "77")

    def test_caller_payload_not_mutated(self):
        data = {
            "particulars": {"section": "X", "roll_number": "1"},
            "submission_meta": {"division": "Y", "roll_number": "2"},
        }

        enforce_submission_consistency(data)

        self.assertEqual(data["particulars"]["section"], "X")
        self.assertEqual(data["particulars"]["roll_number"], "1")


class NormalizeReportDataTests(unittest.TestCase):
    def test_sections_renumbered_from_two(self):
        data = {
            "sections": [
                {"number": 1, "title": "A"},
                {"number": 7, "title": "B"},
            ]
        }

        fixed = normalize_report_data(data)

        self.assertEqual(
            [section["number"] for section in fixed["sections"]],
            [2, 3],
        )

    def test_bullet_string_items_wrapped_as_objects(self):
        data = {
            "sections": [
                {
                    "content": [
                        {
                            "type": "bullet_list",
                            "items": ["plain bullet"],
                        }
                    ]
                }
            ]
        }

        fixed = normalize_report_data(data)

        items = fixed["sections"][0]["content"][0]["items"]
        self.assertEqual(items, [{"lead_in": None, "text": "plain bullet"}])

    def test_table_headers_key_renamed(self):
        data = {
            "sections": [
                {
                    "content": [
                        {
                            "type": "table",
                            "headers": ["Name"],
                            "rows": [["value"]],
                        }
                    ]
                }
            ]
        }

        fixed = normalize_report_data(data)

        block = fixed["sections"][0]["content"][0]
        self.assertIn("header", block)
        self.assertNotIn("headers", block)


class InferenceRegressionTests(unittest.TestCase):
    """Replays the exact step-4 failure from the pipeline log."""

    def test_extract_parse_normalize_validate_succeeds(self):
        payload = json.dumps(online_voting_report_payload())

        # Model emitted the object plus a stray closing brace; the old
        # first-'{'..last-'}' slice kept it and json.loads raised
        # "Extra data: line 1 column 2058 (char 2057)".
        result = payload + "}"

        parsed = json.loads(extract_first_json_object(result))
        report = ExperimentReport.model_validate(
            normalize_report_data(parsed)
        )

        self.assertEqual(report.title, "Online Voting System")
        self.assertEqual(report.particulars.section, "CSE")
        self.assertEqual(report.particulars.roll_number, "123456789")


class ValidateReportRegressionTests(unittest.TestCase):
    """Exercises inference/generate_report.validate_report end-to-end.

    Skipped automatically when the heavy ML dependencies are absent.
    """

    def _validate_report(self):
        try:
            from inference.generate_report import validate_report
        except ImportError as exc:
            self.skipTest(f"ML dependencies unavailable: {exc}")
        return validate_report

    def test_output_with_extra_trailing_brace_validates(self):
        validate_report = self._validate_report()

        report = validate_report(
            json.dumps(online_voting_report_payload()) + "}"
        )

        self.assertEqual(report.title, "Online Voting System")

    def test_duplicated_object_output_validates(self):
        validate_report = self._validate_report()

        payload = json.dumps(online_voting_report_payload())
        report = validate_report(payload + "\n" + payload)

        self.assertEqual(report.title, "Online Voting System")

    def test_premature_close_output_validates(self):
        validate_report = self._validate_report()

        _, broken = broken_premature_close_payload()
        report = validate_report(broken)

        self.assertEqual(report.title, "Online Voting System")
        self.assertEqual(
            report.submission_meta.division,
            "CSE",
        )


class GoldValidationRegressionTests(unittest.TestCase):
    """Replays the step-1 failures for E03/E04/E05_5A2_38."""

    def _extract_gold(self):
        from extract_gold_json import parse_filename, validate_gold

        return parse_filename, validate_gold

    def test_section_division_mismatch_repaired_from_filename(self):
        parse_filename, validate_gold = self._extract_gold()

        metadata = parse_filename(Path("E03_5A2_38.pdf"))
        data = online_voting_report_payload()
        data["submission_meta"].update(metadata)

        report = validate_gold(data, metadata)

        self.assertEqual(report.submission_meta.experiment_number, 3)
        self.assertEqual(report.submission_meta.semester_prefix, "5")
        self.assertEqual(report.submission_meta.division, "A2")
        self.assertEqual(report.submission_meta.roll_number, "38")
        self.assertIsNone(report.submission_meta.part_suffix)
        self.assertEqual(report.particulars.section, "A2")
        self.assertEqual(report.particulars.roll_number, "38")

    def test_without_metadata_particulars_follow_submission_meta(self):
        _, validate_gold = self._extract_gold()

        report = validate_gold(online_voting_report_payload())

        self.assertEqual(report.particulars.section, "CSE")

    def test_part_suffix_parsed_and_preserved(self):
        parse_filename, validate_gold = self._extract_gold()

        metadata = parse_filename(Path("E02_5A2_38a.pdf"))
        data = valid_report_data()
        data["submission_meta"].update(metadata)

        report = validate_gold(data, metadata)

        self.assertEqual(report.submission_meta.part_suffix, "a")
        self.assertEqual(
            report.submission_meta.filename(),
            "E02_5A2_38a.pdf",
        )


if __name__ == "__main__":
    unittest.main()