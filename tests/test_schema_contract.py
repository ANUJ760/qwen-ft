from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "schema"))

from common.prompts import INFERENCE_SYSTEM_PROMPT
from schema import ExperimentReport, TableBlock


def valid_report_data():
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


class SchemaContractTests(unittest.TestCase):
    def test_valid_report_passes(self):
        report = ExperimentReport.model_validate(valid_report_data())

        self.assertEqual(report.submission_meta.filename(), "E02_5A2_33.pdf")

    def test_table_rows_must_be_rectangular(self):
        with self.assertRaises(ValueError):
            TableBlock.model_validate(
                {
                    "type": "table",
                    "header": ["Name", "Value"],
                    "rows": [["Only one column"], ["A", "B"]],
                }
            )

    def test_table_header_width_must_match_rows(self):
        with self.assertRaises(ValueError):
            TableBlock.model_validate(
                {
                    "type": "table",
                    "header": ["Name"],
                    "rows": [["A", "B"]],
                }
            )

    def test_particulars_must_match_submission_meta(self):
        data = valid_report_data()
        data["submission_meta"]["roll_number"] = "38"

        with self.assertRaises(ValueError):
            ExperimentReport.model_validate(data)

    def test_shared_prompt_mentions_required_section_key(self):
        self.assertIn('"number"', INFERENCE_SYSTEM_PROMPT)
        self.assertIn("section_number", INFERENCE_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
