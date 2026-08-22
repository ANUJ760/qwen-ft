"""
Structured JSON schema for Experiment Report documents.

This is the CONTRACT between:
  - the fine-tuned LLM (which must emit JSON conforming to this schema)
  - the deterministic formatter (which consumes validated instances and
    produces the DOCX/PDF)

Design notes
------------
- Derived from E01a, E02, E03 (E01b / the embedded SRS is treated as an
  out-of-scope exception per project decision).
- `sections` is intentionally open-ended: the number and titles of body
  sections vary per assignment (per GCR instructions), so the schema does
  not hardcode "Section 2 must be X". Instead it validates *shape*
  (numbering, content block types), not fixed content.
- `content` blocks are a tagged union (discriminated by `type`) so the
  formatter can dispatch each block to the right rendering routine without
  guessing.
- `submission_meta` carries fields that live OUTSIDE the visible document
  body but are required to generate the correct output filename per the
  Compilation Guidelines naming convention (e.g. E01_5A2_33.pdf).
"""

from __future__ import annotations

from datetime import date
from typing import List, Optional, Union, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Content blocks (tagged union)
# ---------------------------------------------------------------------------

class ParagraphBlock(BaseModel):
    type: Literal["paragraph"] = "paragraph"
    text: str = Field(..., min_length=1)


class BulletItem(BaseModel):
    lead_in: Optional[str] = Field(
        None,
        description="Bolded lead-in phrase before the colon, e.g. 'Choice of actors'. "
                     "Omit for plain bullets with no bolded lead-in.",
    )
    text: str = Field(..., min_length=1)


class BulletListBlock(BaseModel):
    type: Literal["bullet_list"] = "bullet_list"
    items: List[BulletItem] = Field(..., min_length=1)


class FigureBlock(BaseModel):
    type: Literal["figure"] = "figure"
    image_ref: str = Field(..., description="Path or identifier of the source image asset.")
    caption: str = Field(..., description="e.g. 'Fig. 1 \u2014 UML Use Case Diagram of the OVRS'")
    figure_number: Optional[int] = None


class TableBlock(BaseModel):
    type: Literal["table"] = "table"
    header: Optional[List[str]] = None
    rows: List[List[str]] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _rows_are_rectangular(self) -> "TableBlock":
        width = len(self.rows[0])
        if width == 0:
            raise ValueError("table rows must contain at least one column")

        for index, row in enumerate(self.rows, start=1):
            if len(row) != width:
                raise ValueError(
                    "table rows must all have the same number of columns; "
                    f"row 1 has {width}, row {index} has {len(row)}"
                )

        if self.header is not None and len(self.header) != width:
            raise ValueError(
                "table header must have the same number of columns as rows; "
                f"header has {len(self.header)}, rows have {width}"
            )

        return self


ContentBlock = Union[ParagraphBlock, BulletListBlock, FigureBlock, TableBlock]


# ---------------------------------------------------------------------------
# Section
# ---------------------------------------------------------------------------

class Section(BaseModel):
    number: int = Field(..., ge=1, description="Section ordinal, e.g. 2 for '2. Short Description...'")
    title: str = Field(..., min_length=1, description="Title WITHOUT the leading number, e.g. 'Short Description of the UML Modeling View'")
    content: List[ContentBlock] = Field(..., min_length=1)

    @property
    def heading(self) -> str:
        """Rendered heading text, e.g. '2. Short Description of the UML Modeling View'."""
        return f"{self.number}. {self.title}"


# ---------------------------------------------------------------------------
# Particulars (the fixed front-matter block required by guideline note b)
# ---------------------------------------------------------------------------

class Particulars(BaseModel):
    case_study_title: str
    aim: str
    problem_statement: str
    author: str
    section: str = Field(..., description="e.g. 'A2'")
    roll_number: str
    date_of_compilation: date


# ---------------------------------------------------------------------------
# Submission metadata (drives filename generation, not rendered in-body)
# ---------------------------------------------------------------------------

class SubmissionMeta(BaseModel):
    experiment_number: int = Field(..., ge=1)
    semester_prefix: str = Field(..., description="e.g. '5' as seen in E01_5A2_33")
    division: str = Field(..., description="e.g. 'A2'")
    roll_number: str
    part_suffix: Optional[str] = Field(
        None,
        description="'a', 'b', ... — only set when a single experiment submission "
                     "comprises multiple documents (per guideline note e).",
    )

    @field_validator("part_suffix")
    @classmethod
    def _lowercase_suffix(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip().lower()
        if not v.isalpha() or len(v) != 1:
            raise ValueError("part_suffix must be a single letter, e.g. 'a'")
        return v

    def filename(self, ext: str = "pdf") -> str:
        base = f"E{self.experiment_number:02d}_{self.semester_prefix}{self.division}_{self.roll_number}"
        if self.part_suffix:
            base += self.part_suffix
        return f"{base}.{ext}"


# ---------------------------------------------------------------------------
# Top-level document
# ---------------------------------------------------------------------------

class ExperimentReport(BaseModel):
    title: str = Field(..., description="Rendered as the 14pt bold ALL-CAPS document title.")
    particulars: Particulars
    sections: List[Section] = Field(..., min_length=1)
    submission_meta: SubmissionMeta

    @model_validator(mode="after")
    def _sections_numbered_sequentially(self) -> "ExperimentReport":
        expected = 2  # Section 1 is conventionally the Particulars block itself
        for s in self.sections:
            if s.number != expected:
                raise ValueError(
                    f"Sections must be numbered sequentially starting at 2 "
                    f"(Particulars occupies section 1). Expected {expected}, got {s.number}."
                )
            expected += 1
        return self

    @model_validator(mode="after")
    def _particulars_match_submission_meta(self) -> "ExperimentReport":
        if self.particulars.roll_number != self.submission_meta.roll_number:
            raise ValueError(
                "particulars.roll_number must match "
                "submission_meta.roll_number"
            )

        if self.particulars.section != self.submission_meta.division:
            raise ValueError(
                "particulars.section must match submission_meta.division"
            )

        return self


# ---------------------------------------------------------------------------
# Convenience: emit JSON Schema (for prompting the LLM / RAG grounding)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    print(json.dumps(ExperimentReport.model_json_schema(), indent=2))
