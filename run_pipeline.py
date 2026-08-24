from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "inference" / "test_inputs" / "testdata1.txt"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"


def run_step(name: str, command: list[str], env: dict[str, str] | None = None) -> None:
    print()
    print("=" * 80)
    print(name)
    print("=" * 80)
    print("$ " + " ".join(str(part) for part in command))
    print()

    subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
    )


def render_outputs(json_path: Path, output_dir: Path, render_pdf: bool) -> tuple[Path, Path | None]:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "schema"))

    from formatter.formatter import render_docx, render_pdf as render_pdf_file
    from schema import ExperimentReport

    with json_path.open("r", encoding="utf-8") as f:
        report = ExperimentReport.model_validate(json.load(f))

    output_dir.mkdir(parents=True, exist_ok=True)

    docx_path = output_dir / report.submission_meta.filename(ext="docx")
    render_docx(report, str(docx_path))

    pdf_path = None
    if render_pdf:
        pdf_path = Path(render_pdf_file(report, str(output_dir)))

    return docx_path, pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the document compilation pipeline sequentially: extract gold "
            "JSON, generate training pairs, fine-tune, infer JSON, and render "
            "DOCX/PDF outputs."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Raw notes input file for the final inference step.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated JSON, DOCX, and PDF outputs.",
    )

    parser.add_argument(
        "--variants-per-doc",
        type=int,
        default=4,
        help="Raw-note variants to generate per gold document.",
    )

    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help=(
            "Optional filename filters passed to gold extraction, such as "
            "E05 or E05_5A2_33.pdf."
        ),
    )

    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Force Gemini to rescan documents even when gold JSON exists.",
    )

    parser.add_argument(
        "--force-raw",
        action="store_true",
        help="Force Gemini to regenerate raw-note variants.",
    )

    parser.add_argument(
        "--dry-run-dataset",
        action="store_true",
        help="Use mock raw-note generation instead of Gemini for dataset generation.",
    )

    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip gold JSON extraction.",
    )

    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Skip training-pair generation.",
    )

    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip QLoRA training and use the existing adapter checkpoint.",
    )

    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Render DOCX only and skip LibreOffice PDF conversion.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python = sys.executable

    input_path = args.input
    if not input_path.is_absolute():
        input_path = ROOT / input_path

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    json_output = output_dir / "generated_report.json"

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not args.skip_extract:
        command = [
            python,
            "dataset/extract_gold_json.py",
        ]

        if args.only:
            command.extend(["--only", *args.only])

        if args.force_extract:
            command.append("--force")

        run_step("1. Extract gold JSON", command)

    if not args.skip_dataset:
        command = [
            python,
            "dataset/generate_dataset.py",
            "--variants-per-doc",
            str(args.variants_per_doc),
        ]

        if args.force_raw:
            command.append("--force-raw")

        if args.dry_run_dataset:
            command.append("--dry-run")

        run_step("2. Generate teacher-distillation dataset", command)

    if not args.skip_train:
        run_step(
            "3. Fine-tune QLoRA adapter",
            [
                python,
                "finetune/train_qlora.py",
            ],
        )

    run_step(
        "4. Generate validated report JSON",
        [
            python,
            "inference/generate_report.py",
            "--input",
            str(input_path),
            "--output",
            str(json_output),
        ],
    )

    print()
    print("=" * 80)
    print("5. Render final documents")
    print("=" * 80)

    docx_path, pdf_path = render_outputs(
        json_output,
        output_dir,
        render_pdf=not args.skip_pdf,
    )

    print()
    print("Pipeline complete.")
    print(f"JSON: {json_output}")
    print(f"DOCX: {docx_path}")
    if pdf_path:
        print(f"PDF : {pdf_path}")
    else:
        print("PDF : skipped")


if __name__ == "__main__":
    main()
