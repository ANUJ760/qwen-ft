#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

INPUT="inference/test_inputs/testdata1.txt"
OUTPUT_DIR="outputs"
VARIANTS_PER_DOC=4

ONLY=()

FORCE_EXTRACT=0
FORCE_RAW=0
DRY_RUN_DATASET=0
SKIP_EXTRACT=0
SKIP_DATASET=0
SKIP_TRAIN=0
SKIP_PDF=0

ADAPTER_DIR="finetune/checkpoints/experiment-report"

usage() {
    sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'

    echo
    echo "Options:"
    echo "  --input PATH            Raw notes input file"
    echo "  --output-dir DIR        Output directory"
    echo "  --variants-per-doc N    Raw-note variants per gold document"
    echo "  --only FILTER...        Only process matching examples"
    echo "  --force-extract         Force Gemini gold extraction"
    echo "  --force-raw             Force Gemini raw-note generation"
    echo "  --dry-run-dataset       Generate mock dataset without Gemini"
    echo "  --skip-extract          Skip gold JSON extraction"
    echo "  --skip-dataset          Skip training-pair generation"
    echo "  --skip-train            Skip QLoRA training"
    echo "  --skip-pdf              Skip PDF rendering"
    echo "  -h, --help              Show this help"

    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in

        --input)
            [[ $# -ge 2 ]] || {
                echo "ERROR: --input requires a path." >&2
                exit 1
            }
            INPUT="$2"
            shift 2
            ;;

        --output-dir)
            [[ $# -ge 2 ]] || {
                echo "ERROR: --output-dir requires a directory." >&2
                exit 1
            }
            OUTPUT_DIR="$2"
            shift 2
            ;;

        --variants-per-doc)
            [[ $# -ge 2 ]] || {
                echo "ERROR: --variants-per-doc requires a number." >&2
                exit 1
            }
            VARIANTS_PER_DOC="$2"
            shift 2
            ;;

        --only)
            shift

            while [[ $# -gt 0 && "$1" != --* ]]; do
                ONLY+=("$1")
                shift
            done
            ;;

        --force-extract)
            FORCE_EXTRACT=1
            shift
            ;;

        --force-raw)
            FORCE_RAW=1
            shift
            ;;

        --dry-run-dataset)
            DRY_RUN_DATASET=1
            shift
            ;;

        --skip-extract)
            SKIP_EXTRACT=1
            shift
            ;;

        --skip-dataset)
            SKIP_DATASET=1
            shift
            ;;

        --skip-train)
            SKIP_TRAIN=1
            shift
            ;;

        --skip-pdf)
            SKIP_PDF=1
            shift
            ;;

        -h|--help)
            usage
            ;;

        *)
            echo "ERROR: Unknown option: $1" >&2
            echo
            usage
            ;;
    esac
done


echo
echo "================================================================"
echo " Document Compilation Pipeline"
echo "================================================================"
echo


PYTHON="${PYTHON:-python}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "ERROR: Python interpreter not found: $PYTHON" >&2
    echo "Activate your virtual environment first." >&2
    exit 1
fi


PY_VERSION="$("$PYTHON" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"

PY_MAJOR="$("$PYTHON" -c 'import sys; print(sys.version_info.major)')"
PY_MINOR="$("$PYTHON" -c 'import sys; print(sys.version_info.minor)')"

echo "Python: $PY_VERSION"
echo "Interpreter: $("$PYTHON" -c 'import sys; print(sys.executable)')"


if [[ "$PY_MAJOR" -lt 3 || ( "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 10 ) ]]; then
    echo
    echo "ERROR: Python 3.10+ is required." >&2
    exit 1
fi


echo
echo "Checking Python dependencies..."


"$PYTHON" - <<'PY'
import importlib.util
import sys

required = {
    "pydantic": "pydantic",
    "docx": "python-docx",
    "google.genai": "google-genai",
    "yaml": "pyyaml",
    "torch": "torch",
    "transformers": "transformers",
    "datasets": "datasets",
    "accelerate": "accelerate",
    "peft": "peft",
    "trl": "trl",
    "bitsandbytes": "bitsandbytes",
}

missing = []

for module, package in required.items():
    if importlib.util.find_spec(module) is None:
        missing.append(package)

if missing:
    print()
    print("Missing Python packages:")
    for package in missing:
        print(f"  - {package}")
    print()
    print("Install them with:")
    print("  pip install -U " + " ".join(missing))
    sys.exit(1)

print("All required Python packages found.")
PY


if [[ "$SKIP_EXTRACT" -eq 0 || "$SKIP_DATASET" -eq 0 ]]; then

    echo
    echo "Checking GEMINI_API_KEY..."

    if [[ -z "${GEMINI_API_KEY:-}" ]]; then
        echo
        echo "ERROR: GEMINI_API_KEY is not set." >&2
        echo
        echo "Set it with:"
        echo
        echo '  export GEMINI_API_KEY="your_api_key_here"'
        echo
        echo "Or skip Gemini-dependent stages with:"
        echo
        echo "  --skip-extract --skip-dataset"
        echo
        exit 1
    fi

    echo "GEMINI_API_KEY: set"

fi


if [[ "$SKIP_PDF" -eq 0 ]]; then

    echo
    echo "Checking LibreOffice..."

    if command -v soffice >/dev/null 2>&1; then
        echo "soffice: found"
    else
        echo
        echo "WARNING: LibreOffice (soffice) was not found."
        echo "PDF rendering may fail."
        echo "Use --skip-pdf if you only need DOCX output."
    fi

fi


if [[ "$SKIP_TRAIN" -eq 1 ]]; then

    echo
    echo "Checking existing LoRA adapter..."

    if [[ ! -d "$ADAPTER_DIR" ]]; then
        echo
        echo "ERROR: --skip-train was specified, but no adapter exists:"
        echo "  $ADAPTER_DIR"
        echo
        echo "Either remove --skip-train or provide the adapter."
        exit 1
    fi

    echo "LoRA adapter: found"

fi


if [[ ! -f "run_pipeline.py" ]]; then
    echo
    echo "ERROR: run_pipeline.py not found."
    echo "Expected:"
    echo "  $SCRIPT_DIR/run_pipeline.py"
    exit 1
fi


if [[ ! -d "examples" ]]; then
    echo
    echo "ERROR: examples/ directory not found."
    exit 1
fi


if [[ ! -d "schema" ]]; then
    echo
    echo "ERROR: schema/ directory not found."
    exit 1
fi


CMD=(
    "$PYTHON"
    run_pipeline.py
    --input "$INPUT"
    --output-dir "$OUTPUT_DIR"
    --variants-per-doc "$VARIANTS_PER_DOC"
)


if [[ ${#ONLY[@]} -gt 0 ]]; then
    CMD+=(--only "${ONLY[@]}")
fi

if [[ "$FORCE_EXTRACT" -eq 1 ]]; then
    CMD+=(--force-extract)
fi

if [[ "$FORCE_RAW" -eq 1 ]]; then
    CMD+=(--force-raw)
fi

if [[ "$DRY_RUN_DATASET" -eq 1 ]]; then
    CMD+=(--dry-run-dataset)
fi

if [[ "$SKIP_EXTRACT" -eq 1 ]]; then
    CMD+=(--skip-extract)
fi

if [[ "$SKIP_DATASET" -eq 1 ]]; then
    CMD+=(--skip-dataset)
fi

if [[ "$SKIP_TRAIN" -eq 1 ]]; then
    CMD+=(--skip-train)
fi

if [[ "$SKIP_PDF" -eq 1 ]]; then
    CMD+=(--skip-pdf)
fi


echo
echo "================================================================"
echo " Configuration"
echo "================================================================"
echo
echo "Input:             $INPUT"
echo "Output directory:  $OUTPUT_DIR"
echo "Variants/doc:      $VARIANTS_PER_DOC"
echo "Force extraction:  $FORCE_EXTRACT"
echo "Force raw data:    $FORCE_RAW"
echo "Skip extraction:   $SKIP_EXTRACT"
echo "Skip dataset:      $SKIP_DATASET"
echo "Skip training:     $SKIP_TRAIN"
echo "Skip PDF:          $SKIP_PDF"

if [[ ${#ONLY[@]} -gt 0 ]]; then
    echo "Only:              ${ONLY[*]}"
fi

echo
echo "================================================================"
echo " Executing Pipeline"
echo "================================================================"
echo

printf 'Command:'
printf ' %q' "${CMD[@]}"
echo
echo


exec "${CMD[@]}"