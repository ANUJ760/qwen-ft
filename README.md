# Finetuned model for document generation

Pipeline: raw content → fine-tuned LLM → structured JSON →
validation → deterministic formatter → DOCX/PDF, matching a submission's
Compilation Guidelines.

## Project structure

```
doc_compiler/
├── README.md                    (this file)
│
├── guidelines/
│   └── Compilation_Guidelines.pdf   Source formatting rules (margins, fonts,
│                                     spacing, naming convention). The
│                                     formatter is a literal translation of
│                                     this file — see formatter/formatter.py.
│
├── examples/                    Practical/example source documents used to
│   ├── training-data1         derive the content schema and (later) seed
│   ├── training-data2          the teacher-distillation dataset.
│   ├── training-data3          
│   ├── training-data4          
│   └── training-data5          
│        
│
├── schema/
│   └── schema.py                 [BUILT] Pydantic models (ExperimentReport,
│                                  Particulars, Section, content blocks,
│                                  SubmissionMeta). The contract between the
│                                  LLM's output and the formatter's input.
│                                  Also the validation gate (Step 4 of the
│                                  pipeline).
│
├── formatter/
│   └── formatter.py              [BUILT] Deterministic ExperimentReport ->
│                                  DOCX -> PDF renderer. Zero content-
│                                  generation logic; every parameter maps
│                                  directly to guidelines/Compilation_
│                                  Guidelines.pdf. Verified against real
│                                  E02-shaped input (rendered + visually
│                                  inspected).
│
├── dataset/                      [BUILT]
│   ├── generate_dataset.py       Teacher-distillation script: for each
│   │                             example doc, (a) reverse-generates a
│   │                             plausible messy "raw content" input, and
│   │                             (b) produces the target ExperimentReport
│   │                             JSON. Every generated pair is validated
│   │                             against schema.py before being kept.
│   ├── raw_content/              Generated synthetic raw-content inputs.
│   ├── target_json/              Generated + validated target JSON per pair.
│   └── training_pairs.jsonl      Final assembled fine-tuning dataset
│                                 (prompt/completion or chat-format pairs).
│
├── finetune/                     [BUILT]
│   ├── train_qlora.py            LoRA/QLoRA fine-tuning script (local GPU,
│   │                             8-12GB VRAM -> 7B-class base model,
│   │                             4-bit quant).
│   ├── configs/
│   │   └── qlora_config.yaml     LoRA rank/alpha, target modules, training
│   │                             hyperparameters.
│   └── checkpoints/              Saved LoRA adapters.
│
├── inference/                    [BUILT]
│   └── generate_report.py        Runs fine-tuned model against new raw
│                                 content, with the current Compilation
│                                 Guidelines injected via prompting/RAG
│                                 (not baked into the fine-tune, so
│                                 guideline changes don't require
│                                 retraining). Outputs raw JSON candidate.
│
├── validation/                   [NOT YET BUILT — logic mostly lives in
│   └── validate_json.py          schema.py already; this wraps it with
│                                 retry-on-invalid-JSON handling for the
│                                 inference loop.]
│
├── evaluation/                   [NOT YET BUILT]
│   └── evaluate.py               Compares generated docs against held-out
│                                 practical examples: structural fidelity,
│                                 content fidelity, formatting fidelity.
│
└── outputs/                      Final rendered DOCX/PDF, named per the
                                  guideline's convention (e.g. E02_5A2_33.pdf),
                                  via SubmissionMeta.filename().
```

## Pipeline recap

```
Raw Content --> Fine-tuned LLM (+ Guidelines via RAG/prompting)
            --> Structured JSON (schema.py contract)
            --> Validation (schema.py / validation/validate_json.py)
            --> Formatter (formatter/formatter.py)
            --> DOCX --> PDF (outputs/)
```


