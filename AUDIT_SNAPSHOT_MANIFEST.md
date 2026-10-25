# VANTAGE current-source audit snapshot

This snapshot is a source-and-test artifact for independent review of the
current repaired working tree.  It contains application source, tests, text
and JSONL training data, deployment/docs, and lightweight model configuration
artifacts.

Excluded deliberately:

- GGUF base models and LoRA weights (`*.gguf`)
- training output checkpoints (`training/out/`, including `*.safetensors`)
- experiment results (`results/`)
- SQLite replay databases and the recorded camera clip (these do not compress
  sufficiently for the portable handoff limit)
- runtime caches, Python bytecode, temporary test files, and local logs
- generated previous release archives and unsafe replay samples

The omitted learned artifacts are not needed to inspect the deterministic
parser, validator, executor, protocol, ingestion, evaluation, UI, or
regression-test changes.  Tests that require the omitted replay databases or
camera clip will need those separately.  Model configuration contracts and
model readme / license files remain included where present.

Archive provenance is recorded by the SHA-256 sidecar created alongside this
file's ZIP archive.  This manifest describes the archive policy; it is not a
claim that a particular model weight is present.
