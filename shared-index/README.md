# shared-index

BM25 files (`manifest.json`, `chunks.jsonl`) of the index collections in the shared Qdrant cluster.
`init-start.cmd` runs `scripts.use_shared_index`, which points each PC's `RUNTIME_DIR/indexes/active_<profile>.json`
at these folders. Nothing here is written to Qdrant.

Keep exactly one folder per profile (`bge`, `qwen`), matching the collections in Qdrant.

After rebuilding the shared index (index builder only):
1. Delete the old folder for that profile here.
2. Copy `RUNTIME_DIR/indexes/<new collection>/manifest.json` and `chunks.jsonl` into `shared-index/<new collection>/`.
3. Commit, and tell the team to pull and run `init-start` again.
