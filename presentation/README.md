# Five-slide presentation

Open `LLM_AuthorBench_Assignment.pptx` in PowerPoint or another compatible editor. Text and result tables are editable. Source references and interpretation details are in speaker notes.

The deck uses measured values from `results/presentation_data.json`. The original and ablated ranking tables are in the repository. Re-running the machine-learning pipeline updates the CSV/Markdown outputs; it does not automatically overwrite the reviewed PowerPoint.

`build_deck.mjs` is the source used in the supplied artifact authoring runtime. It requires Node.js and `@oai/artifact-tool`, plus that runtime's presentation finalization helpers. These are presentation-authoring dependencies, not requirements for running the ML experiment. Set `PROJECT_ROOT`, `SKILL_DIR`, `RUNTIME_PYTHON`, and `RUNTIME_NODE_MODULES` to the corresponding local runtime paths. Run a copy of the source in a private build directory whose `node_modules` points to the supplied packages. The finalizer intentionally refuses to overwrite an existing final deck, so use a new final filename for revisions.

The deck has five slides including its opening dataset slide. Pending licence/claim status is visible. Member names are omitted until provided by the group.
