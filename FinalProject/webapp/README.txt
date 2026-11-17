LOCAL CODE ATTRIBUTION PLATFORM

Setup, notebook links, model verification and commands: ../README.md
Start from the repository root: python FinalProject/webapp/server.py --device cpu
Then visit http://127.0.0.1:8765.

Pages: / (hub), /live, /generate, /study, /results.
Inference supports both classifiers, comments kept/removed and first512/chunks.
Full-code mode averages scores across overlapping windows. Scores are uncalibrated.
Manual input is never executed or saved. Generated study code IS saved as evidence.
OpenRouter generation uses your credits; local classification/retesting does not.
Keys remain in request/worker memory and are excluded from saved study results.

Test all analyzes the active batch. Test held-out examples replaces that batch with
five benchmark examples. Import downloaded generation JSON to restore older outputs.

Saved study retest: select run, View saved study, choose coverage, then
Retest saved code locally (no API key). Existing generations are reused.
Progress saves under ../study_results/<run_id>/. Download ZIPs for backup.
Import experiment/study ZIPs on /results; private evidence is not bundled in GitHub.

Prepare checkpoints/tokenizer with setup_models.py before starting.
Keep the server process open while inference or studies run.
