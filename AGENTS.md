# AGENTS

## Mission
- Prioritize safe, evidence-based updates to the CVPR report workflow, especially polish work in [report_pw_pp.tex](report_pw_pp.tex).
- Keep all performance and accuracy claims tied to local CSV/log artifacts only.

## Primary References
- Project commands and artifact map: [README.md](README.md)
- Main report source: [report_pw_pp.tex](report_pw_pp.tex)
- Existing specialized custom agent: [.github/agents/cvpr-report-integration.agent.md](.github/agents/cvpr-report-integration.agent.md)

## High-Value Commands
- Build images: `docker compose build trainer-classical trainer-quantum`
- Run sweeps: `bash run_all_classical.sh` then `bash run_all_quantum.sh`
- Extract latest-run metrics: `docker compose run --rm extract-results python3 extract_results.py --latest-run --csv summary_results_latest.csv`
- Compile report: `pdflatex -interaction=nonstopmode report_pw_pp.tex`
- Resolve bibliography: `bibtex parallel_qkernel_report_final` then run `pdflatex` twice

## Report Polishing Rules (CVPR Two-Column)
- Keep TikZ diagrams width-constrained in one column (for example with `\\resizebox{\\columnwidth}{!}{...}`) to avoid cross-column overflow.
- Preserve backend naming and formatting consistency: `\\texttt{numpy}`, `\\texttt{torch}`, `\\texttt{cuda\\_states}`.
- Preserve metric formatting consistency in prose and tables (F1/AUC wording, stable decimal precision inside one table).
- Verify every updated numeric statement against its source CSV/log before finalizing text.
- Prefer minimal edits that do not change scientific claims unless new extracted data is explicitly provided.

## Data Source Expectations For Tables/Text
- Latest-run quantum summary usually comes from `summary_results_latest.csv` (or `summary_results_latest_alias.csv` when present).
- Paired classical-vs-quantum deltas come from `summary_comparison_by_dataset_difficulty.csv`.
- Benchmark figures/tables should align with files under `benchmark_results/`.

## Guardrails
- Do not invent metrics, rows, figure interpretations, or references.
- Do not change experiment code for a report-only polishing request.
- If numbers across CSVs conflict, stop and ask which source has priority.
- If LaTeX compiles with warnings affecting layout, report them and propose targeted fixes.

## Suggested Workflow For "Polish Report" Requests
1. Confirm which result scope is intended (latest run, specific run, or historical aggregate).
2. Validate source CSV/log files before touching report text/tables.
3. Update only the affected report sections.
4. Compile LaTeX/BibTeX and check for unresolved refs, overfull boxes, and float placement regressions.
5. Summarize exactly which report claims/tables changed and which source files justified them.
