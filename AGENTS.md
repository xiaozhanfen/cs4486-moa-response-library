# Final project scope

- This directory is the accepted MoA Response Library project. Historical projects live in `../archive/`.
- `response_release/` is the exact scored Version 1 bundle. Preserve its files and hashes. Build future experiments into a fresh release directory.
- Actual score is Public 0.01814 and Private 0.01590, submission 57940904. The group's reference thresholds are 0.01805 and 0.01602. Report the two comparisons separately.
- Use `predict.py` for local inference, `verify_project.py` for parity checks, and `reproduce.py` for fresh training artifacts.
- Preserve the frozen `data/folds.csv`. Describe its small-drug grouping and nine larger groups that can cross folds accurately.
- Do not describe AI-assisted designs or implementations as independent student contributions. Preserve attribution and actual experiment evidence.
- General project development can add or change modules within the user's authorized scope. Additional restrictions on method changes apply only to an explicitly requested strict reproduction of a named paper.
