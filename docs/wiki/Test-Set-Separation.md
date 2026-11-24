# Test set separation

The test set is only our own recordings of 1–2 participants who are in no training or validation data. The code enforces this:

1. Splits are by participant, so all reps of one person land in one split (`formcoach/splits.py`).
2. Public-data participants (prefixes `mdl`, `zsq`, `faq`, `pub`, `reh`) can only go to train or val. Pinning one to test raises an error.
3. `02_make_splits.py` stops if any public rep ends up in test.
4. Model B situations take the split of the rep they came from, so its test set has the same people.
5. `10_data_report.py` prints the test participants, any person shared between test and train/val (must be "none") and the count of public reps in test (must be 0). Paste this into the report.
6. Thresholds, epochs and prompts are picked on val only. Test is run once at the end.

`tests/test_core.py::test_public_data_never_in_test` checks rule 2.
