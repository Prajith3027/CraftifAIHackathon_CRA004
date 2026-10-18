# Testing
Command: `python -m pytest -q` (pytest style) or `python -m unittest discover -s tests -v`.

**Actually executed (build sandbox, Python 3.12.3):** `python -m unittest discover -s tests` → **Ran 44 tests … OK**.
`pytest` was **not installed** in the sandbox (no network), so the pytest runner itself was not run; tests are `unittest.TestCase`
classes that pytest collects.

| Required area | Tests |
|---|---|
| point-in-polygon | `TestZone.test_point_in_polygon`, `test_foot_point_uses_bottom_centre` |
| enter/leave | `TestZone.test_enter_and_leave`, `TestPersistence.test_enter_leave_resolves` |
| persistence | `TestPersistence.*` |
| confidence filtering | `TestConfidence.*` |
| temporary detection loss | `TestMissingDetections.*` |
| dedup | `TestDedup.*` |
| JSON / CSV export | `TestExport.test_json_export`, `test_csv_export`, store tests |
| invalid config | `TestConfig.*`, `TestEngineValidation.*`, invalid zones |
| baseline vs proposed | `TestBaselineVsProposed.*`, `test_synthetic_comparison_is_consistent` |
| pipeline/video/errors | `tests/test_pipeline.py` (generated AVI, missing/corrupt file, missing YOLO) |

**Not verified:** the Streamlit UI was exercised only by a **stubbed-Streamlit smoke harness** (all 7 pages + synthetic live loop executed
without exceptions); real browser rendering, plotly charts, webcam capture, YOLO, and HOG accuracy on real people were **not tested** here.
