import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def pytest_report_header(config):
    n = sum(os.path.isfile(os.path.join(DATA_DIR, d, "hand_boxes.json"))
            for d in (os.listdir(DATA_DIR) if os.path.isdir(DATA_DIR) else []))
    if n == 0:
        return ("WARNING: no clips under data/ -- real-data tests will fail. "
                "Run `python scripts/download_data.py` first (see README).")
    return f"real-data clips found under data/: {n}"
