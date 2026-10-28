"""Check a running local or Docker API against the saved Colab prediction."""
import argparse
import json
import math
from pathlib import Path
from urllib.request import Request, urlopen

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8000")
args = parser.parse_args()
artifacts = Path(__file__).resolve().parent / "artifacts"
sample = json.loads((artifacts / "sample_request.json").read_text(encoding="utf-8-sig"))
expected = json.loads((artifacts / "expected_prediction.json").read_text(encoding="utf-8-sig"))

request = Request(
    args.url.rstrip("/") + "/predict",
    data=json.dumps(sample, allow_nan=False).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urlopen(request, timeout=30) as response:
    result = json.load(response)

actual = result["predicted_quality"]
reference = expected["predicted_quality"]
if not math.isclose(actual, reference, rel_tol=1e-8, abs_tol=1e-8):
    raise SystemExit(f"Mismatch: API={actual}, Colab={reference}")
print(f"PASS: API prediction {actual} matches saved Colab prediction {reference}.")
