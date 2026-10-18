"""Regenerate the SYNTHETIC dataset + evaluation. Run: python scripts_generate_synthetic.py"""
import json
from src.config import load_config
from src.evaluation import run_scenario
from src.exporters import evaluation_to_json, write_text
from src.synthetic import mixed_scenario, save_dataset

sc = mixed_scenario()
save_dataset(sc, "data/synthetic_dataset.json")
json.dump(sc.ground_truth, open("data/synthetic_ground_truth.json", "w"), indent=2)
res = run_scenario(sc, load_config(use_user=False))
write_text("docs/evaluation_synthetic.json", evaluation_to_json(res))
for m, v in res["methods"].items():
    print(m, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v["metrics"].items()})
print(res["performance"])
