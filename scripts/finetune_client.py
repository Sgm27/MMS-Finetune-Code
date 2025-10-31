import requests, time, json, sys
from pathlib import Path
from huggingface_hub import create_repo
import os
from dotenv import load_dotenv

load_dotenv(override=True)
HF_TOKEN = os.getenv("HF_TOKEN")

SERVER = "https://finetune-tts.sonktx.online/"

def create_repository(run_id: str):
    owner = "sonktx"
    repo_name = f"vits-finetuned-vie-{run_id}"
    url = create_repo(
        repo_id=f"{owner}/{repo_name}",
        token=HF_TOKEN,
        private=False,
        exist_ok=True,
        repo_type="model",
    )
    print(f"Repository created at {url}")

def finetune(run_id :str):
    with open("./finetune_mms_vie.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["push_to_hub"] = True
    cfg["hub_model_id"] = f"sonktx/vits-finetuned-vie-{run_id}"
    cfg["dataset_config_name"] = run_id
    cfg["num_train_epochs"] = 5
    with open(f"./finetune_mms_vie_{run_id}.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    
    with cfg.open("rb") as f:
        r = requests.post(f"{SERVER}/finetune", files={"file": (cfg.name, f, "application/json")}, timeout=60)
    r.raise_for_status()
    print("✓ started")

    while True:
        s = requests.get(f"{SERVER}/finetune/status", timeout=30).json()
        print("status:", s.get("status"))

        if s.get("status") == "completed":
            metrics = s.get("metrics", {})
            Path("training_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            print("🎉 done. saved -> training_metrics.json")
            break
        if s.get("status") == "failed":
            print("❌ failed:", s.get("error"))
            sys.exit(1)

        time.sleep(30)

if __name__ == "__main__":
    finetune()
