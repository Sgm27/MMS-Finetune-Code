import requests, time, json, sys
from pathlib import Path

SERVER = "https://finetune-tts.sonktx.online/"

def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_finetune_mms_vie.json>")
        sys.exit(1)

    cfg = Path(sys.argv[1])
    if not cfg.exists():
        print("Không tìm thấy file cấu hình:", cfg)
        sys.exit(1)

    # 1) Tạo job
    with cfg.open("rb") as f:
        r = requests.post(f"{SERVER}/finetune", files={"file": (cfg.name, f, "application/json")}, timeout=60)
    r.raise_for_status()
    print("✓ started")

    # 2) Poll 30s/lần
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
    main()
