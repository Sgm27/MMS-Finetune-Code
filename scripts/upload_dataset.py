from huggingface_hub import login
from datasets import Dataset, DatasetDict, Audio
from dotenv import load_dotenv
load_dotenv(override=True)
import os, glob, random

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(HF_TOKEN) 

repo_id = "sonktx/httm_v2"        
config_name = "female"        
AUDIO_DIR = "../dataset/audio_16k/"
TXT_DIR   = "../dataset/transcripts/"
SR = 16000

def collect_rows(audio_dir, txt_dir, speaker_tag):
    rows = []
    audio_paths = sorted(
        glob.glob(os.path.join(audio_dir, "**", "*.*"), recursive=True)
    )
    exts = {".wav", ".flac", ".mp3"}
    for ap in audio_paths:
        ext = os.path.splitext(ap)[1].lower()
        if ext not in exts:
            continue
        stem = os.path.splitext(os.path.basename(ap))[0]
        tp = os.path.join(txt_dir, stem + ".txt")
        if not os.path.exists(tp):
            print(f"[WARN] Missing transcript for: {ap}")
            continue
        with open(tp, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            print(f"[WARN] Empty transcript: {tp}")
            continue
        rows.append({"audio": ap, "sentence": text, "speaker_id": speaker_tag})
    return rows

rows = collect_rows(AUDIO_DIR, TXT_DIR, config_name)
if len(rows) == 0:
    raise SystemExit("No (audio, transcript) pairs found. Check your folders.")

# ===== 3) Shuffle + split
random.shuffle(rows)
split = int(0.95 * len(rows)) if len(rows) > 20 else max(1, int(0.9 * len(rows)))
train_rows = rows[:split]
val_rows   = rows[split:] or rows[: max(1, len(rows)//10)]

def to_ds(items):
    ds = Dataset.from_list(items)
    ds = ds.cast_column("audio", Audio(sampling_rate=SR))
    return ds

dset = DatasetDict(
    train=to_ds(train_rows),
    validation=to_ds(val_rows),
)
print(dset)

# ===== 4) Push lên Hub (dùng config_name để tạo subset "female")
#    Sau này load bằng: load_dataset("sonktx/httm_v2", "female")
dset.push_to_hub(
    repo_id=repo_id,
    config_name=config_name,      # tên subset
    private=True,                 # đổi False nếu muốn công khai
    commit_message=f"Initial upload ({config_name})",
    max_shard_size="500MB"
)

print(f"✅ Done. Load lại bằng: load_dataset('{repo_id}', '{config_name}')")
