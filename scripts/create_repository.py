from huggingface_hub import create_repo
import os
from dotenv import load_dotenv

load_dotenv(override=True)
HF_TOKEN = os.getenv("HF_TOKEN")

owner = "sonktx"
repo_name = "finetune-hf-vits-abc"
url = create_repo(
    repo_id=f"{owner}/{repo_name}",
    token=HF_TOKEN,
    private=False,
    exist_ok=True,
    repo_type="model",
)

print(f"Repository created at {url}")