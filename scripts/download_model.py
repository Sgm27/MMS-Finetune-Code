from huggingface_hub import snapshot_download

repo_id = "sonktx/mms-tts-vie-train"
local_dir = "./models/ABC/"
snapshot_download(repo_id=repo_id, local_dir=local_dir)