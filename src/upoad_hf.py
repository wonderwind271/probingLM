from huggingface_hub import HfApi
api = HfApi()

# Upload all the content from the local folder to your remote Space.
# By default, files are uploaded at the root of the repo
api.upload_large_folder(
    folder_path="/u501/x25luo/codebase/probingLM/ckpt/attached_lens",
    repo_id="Luoxiaoxi/GPT2-attached-lens",
    repo_type="model",
)

# ------
from datasets import load_dataset

ds = load_dataset("wikimedia/wikipedia", "20231101.en")['train']
from IPython import embed;embed()