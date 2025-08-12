from huggingface_hub import HfApi
api = HfApi()

# Upload all the content from the local folder to your remote Space.
# By default, files are uploaded at the root of the repo
api.upload_large_folder(
    folder_path="/u501/x25luo/codebase/probingLM/ckpt/native_lens/output/seed42",
    repo_id="Luoxiaoxi/GPT2-native-lens-CHILDS-seed42",
    repo_type="model",
)