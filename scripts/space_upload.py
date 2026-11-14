"""Create (private) and upload the static TurnGuard demo Space, and add it to the models collection."""

from huggingface_hub import HfApi

REPO = "SujayAgrawal/turnguard-demo"
COLLECTION = "SujayAgrawal/turnguard-models-6af5680cfeefa5c5291959dd"

api = HfApi()
api.create_repo(REPO, repo_type="space", space_sdk="static", private=True, exist_ok=True)
api.upload_folder(
    folder_path="space",
    repo_id=REPO,
    repo_type="space",
    allow_patterns=["index.html", "README.md", "*.png"],
    commit_message="TurnGuard demo: recorded episodes",
)
try:
    api.add_collection_item(COLLECTION, item_id=REPO, item_type="space", exists_ok=True)
except Exception as exc:
    print("collection:", exc)
print(f"https://huggingface.co/spaces/{REPO}")
