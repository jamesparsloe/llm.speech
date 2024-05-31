from huggingface_hub import HfApi
import sys
import os

path = sys.argv[1]
filename = os.path.basename(path)

api = HfApi()
api.upload_file(
    path_or_fileobj=path,
    path_in_repo=filename,
    repo_id="jamesparsloe/llm.speech",
)
