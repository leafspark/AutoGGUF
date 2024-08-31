from huggingface_hub import HfApi, snapshot_download, hf_hub_url


class HFTransfer:
    def __init__(self, token) -> None:
        self.api = HfApi(token=token)

    def upload_file(self, path, remote_path, repo, repo_type) -> str:
        status = self.api.upload_file(
            path_or_fileobj=path,
            path_in_repo=remote_path,
            repo_id=repo,
            repo_type=repo_type,
        )
        return status

    def upload_folder(self, local_folder, remote_path, repo, repo_type) -> str:
        status = self.api.upload_folder(
            folder_path=local_folder,
            path_in_repo=remote_path,
            repo_id=repo,
            repo_type=repo_type,
        )
        return status

    def download_repo(self, repo) -> str:
        return snapshot_download(repo_id=repo)

    def get_download_link(self, repo, file) -> str:
        return hf_hub_url(repo_id=repo, filename=file)
