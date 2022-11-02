"""Utilities for reading, writing, and downloading files."""
import shutil
from pathlib import Path

from src.utils.typing import PathLike

import requests
from tqdm.auto import tqdm


def download_file(
    url: str,
    to: PathLike,
    overwrite: bool = True,
    progress: bool = True,
) -> None:
    """Download the file to the given path.

    Args:
        url: The URL to download from.
        to: The path to download to.
        overwrite: If set, overwrite if it already exists.
            Otherwise do not redownload. Defaults to True.
        progress: Display download progress.

    Returns:
        True if the file was downloaded.

    """
    to = Path(to)
    if to.is_file() and not overwrite:
        return

    to.parent.mkdir(exist_ok=True, parents=True)

    response = requests.get(url, stream=True, allow_redirects=True)
    response.raise_for_status()

    if progress:
        file_size = int(response.headers.get("Content-Length", 0))
        with tqdm.wrapattr(
            response.raw, "read", total=file_size, desc=f"download {to}"
        ) as raw:
            with to.open("wb") as handle:
                shutil.copyfileobj(raw, handle)
    else:
        with to.open("wb") as handle:
            handle.write(response.content)
