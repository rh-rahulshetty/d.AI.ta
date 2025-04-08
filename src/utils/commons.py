import os
import tempfile
from urllib.request import urlretrieve


def download_file(url: str) -> str:
    """
    Downloads a file from the given URL and saves it locally.
    :param url: The URL of the file to download.
    """
    try:
        filename = os.path.join(
            tempfile.mkdtemp(),
            os.path.basename(url)
        )
        urlretrieve(url, filename)
        return filename
    except Exception as e:
        print(f"Failed to download the file: {e}")
