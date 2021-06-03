from pathlib import Path
import gzip
import shutil

import urllib.request
from urllib.error import URLError
import urllib.parse

MNIST_PATH = Path("data/mnist")

BASE_URL: str = "https://storage.googleapis.com/cvdf-datasets/mnist/"

RESOURCES: list[str] = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz"
]

def download(url: str, destination_path: Path):
    if destination_path.exists():
        return
    else:
        file_name: str = destination_path.name
        print(f"Downloading {file_name}...")
        try:
            urllib.request.urlretrieve(url, destination_path)
        except URLError:
            raise URLError(f"Failed to download {file_name}")

def unzip(file_path: Path):
    unzipped_path = file_path.with_suffix('')
    if (unzipped_path.exists()):
        return

    with gzip.open(file_path, "rb") as zipped_file:
        with open(unzipped_path, "wb") as unzipped_file:
            shutil.copyfileobj(zipped_file, unzipped_file)
            print(f"Unzipped {file_path.name}")

def main():
    if not MNIST_PATH.exists():
        MNIST_PATH.mkdir(parents=True, exist_ok=True)

    try:
        for resource in RESOURCES:
            url: str = urllib.parse.urljoin(BASE_URL, resource)
            path = MNIST_PATH.joinpath(resource)

            download(url, path)
            unzip(path)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
