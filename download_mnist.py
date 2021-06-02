from pathlib import Path
from urllib.parse import urljoin
from urllib.request import urlretrieve
from urllib.error import URLError
import gzip
import shutil

PATH = Path("data/mnist")
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
        print(f"Downloading {destination_path}...")
        try:
            urlretrieve(url, destination_path)
        except URLError:
            raise URLError("Failed to download resource")

def unzip(file_path: Path):
    unzipped_path = file_path.with_suffix('')
    if (unzipped_path.exists()):
        return

    with gzip.open(file_path, "rb") as zipped_file:
        with open(unzipped_path, "wb") as unzipped_file:
            print(f"Unzipping {file_path}...")
            shutil.copyfileobj(zipped_file, unzipped_file)

def main():
    if not PATH.exists():
        PATH.mkdir(parents=True, exist_ok=True)

    for resource in RESOURCES:
        url: str = urljoin(BASE_URL, resource)
        path = PATH.joinpath(resource)

        download(url, path)
        unzip(path)

if __name__ == "__main__":
    main()
