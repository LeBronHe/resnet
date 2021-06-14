import gzip
import os
from pathlib import Path
import concurrent.futures
import urllib.request

FASHION_PATH = Path("./data/fashion")

URLS = [
    "https://github.com/zalandoresearch/fashion-mnist/blob/master/data/fashion/train-images-idx3-ubyte.gz",
    "https://github.com/zalandoresearch/fashion-mnist/blob/master/data/fashion/train-labels-idx1-ubyte.gz",
    "https://github.com/zalandoresearch/fashion-mnist/blob/master/data/fashion/t10k-images-idx3-ubyte.gz",
    "https://github.com/zalandoresearch/fashion-mnist/blob/master/data/fashion/t10k-labels-idx1-ubyte.gz"
]

def download(url, destination):
    if destination.exists():
        print(f"{destination} already exists, skipping...")
        return

    with urllib.request.urlopen(url) as response:
        with open(destination, "wb") as output:
            output.write(gzip.decompress(response.read()))

def main():
    if not FASHION_PATH.is_dir():
        FASHION_PATH.mkdir(parents=True, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for url in URLS:
            destination = FASHION_PATH.joinpath(os.path.basename(url)).with_suffix("")
            executor.submit(download, url, destination)

if __name__ == "__main__":
    main()
