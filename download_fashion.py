import gzip
import concurrent.futures
import urllib.request
import urllib.parse
from io import BytesIO
from pathlib import Path

FASHION_PATH = Path("./data/fashion")

BASE_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"

RESOURCES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz"
]

def download(url, destination):
    if destination.exists():
        print(f"{destination} already exists, skipping...")
        return

    response = urllib.request.urlopen(url)

    with BytesIO(response.read()) as compressed:
        with gzip.GzipFile(fileobj=compressed) as uncompressed:
            with open(destination, "wb") as output:
                output.write(uncompressed.read())

def main():
    if not FASHION_PATH.is_dir():
        FASHION_PATH.mkdir(parents=True, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for resource in RESOURCES:
            url = urllib.parse.urljoin(BASE_URL, resource)
            destination = FASHION_PATH.joinpath(resource).with_suffix("")
            executor.submit(download, url, destination)

if __name__ == "__main__":
    main()
