import argparse
import requests
import os

def download_data(url: str, output_path: str) -> None:
    """Download dataset from a URL and save it to the given path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    response = requests.get(url)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)

def main():
    parser = argparse.ArgumentParser(description="Download the Twitter airline sentiment dataset")
    parser.add_argument("--url", type=str, default="https://raw.githubusercontent.com/mohamadrazzimy/datasets/main/airline-tweets.csv", help="URL to the dataset CSV")
    parser.add_argument("--output", type=str, default="data/raw/airline_tweets.csv", help="Output path for downloaded CSV")
    args = parser.parse_args()
    download_data(args.url, args.output)
    print(f"Dataset downloaded to {args.output}")

if __name__ == "__main__":
    main()
