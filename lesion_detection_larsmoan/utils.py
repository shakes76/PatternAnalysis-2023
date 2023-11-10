import os
from pathlib import Path
from dotenv import load_dotenv

# Note: you need a .env file in the root of your repository with REPO_DATA_DIR: "path" for this method to work
def get_data_dir():
    load_dotenv()
    dir_name = "REPO_DATA_DIR"
    if dir_name in os.environ:
        path = os.getenv(dir_name)
    else:
        return Exception(
            "No data directory found in the .env file or missing a .env file in the repo's root"
        )
    p = Path(path)
    return p


if __name__ == "__main__":
    print(get_data_dir())
