from dotenv import load_dotenv
import os
import boto3
import shutil
from typing import List, Optional

def move_selected_local_data_to_old(local_dir: str, base_types: List[str]) -> None:
    """
    Moves older files in local_dir/data to local_dir/old, only if a newer file of the same base type exists.

    Args:
        local_dir (str): The path containing 'data' and 'old' subfolders.
        base_types (List[str]): List of filename base types (e.g., ["OrderDetails"]) to consider for moving.

    Returns:
        None

    Raises:
        None
    """
    data_dir = os.path.join(local_dir, "data")
    old_dir = os.path.join(local_dir, "old")

    if not os.path.exists(data_dir):
        print("No 'data' directory found.")
        return

    if not os.path.exists(old_dir):
        os.makedirs(old_dir)

    files = os.listdir(data_dir)
    for base in base_types:
        base_files = [f for f in files if f.startswith(base + "_") and f.endswith(".csv")]
        if len(base_files) < 2:
            continue

        def extract_start_date(f: str) -> str:
            parts = f[len(base) + 1:].split('-')
            if parts and len(parts[0]) >= 10:
                return parts[0]
            return '0000_00_00'

        base_files_sorted = sorted(base_files, key=extract_start_date, reverse=True)
        newest_file = base_files_sorted[0]
        for old_file in base_files_sorted[1:]:
            src = os.path.join(data_dir, old_file)
            dest = os.path.join(old_dir, old_file)
            counter = 1
            base_dest = dest
            while os.path.exists(dest):
                filename, ext = os.path.splitext(old_file)
                dest = os.path.join(old_dir, f"{filename}_{counter}{ext}")
                counter += 1
            shutil.move(src, dest)
            print(f"Moved old file: {old_file} to {dest}")

def download_selected_folders(prefixes: List[str], local_dir: str, bucket) -> List[str]:
    """
    Downloads files from specified S3 prefixes into a local directory,
    maintaining subdirectory structure and returning any errors encountered.

    Args:
        prefixes (List[str]): List of data prefixes to fetch from S3.
        local_dir (str): Local directory where data should be saved.
        bucket: Boto3 S3 Bucket resource to download from.

    Returns:
        List[str]: List of paths of files that failed to download.

    Raises:
        None (exceptions are caught internally and appended to results)
    """
    os.makedirs(local_dir, exist_ok=True)
    errors: List[str] = []
    for prefix in prefixes:
        print(f"\nFiles to download from '{prefix}':")
        for obj in bucket.objects.filter(Prefix=prefix):
            if obj.key.endswith("/"):
                continue
            relative_path = os.path.relpath(obj.key, start=prefix)
            target_path = os.path.join(local_dir, prefix, relative_path)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            print(f"- {obj.key}")
            try:
                bucket.download_file(obj.key, target_path)
                print(f"Downloaded: {obj.key} to {target_path}")
            except Exception as e:
                print(f"Error downloading {obj.key}: {e}")
                errors.append(obj.key)
    return errors

def pull_data() -> None:
    """
    Downloads selected folders from S3, moves older files to archive folder if needed,
    and handles any errors or critical failures during the process.

    Args:
        None

    Returns:
        None

    Raises:
        None (all errors are handled and logged internally)
    """
    local_dir: str = './data_handling'
    load_dotenv()
    aws_access_key_id: Optional[str] = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key: Optional[str] = os.getenv('AWS_SECRET_ACCESS_KEY')
    region: Optional[str] = os.getenv('AWS_DEFAULT_REGION')
    bucket_name: str = "staffocaster-somebodypeople-bucket"
    session: boto3.Session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region
    )
    s3 = session.resource('s3')
    bucket = s3.Bucket(bucket_name)

    try:
        errors: List[str] = download_selected_folders(["data/", "preds/"], local_dir, bucket)
        if not errors:
            file_types = ["OrderDetails", "Shifts_Closed", "sp_hourly_sales", "TimeEntries"]
            move_selected_local_data_to_old(local_dir, file_types)
            print("All data files downloaded and old files moved.")
        else:
            print("Some files failed to download:", errors)
    except Exception as e:
        print("Critical process failure:", e)

if __name__ == "__main__":
    print('running')
    pull_data()
