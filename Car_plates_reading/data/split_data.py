import splitfolders
from pathlib import Path

def split_data(directory_path: str, yolo_dir: str) -> None:

    print("Splitting data...")
    input_folder = Path(directory_path)
    output_folder = Path(yolo_dir) / "data" / "plates_detection"

    splitfolders.ratio(
        input_folder,
        output=output_folder,
        seed=42,
        ratio=(0.8, 0.2),
        group_prefix=None
    )

    print("Moving files finished.")