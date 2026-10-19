import os
from pathlib import Path
import pandas as pd


def load_highway_dataset(
    year: int | str,
    direction: str,
    base_dir: str | Path | None = None,
    header_row: int = 1,
) -> pd.DataFrame:
    """Loads a traffic Excel dataset based on year and road direction.

    Parameters:
    -----------
    year : int or str
        The target year directory (e.g., 1400, 1401, '1400').
    direction : str
        Direction of the road: 'Karaj_Tehran' (or 'kt') or 'Tehran_Karaj' (or
        'tk').
    base_dir : str, Path, or None
        Root directory containing 'datasets'. If None, resolves relative to
        script/notebook location.
    header_row : int
        Zero-indexed header row location (default is 1 for these specific Excel
        files).

    Returns:
    --------
    pd.DataFrame
        Loaded Pandas DataFrame containing the traffic dataset.
    """
    year_str = str(year).strip()

    # Normalize direction aliases
    dir_clean = direction.strip().lower()
    if dir_clean in ["karaj_tehran", "kt", "karaj-tehran"]:
        normalized_dir = "Karaj_Tehran"
    elif dir_clean in ["tehran_karaj", "tk", "tehran-karaj"]:
        normalized_dir = "Tehran_Karaj"
    else:
        raise ValueError(
            f"Invalid direction '{direction}'. Use 'Karaj_Tehran' or 'Tehran_Karaj'."
        )

    # Determine project root
    if base_dir is None:
        base_path = Path.cwd()
    else:
        base_path = Path(base_dir)

    target_folder = base_path / "datasets" / year_str

    if not target_folder.exists():
        raise FileNotFoundError(
            f"Dataset folder for year '{year_str}' not found at: {target_folder}"
        )

    # Search for matching Excel file pattern (e.g., 1400_7_Karaj_Tehran.xlsx)
    pattern = f"{year_str}_*_{normalized_dir}.xlsx"
    matching_files = list(target_folder.glob(pattern))

    if not matching_files:
        raise FileNotFoundError(
            f"No dataset file matching pattern '{pattern}' inside '{target_folder}'."
        )

    file_path = matching_files[0]
    print(f"Loading dataset from: {file_path}")

    # Read dataset
    df = pd.read_excel(file_path, header=header_row)

    # Standardize common column names to English if present
    column_mapping = {
        "تعداد کل وسیله نقلیه": "total_vehicles",
        "سرعت متوسط": "average_speed",
        "تعداد تخلف سرعت غیر مجاز": "speeding_violations",
        "تعداد تخلف فاصله غیر مجاز": "tailgating_violations",
        "تعداد تخلف سبقت غیر مجاز": "passing_violations",
        "زمان شروع": "start_time",
        "زمان پایان": "end_time",
    }

    df = df.rename(columns=column_mapping)

    # Extract hour feature if start_time exists
    if "start_time" in df.columns:
        df["hour"] = df["start_time"].apply(
            lambda x: int(str(x).split(" ")[1].split(":")[0])
            if pd.notnull(x) and " " in str(x)
            else None
        )

    return df