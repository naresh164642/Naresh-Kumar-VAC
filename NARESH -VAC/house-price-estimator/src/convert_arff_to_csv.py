"""
Convert Ames Housing ARFF file to clean raw CSV using only standard library.
"""

import csv
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARFF_PATH = DATA_DIR / "ames_housing.arff"
CSV_PATH = DATA_DIR / "ames_housing_raw.csv"

def convert_arff_to_csv():
    if not ARFF_PATH.exists():
        print(f"ARFF file not found: {ARFF_PATH}")
        return

    print("Parsing ARFF header...")
    attributes = []
    with open(ARFF_PATH, "r", encoding="utf-8") as f:
        in_data = False
        data_rows = []
        for line in f:
            line_str = line.strip()
            if not line_str or line_str.startswith("%"):
                continue

            if not in_data:
                if line_str.lower().startswith("@attribute"):
                    # Extract attribute name
                    match = re.match(r"@attribute\s+['\"]?([^'\"\s]+)['\"]?", line_str, re.IGNORECASE)
                    if match:
                        attributes.append(match.group(1))
                elif line_str.lower().startswith("@data"):
                    in_data = True
            else:
                # In data section: parse CSV row
                data_rows.append(line_str)

    print(f"Extracted {len(attributes)} attributes and {len(data_rows)} data rows.")

    # Write CSV
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(attributes)
        csv_reader = csv.reader(data_rows)
        for row in csv_reader:
            writer.writerow(row)

    print(f"Successfully converted ARFF to CSV: {CSV_PATH} ({len(data_rows)} records)")

if __name__ == "__main__":
    convert_arff_to_csv()
