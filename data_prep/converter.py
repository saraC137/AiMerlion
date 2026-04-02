import csv
import json
import argparse
import os

def csv_to_json(csv_filepath, json_filepath):
    """
    Converts a CSV file to a JSON file (list of objects).
    The CSV header row is used as the keys for the JSON objects.
    """
    # Check if the input file exists
    if not os.path.exists(csv_filepath):
        print(f"❌ Error: Input CSV file not found at '{csv_filepath}'")
        return

    data = []
    
    try:
        # Open the CSV file for reading
        with open(csv_filepath, mode='r', encoding='utf-8', newline='') as csv_file:
            # Use csv.DictReader to read the data as a dictionary per row
            csv_reader = csv.DictReader(csv_file)
            
            for row in csv_reader:
                data.append(row)

    except Exception as e:
        print(f"❌ An error occurred while reading the CSV file: {e}")
        return

    try:
        # Open the JSON file for writing
        with open(json_filepath, mode='w', encoding='utf-8') as json_file:
            # Write the Python data structure to the file, with pretty printing (indent=4)
            json.dump(data, json_file, indent=4)
        
        print(f"✅ Conversion successful!")
        print(f"📦 Data converted from '{csv_filepath}' to '{json_filepath}'")

    except Exception as e:
        print(f"❌ An error occurred while writing the JSON file: {e}")


if __name__ == '__main__':
    # 1. Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Convert a CSV file to a JSON file (list of objects).",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # 2. Define the expected arguments
    parser.add_argument(
        'csv_file', 
        type=str, 
        help="The path to the input CSV file."
    )
    
    parser.add_argument(
        'json_file', 
        type=str, 
        help="The path where the output JSON file will be saved."
    )

    # 3. Parse the arguments provided by the user
    args = parser.parse_args()

    # 4. Call the main conversion function with the provided file paths
    csv_to_json(args.csv_file, args.json_file)