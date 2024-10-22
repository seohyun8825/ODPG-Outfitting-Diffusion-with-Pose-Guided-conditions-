import csv
# add delimiter=':' if fieldname is divided by :

# Define the path to your input and output CSV files
input_csv_file_path = 'fashion/fashion-resize-pairs-train-unmasked.csv'
output_csv_file_path = 'fashion/fashion-resize-pairs-train.csv'


# Read the input CSV file and process each row
with open(input_csv_file_path, mode='r', newline='') as input_file:
    reader = csv.DictReader(input_file)
    fieldnames = reader.fieldnames
    print(fieldnames)

    # Prepare to write to the output CSV file
    with open(output_csv_file_path, mode='w', newline='') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            from_file = row['from']
            # Add '_masked' before the file extension
            from_file_masked = from_file.rsplit('.', 1)
            from_file_masked = f"{from_file_masked[0]}_masked.{from_file_masked[1]}"
            row['from'] = from_file_masked

            # Write the modified row to the new CSV file
            writer.writerow(row)

print(f"Processed file saved as {output_csv_file_path}")
