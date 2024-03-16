import pandas as pd
import os

def process_csv(csv_file):
    df = pd.read_csv(csv_file, names=["name", "imageurl", "category"])

    rows_to_delete = []
    for index, row in df.iterrows():
        if row['imageurl'].lower().endswith('.gif') or row['imageurl'].lower().endswith('.png'):
            rows_to_delete.append(index)
    print(len(rows_to_delete))
    if rows_to_delete:
        df.drop(rows_to_delete, inplace=True)
        output_csv_file = os.path.splitext(csv_file)[0] + "_filtered.csv"
        df.to_csv(output_csv_file, index=False)
        print(f"Filtered CSV saved to {output_csv_file}")

if __name__ == "__main__":
    csv_file = r'C:\Users\kyley\Desktop\CS\C147\C147 Amazon Classification\products_uniform_clean.csv'  # Change this to your CSV file name
    process_csv(csv_file)
