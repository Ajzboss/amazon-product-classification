import pandas as pd
import os
import requests
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def download_image(image_url):
    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
        return True
    except Exception as e:
        # print(f"Error downloading image from {image_url}")
        return False

def process_row(row):
    image_url = row['imageurl']
    if download_image(image_url):
        return row
    else:
        return None

def process_csv(csv_file):
    df = pd.read_csv(csv_file, names=["name","imageurl","category"])

    rows_to_delete = []
    with ThreadPoolExecutor() as executor:  # Adjust max_workers as needed
        results = list(tqdm(executor.map(process_row, df.to_dict('records')), total=len(df)))
        
    for idx, result in enumerate(results):
        if result is None:
            rows_to_delete.append(idx)

    if rows_to_delete:
        df.drop(rows_to_delete, inplace=True)
        df.to_csv(r'C:\Users\kyley\Desktop\CS\C147\C147 Amazon Classification\products_uniform_clean.csv', index=False)

if __name__ == "__main__":
    csv_file = r"C:\Users\kyley\Desktop\CS\C147\C147 Amazon Classification\products_uniform.csv"  # Change this to your CSV file name
    process_csv(csv_file)