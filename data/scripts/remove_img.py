import csv
import sys
  
csv.field_size_limit(2147483647)
with open("products.normalized.csv", "r") as source: 
    reader = csv.reader(source) 
    with open("products_noimg.csv", "w", newline='') as result: 
        writer = csv.writer(result) 
        for r in reader: 
            if r:
                writer.writerow((r[0], r[2]))