"""Parses Amazon product metadata found at http://snap.stanford.edu/data/amazon/productGraph/metadata.json.gz"""

import csv, sys, yaml
from yaml import CLoader as Loader


def usage():
    print("""
USAGE: python parse.py metadata.json
""")
    sys.exit(0)


def main(argv):
    if len(argv) < 2:
        usage()
    filename = sys.argv[1]
    with open(filename, 'r') as f:
        count, good, bad = 0, 0, 0
        out = csv.writer(open("products.csv", "w"))
        for line in f:
            count += 1
            if not (count % 100000):
                print("count:", count, "good:", good, ", bad:", bad)
            if ("'title':" in line) and ("'categories':" in line):
                try:
                    line = line.rstrip().replace("\\'", "''")
                    product = yaml.load(line, Loader=Loader)
                    title, categories = product['title'], product['categories']
                    description = product['description'] if 'description' in product else ''
                    # Append description to title
                    title = title + " " + description if description != '' else title
                    # Handle Images
                    imgURL = product['imUrl']
                    if imgURL == None or "no-img" in imgURL:
                        bad += 1
                    else:
                        category = next(iter(categories), None)
                        category = next(iter(category), None)
                        out.writerow([title, imgURL, category])
                        good += 1
                except Exception as e:
                    print(line)
                    print(e)
                    bad += 1
        print("good:", good, ", bad:", bad)


if __name__ == "__main__":
    main(sys.argv)