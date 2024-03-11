"""Parses Amazon product metadata found at http://snap.stanford.edu/data/amazon/productGraph/metadata.json.gz"""

import csv, sys, yaml
from yaml import CLoader as Loader


def usage():
    print("""
USAGE: python parse.py metadata.json
""")
    sys.exit(0)

top_categories = ['books', 
                  'clothing, shoes & jewelry', 
                  'sports & outdoors', 
                  'electronics', 
                  'toys & games', 
                  'health & personal care',
                  'grocery & gourmet food',
                  'musical instruments',
                  'arts, crafts & sewing',
                  'patio, lawn & garden']

def main(argv):
    if len(argv) < 2:
        usage()
    filename = sys.argv[1]
    with open(filename, 'r') as f:
        count, good, bad, not_top = 0, 0, 0, 0
        out = csv.writer(open("products.csv", "w", newline=''))
        for line in f:
            count += 1
            if not (count % 100000):
                print("count:", count, "good:", good, ", bad:", bad, "not top:", not_top)
            if ("'title':" in line) and ("'categories':" in line):
                try:
                    line = line.rstrip().replace("\\'", "''")
                    product = yaml.load(line, Loader=Loader)
                    title, categories = product['title'], product['categories']
                    description = product['description'] if 'description' in product else ''
                    title = title + " " + description
                    # categories = ' / '.join([item for sublist in categories for item in sublist])
                    imgURL = product['imUrl']
                    category = next(iter(categories), None)
                    category = next(iter(category), None)
                    # print([title, imgURL, category])
                    
                    if any(category.casefold() in top_category.casefold() for top_category in top_categories) and category != '':
                        if imgURL and not("no-img" in imgURL):
                            out.writerow([title, imgURL, category])
                            good+=1
                        else:
                            bad+=1
                    else:
                        not_top+=1
                except Exception as e:
                    # print(line)
                    # print(e)
                    bad += 1
        print("good:", good, ", bad:", bad, "not top:", not_top)


if __name__ == "__main__":
    main(sys.argv)