import csv
import sys
import os

import urllib.request 
from PIL import Image 
  
# Retrieving the resource located at the URL 
# and storing it in the file name a.png 
in_file = sys.argv[1]
with open(in_file, 'r') as f:
    reader = csv.reader(f)
    count = 0
    for row in reader:
        if row:
            url = row[1]
            dir = "./img/" + row[2]
            if not(os.path.isdir(dir)):
                os.mkdir(dir)
            urllib.request.urlretrieve(url, dir + "/" + str(count) + ".jpg")
            count += 1