from struct import unpack
from tqdm import tqdm
from pathlib import Path
import os


marker_mapping = {
    0xffd8: "Start of Image",
    0xffe0: "Application Default Header",
    0xffdb: "Quantization Table",
    0xffc0: "Start of Frame",
    0xffc4: "Define Huffman Table",
    0xffda: "Start of Scan",
    0xffd9: "End of Image"
}


class JPEG:
    def __init__(self, image_file):
        with open(image_file, 'rb') as f:
            self.img_data = f.read()
    
    def decode(self):
        data = self.img_data
        while(True):
            marker, = unpack(">H", data[0:2])
            # print(marker_mapping.get(marker))
            if marker == 0xffd8:
                data = data[2:]
            elif marker == 0xffd9:
                return
            elif marker == 0xffda:
                data = data[-2:]
            else:
                lenchunk, = unpack(">H", data[2:4])
                data = data[2+lenchunk:]            
            if len(data)==0:
                break        


bads = []

pathlist = Path(r"C:/Users/kyley/Desktop/CS/C147/C147 Amazon Classification/img_uniform/").glob(r'**/*.jpg')
for path in tqdm(list(pathlist)):
    # because path is object not string
    path_in_str = str(path)   
    image = JPEG(path_in_str)
    try:
        image.decode()   
    except:
        bads.append(path_in_str)

# for img in tqdm(images):
#   image = osp.join(root_img,img)
#   image = JPEG(image) 
#   try:
#     image.decode()   
#   except:
#     bads.append(img)


for name in bads:
  print(name)