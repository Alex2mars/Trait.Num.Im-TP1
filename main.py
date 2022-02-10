import cv2
import argparse
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

from detect_changes import *

parser = argparse.ArgumentParser()

parser.add_argument("folder_path", help="Path to the folder containing images to analyse")

args = parser.parse_args()

image_folder = args.folder_path

reference_image_path = join(image_folder, "Reference.JPG")
if not isfile(reference_image_path):
    print("Le dossier ne contient pas l'image de référence ! Elle doit être nommée Reference.JPG")
    exit()

image_paths = [join(image_folder, f) for f in listdir(image_folder) if f != "Reference.JPG" and isfile(join(image_folder, f))]

reference_image = cv2.imread(reference_image_path)
images = [cv2.imread(f) for f in image_paths]

def show_with_matplotlib(img, title="Image"):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB:
    img_RGB = img[:, :, ::-1]

    # Show the image using matplotlib:
    plt.imshow(img_RGB)
    plt.title(title)
    plt.show()


n = 1
print(reference_image_path, image_paths[n])
detect_changes(reference_image, images[n])
