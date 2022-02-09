import cv2
import argparse
from os import listdir
from os.path import isfile, join

parser = argparse.ArgumentParser()

parser.add_argument("folder_path", help="Path to the folder containing images to analyse")

args = parser.parse_args()

image_folder = args.folder_path

reference_image_path = join(image_folder, "Reference.JPG")
if not isfile(reference_image_path):
    print("Le dossier ne contient pas l'image de référence ! Elle doit être nommée Reference.JPG")
    exit()

image_paths = [join(image_folder, f) for f in listdir(image_folder) if f != "Reference.JPG" and isfile(join(image_folder, f))]

print(reference_image_path)
print(image_paths)
