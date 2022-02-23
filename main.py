import cv2
import argparse
from os import listdir
from os.path import basename
from os.path import join
import detect_changes


def get_image_paths(image_folder):
    all_image_path = list()
    rooms = listdir(image_folder)
    for room in rooms:
        list_filename = listdir(f"{image_folder}\\{room}")
        if not 'Reference.JPG' in list_filename:
            print("Le dossier {} ne contient pas l'image de référence ! Elle doit être nommée Reference.JPG".format(room))
            exit()
        all_image_path.append((room, [join(f"{image_folder}\\{room}", filename) for filename in list_filename]))
    return all_image_path


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("images_path", help="Path to the folder containing folders of images to analyse")
args = parser.parse_args()

# Get images and detect changes
images_folder = args.images_path

image_paths = get_image_paths(images_folder)

for room_tuple in image_paths:
    room = room_tuple[0]
    room_image_paths = room_tuple[1]

    ref_img_path = [image_path for image_path in room_image_paths if basename(image_path) == "Reference.JPG"][0]
    room_image_paths.remove(ref_img_path)

    for room_image_path in room_image_paths:
        img_with_detected_objects = detect_changes.detect_changes(ref_img_path, room_image_path)
        cv2.imshow(f"{room} avec objets detectes - {basename(room_image_path)}", img_with_detected_objects)

        if detect_changes.DEBUG:
            cv2.waitKey(0)

cv2.waitKey(0)
cv2.destroyAllWindows()
