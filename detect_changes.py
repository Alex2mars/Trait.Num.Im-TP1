from time import sleep

import cv2
import numpy as np

DEBUG = True
FLOOR_RATIO = 0.3

def detect_changes(ref_img_path, img_path, room, img_name):
    # Load reference image and image to analyse changes
    ref_img = cv2.imread(ref_img_path)
    new_img = cv2.imread(img_path)
    # Resize the two images (no need for a large image, longer to compute with no additional information for detection)
    ref_img = cv2.resize(ref_img, (800, 400))
    new_img = cv2.resize(new_img, (800, 400))

    # === FLOOR DETECTION ===

    # Ref image to gray
    floor_ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

    # TRYING FIRST METHOD TO DETECT FLOOR
    # Bilateral filter to smoothen and keep edges
    bilateral_img = cv2.bilateralFilter(floor_ref_gray, 3, 75, 75)
    # Gaussian blur to increase edge detection
    gaussian_blur = cv2.GaussianBlur(bilateral_img, (3, 3), 0, sigmaY=0)

    # To check the result of the filter
    debug_image("Gaussian blur floor detection", gaussian_blur)

    # First method uses Canny algorithm to detect edges
    floor_edge_detected = cv2.Canny(gaussian_blur, 40, 200)
    # After we use a dilatation + erosion to keep edges without noise, anb join edges if not totally joined
    floor_edge_detected = cv2.dilate(floor_edge_detected, build_morph_kernel(cv2.MORPH_RECT, (16, 16)))
    floor_edge_detected = cv2.erode(floor_edge_detected, build_morph_kernel(cv2.MORPH_RECT, (8, 8)))
    # Threshold only used to revert white and blacks (to create mask in the future)
    _, floor_edge_detected = cv2.threshold(floor_edge_detected, 128, 255, cv2.THRESH_BINARY_INV)

    # Find the floor contour
    contours, _ = cv2.findContours(floor_edge_detected, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_floor = list(sorted(contours, key=cv2.contourArea, reverse=True))[0]

    # Calculate the space that floor is taking with the firsty method
    total_image_pixels = ref_img.shape[0] * ref_img.shape[1]
    floor_pixels = cv2.contourArea(contour_floor)

    # If ratio < FLOOR_RATIO (0.3), then we use the second method
    if floor_pixels / float(total_image_pixels) < FLOOR_RATIO:
        # SECOND METHOD LESS SENSIBLE TO DETECT FLOOR, IF THE FIRST ONE IS NOT GREAT FOIR THE IMAGE

        # Firs a gaussian blur to enhance edge detection
        gaussian_blur = cv2.GaussianBlur(floor_ref_gray, (3, 3), 0, sigmaY=0)

        debug_image("Gaussian blur floor detection", gaussian_blur)

        # Edge detection with canny algorithm - Different parameters
        floor_edge_detected = cv2.Canny(gaussian_blur, 160, 230)
        # Dilate and erode to join edges and remove noise
        floor_edge_detected = cv2.dilate(floor_edge_detected, build_morph_kernel(cv2.MORPH_RECT, (16, 16)))
        floor_edge_detected = cv2.erode(floor_edge_detected, build_morph_kernel(cv2.MORPH_RECT, (8, 8)))
        # Threshold only to invert black and white
        _, floor_edge_detected = cv2.threshold(floor_edge_detected, 128, 255, cv2.THRESH_BINARY_INV)

        # Define the floor contour with the new method
        contours, _ = cv2.findContours(floor_edge_detected, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour_floor = list(sorted(contours, key=cv2.contourArea, reverse=True))[0]

    cv2.drawContours(floor_edge_detected, [contour_floor], -1, 128, thickness=20)
    debug_image("Detection du sol - En gris", np.concatenate((floor_edge_detected, floor_ref_gray), axis=1))

    # === ENF OF FLOOR DETECTION - START CHANGES DETECTION ===

    # Once the floor contour is detected, we create a mask to apply to the image -> Only detect on the floor
    mask = np.zeros((ref_img.shape[0], ref_img.shape[1]), dtype="uint8")
    # Fill the contour in white, to create the mask
    mask = cv2.fillPoly(mask, [contour_floor], 255)

    # We apply the mask to the reference image and the treated image (to both image to have a correct diff later)
    masked_ref_img = cv2.bitwise_and(ref_img, ref_img, mask=mask)
    masked_new_img = cv2.bitwise_and(new_img, new_img, mask=mask)

    debug_image("Images masquees", np.concatenate((masked_ref_img, masked_new_img), axis=1))

    # We use bilateral filter on both images to remove noise but keep edges
    bilateral_ref_img = cv2.bilateralFilter(masked_ref_img, 5, 40, 115)
    bilateral_new_img = cv2.bilateralFilter(masked_new_img, 5, 40, 115)

    debug_image("Images avec filtre bilateral", np.concatenate((bilateral_ref_img, bilateral_new_img), axis=1))

    # Images to gray
    gray_ref_img = cv2.cvtColor(bilateral_ref_img, cv2.COLOR_BGR2GRAY)
    gray_new_img = cv2.cvtColor(bilateral_new_img, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold to highlight objects
    thresh_ref_img = cv2.adaptiveThreshold(gray_ref_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 3)
    thresh_new_img = cv2.adaptiveThreshold(gray_new_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 3)

    debug_image("Images avec threshold adaptatif par moyenne", np.concatenate((thresh_ref_img, thresh_new_img), axis=1))

    # We get the difference between thresholded images to only get new objects
    difference_img = cv2.subtract(thresh_ref_img, thresh_new_img)

    debug_image("Difference entre les deux images thresholdees", difference_img)

    # Erode and dilate to remove noise (white points) and close edges for objects
    img_reformed_edges = cv2.erode(difference_img, build_morph_kernel(cv2.MORPH_RECT, (2, 2)))
    img_reformed_edges = cv2.dilate(img_reformed_edges, build_morph_kernel(cv2.MORPH_RECT, (9, 9)))

    debug_image("Image avec contours reformes", img_reformed_edges)

    # Find all external contours
    contours_final = cv2.findContours(img_reformed_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # For each contour, we check if it is big enough (to avoid points), and display it on the image
    return_image = new_img.copy()
    contour_objects = []
    for c in contours_final:
        area = cv2.contourArea(c)
        if area > 500:
            contour_objects.append(c)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(return_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Print the current occupation
    floor_area = cv2.contourArea(contour_floor)
    object_area = sum((cv2.contourArea(cont) for cont in contour_objects))

    occupation = object_area/floor_area
    print(f"L'image {img_name} de la pièce {room} possède une occupation de : {round(occupation*100, 2)}% !")

    return return_image

def debug_image(win_name, img):
    if DEBUG:
        cv2.imshow(win_name, img)


def build_morph_kernel(kernel_type, kernel_size):
    """Creates the specific kernel: MORPH_ELLIPSE, MORPH_CROSS or MORPH_RECT"""

    if kernel_type == cv2.MORPH_ELLIPSE:
        # We build a elliptical kernel
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    elif kernel_type == cv2.MORPH_CROSS:
        # We build a cross-shape kernel
        return cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
    else:  # cv2.MORPH_RECT
        # We build a rectangular kernel:
        return cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
