import cv2

def detect_changes(ref_img, img):
    diff_img = cv2.subtract(ref_img, img)

    cv2.imshow("Diff", diff_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

