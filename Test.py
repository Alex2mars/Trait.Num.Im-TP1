import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

img= cv2.imread('Images/Chambre/Reference.JPG')
img1=cv2.imread('Images/Chambre/IMG_6572.JPG')
img= cv2.resize(img, (800, 400))
img1=cv2.resize(img1, (800, 400))
images=np.concatenate((img,img1),axis=1)

gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray_img1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

gray_img_eqhist=cv2.equalizeHist(gray_img)
gray_img1_eqhist=cv2.equalizeHist(gray_img1)

eqhist_images=np.concatenate((gray_img_eqhist,gray_img1_eqhist),axis=1)

clahe=cv2.createCLAHE(clipLimit=40)
gray_img_clahe=clahe.apply(gray_img_eqhist)
gray_img1_clahe=clahe.apply(gray_img1_eqhist)
images=np.concatenate((gray_img_clahe,gray_img1_clahe),axis=1)

th=80
max_val=255

ret, o1 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_TOZERO)
ret, o3 = cv2.threshold(gray_img1_clahe, th, max_val, cv2.THRESH_TOZERO)

final=np.concatenate((o1,o3),axis=1)
cv2.imwrite("Images/Chambre/Image1.jpg", final)

ret,thresh1 = cv2.threshold(o1,0, 255,  cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret,thresh2 = cv2.threshold(o3,0, 255,  cv2.THRESH_BINARY+cv2.THRESH_OTSU)

difference = cv2.subtract(thresh1,thresh2)

thresh = cv2.threshold(difference, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

mask = np.zeros(thresh1.shape, dtype='uint8')
filled_after = thresh2.copy()
for c in contours:
    area = cv2.contourArea(c)
    if area > 500:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(thresh1, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(thresh2, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.drawContours(mask, [c], 0, (0, 255, 0), -1,)
        cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1)


cv2.imwrite('Images/Chambre/rect.jpeg', np.concatenate((thresh1, thresh2), axis=1))

cv2.imshow("Images",thresh1)
cv2.imshow("Images2",thresh2)
cv2.imshow("images3",difference)

cv2.waitKey(0)
cv2.destroyAllWindows()