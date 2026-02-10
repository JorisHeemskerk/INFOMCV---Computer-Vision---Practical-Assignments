"""
This file allows one to select points on an image for manual calibration.

Code based on geeksforgeeks tutorial `Displaying the coordinates of the 
points clicked on the image using Python-OpenCV`
link: https://www.geeksforgeeks.org/python/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/
"""

import cv2
 
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f"{x},{y}", (x, y), font, 1, (255, 0, 0), 2)
        cv2.imshow('image', img)

    if event == cv2.EVENT_RBUTTONDOWN:
        print(x, y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        b, g, r = img[y, x]
        cv2.putText(img, f"{b},{g},{r}", (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)

if __name__=="__main__":
    img = cv2.imread('assignment_1/data/img_0.jpg', 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()