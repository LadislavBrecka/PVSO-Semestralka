
import cv2

app_name = "PVSO-Semestralka"

cv2.namedWindow(app_name)
vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if vc.isOpened():  # try to get the first frame
    r_val, frame = vc.read()

    while r_val:
        cv2.imshow(app_name, frame)
        r_val, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break

    cv2.destroyWindow(app_name)
    vc.release()


