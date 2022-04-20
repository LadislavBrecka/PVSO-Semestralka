import cv2
import time
import sys

from color import Colors, CmdColors
from detector import detect
from shape import Shapes

# image setting
width = 640
height = 480


def main():
    COLOR: Colors
    SHAPE: Shapes
    if len(sys.argv) < 3:
        COLOR = Colors.GREEN
        SHAPE = Shapes.CIRCLE
    else:
        color_switch = {
            'RED': Colors.RED,
            'GREEN': Colors.GREEN,
            "YELLOW": Colors.YELLOW
        }
        shape_switch = {
            'CIRCLE': Shapes.CIRCLE,
            'RECTANGLE': Shapes.RECTANGLE
        }

        COLOR = color_switch.get(sys.argv[1], Colors.INVALID)
        SHAPE = shape_switch.get(sys.argv[2], Shapes.INVALID)
        if COLOR.value is None or SHAPE.value is None:
            sys.exit("Wrong shape or color entered")

        print(CmdColors.OK_GREEN + "Selected shape is {} with color {}".format(SHAPE.name, COLOR.name) + CmdColors.END)

    # setting up camera and capturing object
    imcap = cv2.VideoCapture(0)
    imcap.set(3, width)  # set width as 640
    imcap.set(4, height)  # set height as 480
    imcap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    imcap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Time is just used to get the Frames Per Second (FPS)
    last_time = time.time()

    while True:
        # Capture the frame
        _, img = imcap.read()

        color_img = detect(img, COLOR, SHAPE)
        # color_img = detect(img, RED, 'RECT')

        # Add a FPS label to image
        text = f"FPS: {int(1 / (time.time() - last_time))}"
        last_time = time.time()
        cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("Detector", color_img)

        # If q is pressed terminate
        if cv2.waitKey(1) == ord('q'):
            break

    # Release and destroy all windows
    imcap.release()
    cv2.destroyAllWindows()


main()




