from tkinter import LEFT, RIGHT

from PIL import Image, ImageTk
import tkinter as tk
import argparse
import datetime
import cv2
import os

from color import Colors
from detector import detect
from shape import Shapes

color_switch = {
    'RED': Colors.RED,
    'GREEN': Colors.GREEN,
    "YELLOW": Colors.YELLOW
}
shape_switch = {
    'CIRCLE': Shapes.CIRCLE,
    'RECTANGLE': Shapes.RECTANGLE
}


class Application:
    def __init__(self):
        """ Initialize application which uses OpenCV + Tkinter. It displays
            a video stream in a Tkinter window and stores current snapshot on disk """
        self.vs = cv2.VideoCapture(0)  # capture video frames, 0 is your default video camera
        self.current_image = None  # current image from the camera

        self.root = tk.Tk()  # initialize root window
        self.root.title("PyImageSearch PhotoBooth")  # set window title
        # self.destructor function gets fired when the window is closed
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)

        self.panel = tk.Label(self.root)  # initialize image panel
        self.panel.pack(side=LEFT, padx=10, pady=10)

        self.second_panel = tk.Label(self.root)  # initialize image panel
        self.second_panel.pack(side=RIGHT, padx=10, pady=10)

        label = tk.Label(self.root, text="DETECTION MENU", font=("", 10))
        label.pack(padx=10, pady=10)

        # Create Dropdown menu
        label = tk.Label(self.root, text="Select color!", font=("", 10))
        label.pack()
        color_options = ["RED", "GREEN"]
        self.clicked_color = tk.StringVar()
        self.clicked_color.set("RED")
        drop = tk.OptionMenu(self.root, self.clicked_color, *color_options)
        drop.pack(pady=(0, 10))

        label = tk.Label(self.root, text="Select shape!", font=("", 10))
        label.pack()
        shape_options = ["RECTANGLE", "CIRCLE"]
        self.clicked_shape = tk.StringVar()
        self.clicked_shape.set("RECTANGLE")
        drop = tk.OptionMenu(self.root, self.clicked_shape, *shape_options)
        drop.pack(pady=(0, 10))

        # start a self.video_loop that constantly pools the video sensor
        # for the most recently read frame
        self.video_loop()

    def video_loop(self):
        """ Get frame from the video stream and show it in Tkinter """
        ok, frame = self.vs.read()  # read frame from video stream
        if ok:
            color_img, thresh = detect(frame,
                               color_switch.get(self.clicked_color.get(), Colors.INVALID),
                               shape_switch.get(self.clicked_shape.get(), Shapes.INVALID))

            cv2image = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA
            self.current_image = Image.fromarray(cv2image)  # convert image for PIL
            imgtk = ImageTk.PhotoImage(image=self.current_image)  # convert image for tkinter
            self.panel.imgtk = imgtk  # anchor imgtk so it does not be deleted by garbage-collector
            self.panel.config(image=imgtk)  # show the image

            cv2image = cv2.cvtColor(thresh, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA
            self.current_image = Image.fromarray(cv2image)  # convert image for PIL
            imgtk = ImageTk.PhotoImage(image=self.current_image)  # convert image for tkinter
            self.second_panel.imgtk = imgtk  # anchor imgtk so it does not be deleted by garbage-collector
            self.second_panel.config(image=imgtk)  # show the image

        self.root.after(30, self.video_loop)  # call the same function after 30 milliseconds

    def destructor(self):
        """ Destroy the root object and release all resources """
        print("[INFO] closing...")
        self.root.destroy()
        self.vs.release()  # release web camera
        cv2.destroyAllWindows()  # it is not mandatory in this application


# start the app
print("[INFO] starting...")
pba = Application()
pba.root.mainloop()
