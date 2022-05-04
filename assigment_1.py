from tkinter import LEFT, RIGHT

from PIL import Image, ImageTk
import tkinter as tk
import cv2

from color import Colors
from detector import detect
from shape import Shapes

color_switch = {
    'RED': Colors.RED,
    'GREEN': Colors.GREEN,
    "YELLOW": Colors.YELLOW,
    "BLUE": Colors.BLUE
}
shape_switch = {
    'CIRCLE': Shapes.CIRCLE,
    'RECTANGLE': Shapes.RECTANGLE
}


class Application:
    def __init__(self):
        """ Initialize application which uses OpenCV + Tkinter. It displays
            a video stream in a Tkinter window and stores current snapshot on disk """
        self.vs = cv2.VideoCapture(0)
        self.current_image = None

        self.root = tk.Tk()
        self.root.title("Pocitacove videnie a spracovanie obrazu")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)

        self.panel = tk.Label(self.root)
        self.panel.pack(side=LEFT, padx=10, pady=10)

        self.second_panel = tk.Label(self.root)
        self.second_panel.pack(side=RIGHT, padx=10, pady=10)

        label = tk.Label(self.root, text="DETECTION MENU", font=("", 10))
        label.pack(padx=10, pady=10)

        # Create Dropdown menus

        label = tk.Label(self.root, text="Select color!", font=("", 10))
        label.pack()
        color_options = ["RED", "GREEN", "YELLOW", "BLUE"]
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

        label = tk.Label(self.root, text="RIGHT IMAGE SEL", font=("", 10))
        label.pack(padx=10, pady=(15, 5))
        self.selected_img = tk.IntVar()
        self.selected_img.set(1)
        R1 = tk.Radiobutton(self.root, text="   HSV Image", variable=self.selected_img, value=1)
        R1.pack(anchor=tk.W)
        R2 = tk.Radiobutton(self.root, text="Shape processed img\n(if present, otherwise HSV)",variable=self.selected_img, value=2)
        R2.pack(anchor=tk.W)

        self.video_loop()

    def video_loop(self):
        """ Get frame from the video stream and show it in Tkinter """
        ok, frame = self.vs.read()
        if ok:
            color_img, thresh = detect(frame,
                                       color_switch.get(self.clicked_color.get(), Colors.INVALID),
                                       shape_switch.get(self.clicked_shape.get(), Shapes.INVALID),
                                       self.selected_img.get())

            cv2image = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA
            self.current_image = Image.fromarray(cv2image)          # convert image for PIL
            imgtk = ImageTk.PhotoImage(image=self.current_image)    # convert image for tkinter
            self.panel.imgtk = imgtk                                # anchor imgtk so it does not be deleted by garbage-collector
            self.panel.config(image=imgtk)                          # show the image

            cv2image = cv2.cvtColor(thresh, cv2.COLOR_BGR2RGBA)     # convert colors from BGR to RGBA
            self.current_image = Image.fromarray(cv2image)          # convert image for PIL
            imgtk = ImageTk.PhotoImage(image=self.current_image)    # convert image for tkinter
            self.second_panel.imgtk = imgtk                         # anchor imgtk so it does not be deleted by garbage-collector
            self.second_panel.config(image=imgtk)                   # show the image

        self.root.after(30, self.video_loop)                # call the same function after 30 milliseconds

    def destructor(self):
        """ Destroy the root object and release all resources """
        print("[INFO] closing...")
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()


print("[INFO] starting...")
pba = Application()
pba.root.mainloop()
