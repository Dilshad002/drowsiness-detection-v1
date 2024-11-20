import tkinter as tk
import customtkinter as ctk 

import torch
import numpy as np

import cv2
from PIL import Image, ImageTk
import vlc 
import time

app = tk.Tk()
app.geometry("600x600")
app.title("Drowsy Boi 4.0")
ctk.set_appearance_mode("dark")

vidFrame = tk.Frame(height=600, width=600)
vidFrame.pack()
vid = ctk.CTkLabel(vidFrame)
vid.pack()

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp9/weights/last.pt', force_reload=True)
cap = cv2.VideoCapture(0)
def detect(): 
    global counter
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    results = model(frame) 
    img = np.squeeze(results.render())

    if len(results.xywh[0]) > 0: 
        dconf = results.xywh[0][0][4]
        dclass = results.xywh[0][0][5]

        if dconf.item() >= 0.50 and dclass.item() == 1.0:
            p = vlc.MediaPlayer(f"file:///1.wav")
            p.play()
            time.sleep(3)

    imgarr = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(imgarr) 
    vid.imgtk = imgtk
    vid.configure(image=imgtk)
    vid.after(10, detect) 


detect()
app.mainloop()