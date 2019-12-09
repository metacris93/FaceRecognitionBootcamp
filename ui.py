import tkinter
import PIL.Image, PIL.ImageTk
from keras import backend as K
import time
from multiprocessing.dummy import Pool
K.set_image_data_format('channels_first')
import cv2
import os
import glob
import numpy as np
from numpy import genfromtxt
import tensorflow as tf
from fr_utils import *
from inception_network import *
from keras.models import load_model
import sys

class App:
    def __init__(self, window, window_title, model, name, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.window.bind('<Escape>', lambda e: self.window.quit())
        self.video_source = video_source
        self.vid = Video(self.video_source)
        self.model = model
        self.name = name
        #self.total = 0

        self.detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        #self.canvas = tkinter.Canvas(window, width = 400, height = 400)
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()

        self.btn_snapshot = tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
        #self.btn_snapshot.pack(side=tkinter.BOTTOM)
        self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)

        self.delay = 15
        self.update()

        self.window.mainloop()

    def cutfaces(self, image, faces_coord):
        faces = []
        for (x,y,w,h) in faces_coord:
            w_rm = int(0.2*w/2)
            faces.append(image[y : y + h, x + w_rm : x + w - w_rm])
        return faces
    
    def GetFaceCoords(self, frame):
        return self.detector.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.3, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    def GetPath(self):
        path = os.path.join('dataset', self.name + '.jpg')
        return path

    def GetCutImage(self, frame, faces_coord):
        PADDING = 25
        faces = self.cutfaces(frame, faces_coord)
        if (len(faces) != 0):
            for (top, right, bottom, left) in faces_coord:
                cv2.rectangle(frame, (top, right), (top + bottom, right + left), (0, 255, 0), 2)
                height, width, channels = frame.shape
                cut_image = frame[max(0, right):min(height, right + left), max(0, top):min(width, top + bottom)]
            return cut_image
        return None

    def snapshot(self):
        ret, frame = self.vid.get_frame()
        if ret:
            faces_coord = self.GetFaceCoords(frame)
            cut_image = self.GetCutImage(frame, faces_coord)
            if cut_image is not None:
                path = self.GetPath()
                cv2.imwrite(path, cut_image)
            #self.total += 1

    def recognise_face(self, imagepath, database, model):
        encoding = img_to_encoding(imagepath, model)
        identity = None
        min_dist = 100
        for (name, db_enc) in database.items():
            
            dist = np.linalg.norm(db_enc - encoding)
            if dist < min_dist:
                min_dist = dist
                identity = name
        
        if min_dist > 0.6:
            return (str(0) , dist)
        else:
            return (str(identity) , dist)

    def update(self):
        name = ''
        ret, frame = self.vid.get_frame()
        if ret:
            #rects = self.detector.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.3, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
            face_coords = self.GetFaceCoords(frame)
            image = self.GetCutImage(frame, face_coords)
            if image is not None:
                cv2.imwrite("temp.jpg", image) 
                database = self.prepare_database()
                face, dist = self.recognise_face("temp.jpg", database, self.model)
                if face != '0':
                    name = face
                else:
                    name = 'DESCONOCIDO'
                os.remove("temp.jpg")

                for (top, right, bottom, left) in face_coords:
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, '%s %s' %(name, dist), (top + 6, right - 6), font, 1.0, (0, 255, 0), 1)

                self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
                self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
            self.window.after(self.delay, self.update)

    def prepare_database(self):
        database = {}
        for file in glob.glob("dataset/*"):
            identity = os.path.splitext(os.path.basename(file))[0]
            database[identity] = img_to_encoding(file, self.model)
        return database

class Video:
    def __init__(self, video_source=0):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source")
        
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

def triplet_loss_function(y_true,y_pred,alpha = 0.3):
	anchor = y_pred[0]
	positive = y_pred[1]
	negative = y_pred[2]
	pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
	neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
	basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
	loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
	return loss

def main():
    name = str(input("Ingrese el nombre: "))
    modelo = model(input_shape = (3,96,96))
    modelo.compile(optimizer = 'adam', loss = triplet_loss_function, metrics = ['accuracy'])
    print('Cargando pesos en el modelo...')
    load_weights_from_FaceNet(modelo)
    print('Pesos cargados...')
    mi_app = App(tkinter.Tk(), "TkInter y OpenCV", modelo, name)
    return 0

if __name__ == '__main__':
    main()