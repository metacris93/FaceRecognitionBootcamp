import tkinter
import PIL.Image, PIL.ImageTk
import cv2
import time

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.window.bind('<Escape>', lambda e: self.window.quit())
        self.video_source = video_source
        self.vid = Video(self.video_source)
        self.total = 0

        #self.canvas = tkinter.Canvas(window, width = 400, height = 400)
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()

        self.btn_snapshot = tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
        #self.btn_snapshot.pack(side=tkinter.BOTTOM)
        self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)

        self.delay = 15
        self.update()

        self.window.mainloop()

    def snapshot(self):
        ret, frame = self.vid.get_frame()
        if ret:
            cv2.imwrite("{}.png".format(str(self.total).zfill(5)), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            self.total += 1

    def update(self):
        ret, frame = self.vid.get_frame()
        if ret:
            detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            rects = detector.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in rects:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
        self.window.after(self.delay, self.update)


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

#class Box:


def main():
    mi_app = App(tkinter.Tk(), "TkInter y OpenCV")
    return 0

if __name__ == '__main__':
    main()