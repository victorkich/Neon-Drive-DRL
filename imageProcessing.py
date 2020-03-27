from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import time 
import mss
import os

# load model
model = load_model('model.h5')
# summarize model.
model.summary()

monitor = {'left': 0, 'top': 0, 'width': 1920, 'height': 1080}
ipixel = 35
epixel = 60

cont = 0
fps_mean = 0
with mss.mss() as sct:
    while True:
        initial_time = time.time()
        sct_img = sct.grab(monitor)
        img_np = np.array(sct_img)
        frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        (thresh, frame) = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
        numbers = [frame[ipixel:epixel,16:34], frame[ipixel:epixel,33:52],frame[ipixel:epixel,60:79],
                    frame[ipixel-1:epixel+1,78:96]]
        resized = []
        reformed = []
        for i in range(4):
            resized.append(cv2.resize(numbers[i], (28,28), interpolation = cv2.INTER_NEAREST))
            reformed.append(model.predict(resized[i].reshape(1, 28, 28, 1)).argmax())
        cv2.imshow("frame", resized[3])
        _ = cv2.waitKey(1)
        fps = 1/(time.time() - initial_time)
        os.system("clear")
        print("\nFPS: ", fps)
        fps_mean = 0	
        print(reformed)
            
    cv2.destroyAllWindows()
