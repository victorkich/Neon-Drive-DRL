# -*- coding: utf-8 -*-
''' Modules for installation -> torch, torchvision, numpy, keyboard, cv2, mss.
    Use pip3 install 'module'.
'''
from skimage.filters import threshold_triangle
from skimage import img_as_ubyte
import torchvision.transforms as T
from PIL import Image
import numpy as np
import threading
import keyboard
import torch
import cv2
import time
import mss

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def histogram(image):
    ''' Histogram generate function
    '''
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


class env():
    def __init__(self, resolution, noise=False):
        self.w, self.h = resolution
        self.movements = ["a", "d", "w"]
        self.noise = noise

        img = cv2.imread('gameover.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.hist_restart = histogram(img)

        imageCapture = threading.Thread(name='imageCapture', target=self.imageCapture)
        imageCapture.setDaemon(True)
        imageCapture.start()

    def sap_noise(self, image):
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)

        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out

    def imageCapture(self):
        monitor = {'left': 0, 'top': 0, 'width': self.w, 'height': self.h}
        with mss.mss() as sct:
            while True:
                sct_img = sct.grab(monitor)
                img_np = np.array(sct_img)
                if self.noise:
                    img_np = self.sap_noise(img_np)
                gray_frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
                threash_triangle = threshold_triangle(gray_frame)
                binary_triangle = gray_frame > threash_triangle
                binary_triangle = img_as_ubyte(binary_triangle)
                cropped_bw_frame = binary_triangle[int(7*(self.h/13.0)):int(self.h-(self.h/5)), int(self.w/5):int(self.w-(self.w/5))]
                resized_bw_frame = cv2.resize(cropped_bw_frame, (int(160), int(90)), interpolation=cv2.INTER_AREA)
                self.bw_frame = cv2.bitwise_not(resized_bw_frame)
                self.rgb_frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                _ = cv2.waitKey(1)

    def reset(self):
        time.sleep(0.4)
        keyboard.send("enter")
        time.sleep(3.5)
        self.initial_time = time.time()

    def step(self, action):
        done = False
        keyboard.send(self.movements[action])
        time.sleep(0.25)
        hist = histogram(self.rgb_frame)
        comparison = cv2.compareHist(self.hist_restart, hist, cv2.HISTCMP_BHATTACHARYYA)
        if comparison > 0.15:
            rew = 0
            multiplication = time.time() - self.initial_time
            if multiplication >= 2:
                rew = 1
        else:
            done = True
            rew = 0
        return [], rew, done, []

    def get_screen(self):
        return resize(self.bw_frame).unsqueeze(0).to(device)
