from skimage.filters import threshold_triangle
from cv2 import VideoCapture, VideoWriter
from skimage import img_as_ubyte
from tqdm import tqdm
import cv2

w = 1920
h = 1080

cap = VideoCapture('neondrl.mp4')
ret, frame = cap.read()
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
h1, w1 = gray_frame.shape

threash_triangle = threshold_triangle(gray_frame)
binary_triangle = gray_frame > threash_triangle
binary_triangle = img_as_ubyte(binary_triangle)
h2, w2 = binary_triangle.shape

cropped_bw_frame = binary_triangle[int(7 * (h / 13.0)):int(h - (h / 5)), int(w / 5):int(w - (w / 5))]
h3, w3 = cropped_bw_frame.shape

resized_bw_frame = cv2.resize(cropped_bw_frame, (int(160), int(90)), interpolation=cv2.INTER_AREA)
bw_frame = cv2.bitwise_not(resized_bw_frame)
last = bw_frame
h4, w4 = bw_frame.shape

# Define the codec and create VideoWriter object.
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out1 = VideoWriter('1_gray.mp4', fourcc, 60.0, (w1, h1))
out2 = VideoWriter('2_binary_triangle.mp4', fourcc, 60.0, (w2, h2))
out3 = VideoWriter('3_cropped_bw.mp4', fourcc, 60.0, (w3, h3))
out4 = VideoWriter('4_resized_bw.mp4', fourcc, 60.0, (w4, h4))
out5 = VideoWriter('5_difference.mp4', fourcc, 60.0, (w4, h4))

bw = list()
for i in tqdm(range(int(cap.get(7)))):
    ret, frame = cap.read()
    if ret:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out1.write(cv2.cvtColor(gray_frame, cv2.COLOR_BGR2RGB))

        threash_triangle = threshold_triangle(gray_frame)
        binary_triangle = gray_frame > threash_triangle
        binary_triangle = img_as_ubyte(binary_triangle)
        out2.write(cv2.cvtColor(binary_triangle, cv2.COLOR_BGR2RGB))

        cropped_bw_frame = binary_triangle[int(7 * (h / 13.0)):int(h - (h / 5)), int(w / 5):int(w - (w / 5))]
        out3.write(cv2.cvtColor(cropped_bw_frame, cv2.COLOR_BGR2RGB))

        resized_bw_frame = cv2.resize(cropped_bw_frame, (int(160), int(90)), interpolation=cv2.INTER_AREA)
        bw_frame = cv2.bitwise_not(resized_bw_frame)
        bw.append(bw_frame)
        out4.write(cv2.cvtColor(bw_frame, cv2.COLOR_BGR2RGB))

        if i > 20:
            difference = cv2.bitwise_not(bw[i-20] - bw_frame)
            out5.write(cv2.cvtColor(difference, cv2.COLOR_BGR2RGB))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out1.release()
out2.release()
out3.release()
out4.release()
out5.release()
