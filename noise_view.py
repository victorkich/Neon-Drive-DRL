from skimage.filters import threshold_triangle
from cv2 import VideoCapture, VideoWriter
from skimage import img_as_ubyte
from tqdm import tqdm
import noise
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
in1 = VideoWriter('s&p_1.mp4', fourcc, 60.0, (w1, h1))
out1 = VideoWriter('s&p_2.mp4', fourcc, 60.0, (w4, h4))
in2 = VideoWriter('gauss_1.mp4', fourcc, 60.0, (w1, h1))
out2 = VideoWriter('gauss_2.mp4', fourcc, 60.0, (w4, h4))
in3 = VideoWriter('poisson_1.mp4', fourcc, 60.0, (w1, h1))
out3 = VideoWriter('poisson_2.mp4', fourcc, 60.0, (w4, h4))
in4 = VideoWriter('speckle_1.mp4', fourcc, 60.0, (w1, h1))
out4 = VideoWriter('speckle_2.mp4', fourcc, 60.0, (w4, h4))

bw1 = list()
bw2 = list()
bw3 = list()
bw4 = list()
for i in tqdm(range(int(cap.get(7)))):
    ret, frame = cap.read()
    if ret:
        n1 = noise.noisy(frame, "s&p")
        n1 = cv2.normalize(n1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        in1.write(n1)
        gray_frame = cv2.cvtColor(n1, cv2.COLOR_BGR2GRAY)
        threash_triangle = threshold_triangle(gray_frame)
        binary_triangle = gray_frame > threash_triangle
        binary_triangle = img_as_ubyte(binary_triangle)
        cropped_bw_frame = binary_triangle[int(7 * (h / 13.0)):int(h - (h / 5)), int(w / 5):int(w - (w / 5))]
        resized_bw_frame = cv2.resize(cropped_bw_frame, (int(160), int(90)), interpolation=cv2.INTER_AREA)
        bw_frame = cv2.bitwise_not(resized_bw_frame)
        bw1.append(bw_frame)
        if i > 20:
            difference = cv2.bitwise_not(bw1[i-20] - bw_frame)
            out1.write(cv2.cvtColor(difference, cv2.COLOR_BGR2RGB))

        n2 = noise.noisy(frame, "gauss")
        n2 = cv2.normalize(n2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        in2.write(n2)
        gray_frame = cv2.cvtColor(n2, cv2.COLOR_BGR2GRAY)
        threash_triangle = threshold_triangle(gray_frame)
        binary_triangle = gray_frame > threash_triangle
        binary_triangle = img_as_ubyte(binary_triangle)
        cropped_bw_frame = binary_triangle[int(7 * (h / 13.0)):int(h - (h / 5)), int(w / 5):int(w - (w / 5))]
        resized_bw_frame = cv2.resize(cropped_bw_frame, (int(160), int(90)), interpolation=cv2.INTER_AREA)
        bw_frame = cv2.bitwise_not(resized_bw_frame)
        bw2.append(bw_frame)
        if i > 20:
            difference = cv2.bitwise_not(bw2[i-20] - bw_frame)
            out2.write(cv2.cvtColor(difference, cv2.COLOR_BGR2RGB))
            
        n3 = noise.noisy(frame, "poisson")
        n3 = cv2.normalize(n3, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        in3.write(n3)
        gray_frame = cv2.cvtColor(n3, cv2.COLOR_BGR2GRAY)
        threash_triangle = threshold_triangle(gray_frame)
        binary_triangle = gray_frame > threash_triangle
        binary_triangle = img_as_ubyte(binary_triangle)
        cropped_bw_frame = binary_triangle[int(7 * (h / 13.0)):int(h - (h / 5)), int(w / 5):int(w - (w / 5))]
        resized_bw_frame = cv2.resize(cropped_bw_frame, (int(160), int(90)), interpolation=cv2.INTER_AREA)
        bw_frame = cv2.bitwise_not(resized_bw_frame)
        bw3.append(bw_frame)
        if i > 20:
            difference = cv2.bitwise_not(bw3[i-20] - bw_frame)
            out3.write(cv2.cvtColor(difference, cv2.COLOR_BGR2RGB))
            
        n4 = noise.noisy(frame, "speckle")
        n4 = cv2.normalize(n4, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        in4.write(n4)
        gray_frame = cv2.cvtColor(n4, cv2.COLOR_BGR2GRAY)
        threash_triangle = threshold_triangle(gray_frame)
        binary_triangle = gray_frame > threash_triangle
        binary_triangle = img_as_ubyte(binary_triangle)
        cropped_bw_frame = binary_triangle[int(7 * (h / 13.0)):int(h - (h / 5)), int(w / 5):int(w - (w / 5))]
        resized_bw_frame = cv2.resize(cropped_bw_frame, (int(160), int(90)), interpolation=cv2.INTER_AREA)
        bw_frame = cv2.bitwise_not(resized_bw_frame)
        bw4.append(bw_frame)
        if i > 20:
            difference = cv2.bitwise_not(bw4[i-20] - bw_frame)
            out4.write(cv2.cvtColor(difference, cv2.COLOR_BGR2RGB))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
in1.release()
out1.release()
in2.release()
out2.release()
in3.release()
out3.release()
in4.release()
out4.release()
