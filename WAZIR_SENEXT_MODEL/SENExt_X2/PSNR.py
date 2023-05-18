import numpy
import math
import cv2
img1 = cv2.imread(r"GROUND_TRUTH_IMAGES/bird.png")
img2 = cv2.imread(r"WAZIR_MODEL_OUTPUT_GENERATED_IMAGES/bird_sr.png",1)
def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

d=psnr(img1,img2)
print(d)