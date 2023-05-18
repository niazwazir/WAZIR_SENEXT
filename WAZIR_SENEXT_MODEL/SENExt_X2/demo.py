from utils.common import *
from model import SENext
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--scale',        type=int,   default=2,                      help='-')
parser.add_argument("--image-path",   type=str,   default="dataset/test/img_001_SRF_2_LR.png", help='-')##############################################
parser.add_argument("--ckpt-path",    type=str,   default="checkpoint/x2/SENext-x2.h5",   help='-')

FLAGS, unparsed = parser.parse_known_args()
image_path = FLAGS.image_path
ckpt_path = FLAGS.ckpt_path
scale = FLAGS.scale

if scale not in [2, 3, 4]:
    ValueError("scale must be 2, 3, or 4")


# -----------------------------------------------------------
#  read image and save bicubic image
# -----------------------------------------------------------

lr_image = read_image(image_path)
bicubic_image = upscale(lr_image, scale)
write_image("bicubic.png", bicubic_image)

# -----------------------------------------------------------
# preprocess lr image 
# -----------------------------------------------------------

lr_image = rgb2ycbcr(lr_image)
lr_image = norm01(lr_image)
lr_image = tf.expand_dims(lr_image, axis=0)


# -----------------------------------------------------------
#  predict and save image
# -----------------------------------------------------------

model = SENext(scale)
model.load_weights(ckpt_path)
sr_image = model.predict(lr_image)[0]

sr_image = denorm01(sr_image)
sr_image = tf.cast(sr_image, tf.uint8)
sr_image = ycbcr2rgb(sr_image)

write_image("sr.png", sr_image)

import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('sr.png', cv2.IMREAD_UNCHANGED)
img_scaled = cv2.resize(img,None,fx=0.5,fy=0.5)
#img_scaled=ycbcr2rgb(img_scaled)
img_scaled = (cv2.cvtColor(img_scaled, cv2.COLOR_BGR2RGB))
write_image("HR.png", img_scaled)

#######SSIM###########
original = cv2.imread(r"dataset/test/img_001_SRF_2_LR.png")########################################################################
contrast = cv2.imread(r"HR.png",1)
def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()



def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

a=calculate_ssim(original,contrast)
print('SSIM:', a)