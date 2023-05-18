from utils.common import *
from model import SENext

scale = 2
ckpt_path = f"checkpoint/x{scale}/SENext-x{scale}.h5" 

model = SENext(scale)
model.load_weights(ckpt_path)

#ls_data = sorted_list("test/SET5/X2/LR")
#ls_labels = sorted_list("test/SET5/X2/HR")
#ls_data = sorted_list("test/SET14/X2/LR")
#ls_labels = sorted_list("test/SET14/X2/HR")
ls_data = sorted_list("test/BSD100/X2/LR")
ls_labels = sorted_list("test/BSD100/X2/HR")
#ls_data = sorted_list("test/URBAN100/X2/LR")
#ls_labels = sorted_list("test/URBAN100/X2/HR")
#ls_data = sorted_list("test/MANGA109/X2/LR")
#ls_labels = sorted_list("test/MANGA109/X2/HR")

sum_psnr = 0
for i in range(0, len(ls_data)):
    lr_image = read_image(ls_data[i])
    hr_image = read_image(ls_labels[i])

    lr_image = rgb2ycbcr(lr_image)
    hr_image = rgb2ycbcr(hr_image)

    lr_image = norm01(lr_image)
    hr_image = norm01(hr_image)

    lr_image = tf.expand_dims(lr_image, axis=0)
    sr_image = model.predict(lr_image)[0]

    sum_psnr += PSNR(hr_image, sr_image, max_val=1.0).numpy()

print(sum_psnr / len(ls_data))



