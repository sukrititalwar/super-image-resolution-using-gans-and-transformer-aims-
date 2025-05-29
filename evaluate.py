import os
import numpy as np
import tensorflow as tf
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from model import build_generator
from utils import normalize, denormalize

HR_DIR = './data/test/HR'
LR_DIR = './data/test/LR'
LR_SHAPE = (24, 24, 3)
HR_SHAPE = (96, 96, 3)
WEIGHTS_PATH = 'generator_epoch_20.weights.h5'  # <-- updated extension

def preprocess_image(path, target_shape):
    img = tf.keras.preprocessing.image.load_img(path, target_size=target_shape[:2])
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = (img / 127.5) - 1.0
    return img

def main():
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Weights file '{WEIGHTS_PATH}' not found.")
        return

    generator = build_generator()
    generator.load_weights(WEIGHTS_PATH)

    hr_images = sorted([os.path.join(HR_DIR, f) for f in os.listdir(HR_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    lr_images = sorted([os.path.join(LR_DIR, f) for f in os.listdir(LR_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if len(hr_images) == 0 or len(lr_images) == 0:
        print("Error: No images found in test directories.")
        return

    psnr_list = []
    ssim_list = []

    for hr_path, lr_path in zip(hr_images, lr_images):
        hr = preprocess_image(hr_path, HR_SHAPE)
        lr = preprocess_image(lr_path, LR_SHAPE)
        sr = generator.predict(np.expand_dims(lr, 0))[0]

        hr = denormalize(np.expand_dims(hr, 0))[0]
        sr = denormalize(np.expand_dims(sr, 0))[0]

        # For recent scikit-image, use channel_axis instead of multichannel
        psnr_val = psnr(hr, sr, data_range=255)
        ssim_val = ssim(hr, sr, data_range=255, channel_axis=-1)

        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

    print(f"Average PSNR: {np.mean(psnr_list):.2f}")
    print(f"Average SSIM: {np.mean(ssim_list):.4f}")

if __name__ == '__main__':
    main()