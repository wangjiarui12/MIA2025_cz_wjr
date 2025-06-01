import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import glob
from scipy.ndimage import uniform_filter
import argparse
import os

def calculate_psnr_ssim(img1, img2):
    psnr_value = psnr(img1, img2)
    if len(img1.shape) == 3:
        min_dim = min(img1.shape[0], img1.shape[1])
        win_size = min(7, min_dim // 2 * 2 - 1)
        ssim_value = ssim(img1, img2, channel_axis=-1, win_size=win_size)
    else:
        ssim_value = ssim(img1, img2)
    return psnr_value, ssim_value

def calculate_lncc(img1, img2, win_size=7):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    mean1 = uniform_filter(img1, size=win_size)
    mean2 = uniform_filter(img2, size=win_size)

    mean1_sq = uniform_filter(img1 * img1, size=win_size)
    mean2_sq = uniform_filter(img2 * img2, size=win_size)
    mean12 = uniform_filter(img1 * img2, size=win_size)

    sigma1 = np.sqrt(mean1_sq - mean1 ** 2)
    sigma2 = np.sqrt(mean2_sq - mean2 ** 2)
    covariance = mean12 - mean1 * mean2

    epsilon = 1e-5
    lncc_map = covariance / (sigma1 * sigma2 + epsilon)
    return np.mean(lncc_map)

def load_images(image_folder1, image_folder2):
    image_files1 = sorted(glob.glob(os.path.join(image_folder1, "*.[pj][pn]g")))
    image_files2 = sorted(glob.glob(os.path.join(image_folder2, "*.[pj][pn]g")))
    images1 = [cv2.imread(f) for f in image_files1]
    images2 = [cv2.imread(f) for f in image_files2]
    return images1, images2

def resize_images_to_fixed_size(images1, images2, target_size=(512, 512)):
    resized1 = [cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR) for img in images1]
    resized2 = [cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR) for img in images2]
    return resized1, resized2

def calculate_average_metrics(images1, images2):
    total_psnr = total_ssim = total_lncc = 0
    for img1, img2 in zip(images1, images2):
        psnr_value, ssim_value = calculate_psnr_ssim(img1, img2)
        lncc_value = calculate_lncc(img1, img2)
        total_psnr += psnr_value
        total_ssim += ssim_value
        total_lncc += lncc_value
    n = len(images1)
    return total_psnr / n, total_ssim / n, total_lncc / n

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute PSNR, SSIM, and LNCC between two image folders.")
    parser.add_argument("--HQ", type=str, required=True, help="Path to ground truth image folder")
    parser.add_argument("--LQ", type=str, required=True, help="Path to predicted/enhanced image folder")
    parser.add_argument("--size", type=int, default=512, help="Target image size (default: 512)")
    args = parser.parse_args()

    images1, images2 = load_images(args.folder1, args.folder2)
    images1, images2 = resize_images_to_fixed_size(images1, images2, target_size=(args.size, args.size))
    avg_psnr, avg_ssim, avg_lncc = calculate_average_metrics(images1, images2)

    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average LNCC: {avg_lncc:.4f}")
