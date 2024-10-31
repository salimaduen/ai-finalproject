from skimage.restoration import denoise_bilateral, denoise_wavelet, denoise_tv_chambolle
import cv2


def resize_image(image, target_size=(224, 224)):
    return cv2.resize(image, target_size)


def normalize_image(image, normalization=255.0):
    return image / normalization


def denoise_image(image, denoising='non_local_means'):
    denoised_image = None
    if denoising == 'non_local_means':
        denoised_image = cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)
    elif denoising == 'bilateral':
        denoised_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    elif denoising == 'wavelet':
        denoised_image = denoise_wavelet(image)
    elif denoising == 'total_variation':
        denoised_image = denoise_tv_chambolle(image, weight=0.1)
    elif denoising == 'anisotropic':
        denoised_image = denoise_bilateral(image)
    return denoised_image


def contrast_enhancement(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)
