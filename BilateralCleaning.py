import cv2
import numpy as np
import matplotlib.pyplot as plt

def clean_Gaussian_noise_bilateral(im, radius, stdSpatial, stdIntensity):
    # Define constants
    height, width = im.shape
    # Pad the image with zeros
    padded_im = np.pad(im, ((radius, radius), (radius, radius)), mode='constant')
    # Create grid of indices for faster computation
    x, y = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
    # Compute the spatial Gaussian mask
    gs = np.exp(-(x ** 2 + y ** 2) / (2 * stdSpatial ** 2)).astype(np.float64)
    cleanIm = np.zeros_like(im, dtype=np.float64)
    
    # Iterate over each pixel
    for i in range(height):
        for j in range(width):
            # Extract the window around the current pixel
            window = padded_im[i:i+2*radius+1, j:j+2*radius+1]
            # Compute the intensity Gaussian mask
            gi = np.exp(-((window - im[i, j]) ** 2) / (2 * stdIntensity ** 2)).astype(np.float64)
            # Compute the weighted sum
            weighted_sum = np.sum(gs * gi * window)
            # Compute the normalization factor
            normalization_factor = np.sum(gs * gi)
            #Avoid dividing by zero
            if normalization_factor != 0 :
                cleanIm[i, j] = weighted_sum / normalization_factor
            else:
                cleanIm[i, j] =im[i, j]
    
    return cleanIm.astype(np.uint8)

#taj 2% of its diagonal is 8.485281374238571
#NoisyGrayImage 2% of its diagonal is 6.708203932499369
#balls 2% of its diagonal is 6.468044526748405

clear_image_b_taj = clean_Gaussian_noise_bilateral(cv2.imread('taj.jpg', cv2.IMREAD_GRAYSCALE), radius=2, stdSpatial=8.485281374238571, stdIntensity=70)
cv2.imwrite(f'Taj_fixed_image.jpg', clear_image_b_taj)

clear_image_b_NoisyGrayImage = clean_Gaussian_noise_bilateral(cv2.imread('NoisyGrayImage.png', cv2.IMREAD_GRAYSCALE), radius=4, stdSpatial=6.708203932499369, stdIntensity=50)
cv2.imwrite(f'NoisyGrayImage_fixed_image.jpg', clear_image_b_NoisyGrayImage)

clear_image_b_balls = clean_Gaussian_noise_bilateral(cv2.imread('balls.jpg', cv2.IMREAD_GRAYSCALE), radius=1, stdSpatial=6.468044526748405, stdIntensity=70)
cv2.imwrite(f'Balls_fixed_image.jpg', clear_image_b_balls)



plt.figure(figsize=(10,10))
plt.subplot(321)
plt.title('Original Taj Image')
plt.imshow(cv2.imread('taj.jpg', cv2.IMREAD_GRAYSCALE), cmap='gray')

plt.subplot(322)
plt.title('Fixed Taj Image')
plt.imshow(clear_image_b_taj, cmap='gray')

plt.subplot(323)
plt.title('Original NoisyGrayImage Image')
plt.imshow(cv2.imread('NoisyGrayImage.png', cv2.IMREAD_GRAYSCALE), cmap='gray')

plt.subplot(324)
plt.title('Fixed NoisyGrayImage Image')
plt.imshow(clear_image_b_NoisyGrayImage, cmap='gray')

plt.subplot(325)
plt.title('Original Balls Image')
plt.imshow(cv2.imread('balls.jpg', cv2.IMREAD_GRAYSCALE), cmap='gray')

plt.subplot(326)
plt.title('Fixed Balls Image')
plt.imshow(clear_image_b_balls, cmap='gray')
plt.show()


