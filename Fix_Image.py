import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the broken image
broken_path='broken.jpg'
broken = cv2.imread(broken_path, cv2.IMREAD_GRAYSCALE)

plt.subplot(131)
plt.imshow(broken, cmap='gray')
plt.title('Original Image')

# Load the .npy file
data = np.load("noised_images.npy")
# Now we can access our images from the 'data' variable
#-----------------------------------------------------          

#a

#fixed_image = cv2.bilateralFilter(broken, d = 9, sigmaColor = 75, sigmaSpace = 75)

fixed_image = cv2.bilateralFilter(broken, d = 9, sigmaColor = 30, sigmaSpace = 10)
fixed_image = cv2.medianBlur(fixed_image, 3)

cv2.imwrite(f'section_a_fixed_image.jpg', fixed_image)
#cv2.imshow("Fixed Image", fixed_image)


plt.subplot(132)
plt.imshow(fixed_image, cmap='gray')
plt.title('fixed image section a')

#-----------------------------------------------------          

#b
height, width= broken.shape
fixed_image= np.zeros((height, width, 3))
# Iterate through each pixel
for y in range(height):
    for x in range(width):
        # Access the pixel value at coordinates (x, y)
        #pixel_value = image[y, x]
        sum=0
        for img_num in range(200):
            sum=sum+data[img_num][y][x]
        avg=sum/200
        fixed_image[y, x]=avg

cleanIm = np.clip(fixed_image, 0, 255).astype(np.uint8)
cv2.imwrite(f'section_b_fixed_image.jpg', cleanIm)
#cv2.imshow("Fixed Image", cleanIm)

plt.subplot(133)
plt.imshow(cleanIm, cmap='gray')
plt.title('fixed image section b')

plt.show()




