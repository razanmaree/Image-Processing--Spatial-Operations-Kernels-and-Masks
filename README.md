# Image-Processing--Spatial-Operations-Kernels-and-Masks
## ***Question 1 – What happened here :***
In the ‘q1’ folder there’s an image file, “1.jpg”, of an otter:

![image](https://github.com/user-attachments/assets/e066a98e-3e6e-4457-a4bb-187d72622e79)

There are several additional grayscale image files in the q1 folder, where each one is the same image of an otter… Except each image had a spatial operation performed on it. In this question you’ll be tested to see if you can identify just what kind of spatial operation/mask has been applied on each image (assume the original image itself was also in grayscale).

## Bonus:
Fully identify the spatial operations applied on each image as accurately as you can and, using python, apply them yourselves on the original image (after you convert it to grayscale)! 



## ***Question 2 – Biliteral analysis :***
We’ve seen that in some simple cases gaussian noise in an image can be reasonably dealt with by applying gaussian blurring, however that won’t always be the best method. In this question we’ll be using bilateral filtering to clean up an image – and we’ll implement it ourselves!
In the ‘q2’ folder you’re given 3 different, noisy images and a skeleton of a python script file, BilateralCleaning.py"".

**Implement the following function in the python file you’re given:**

clean_Gaussian_noise_bilateral(im, radius, stdSpatial, stdIntensity)
Inputs:
- im - grayscale image
- radius – the radius of the window (window is square)
- stdSpatial – the std of the Gaussian window used for the spatial weight.
- stdIntensity – the std of the Gaussian window used for the intensity weight.

Output: cleanIm - grayscale image.

This function applies bilateral filtering to the given image. Bilateral filtering replaces each pixel with a weighted average of its neighbors where the weights are determined according to the spatial (coordinates) and intensity (values) distances.


## ***Question 3 – Fix me! :***
A mischievous entity added quite a bit of disruptive noise to an image of vegetables, as seen below:

![image](https://github.com/user-attachments/assets/4bf2942f-00ad-47e4-9722-946efb20d34f)

It’s up to you to fix the image!

In the ‘q3’ folder you’re given the above image as broken.jpg"". Your mission to attempt to fix the image to the best of your ability in 2 different ways:
- Use only broken.jpg itself and fix it as best as you can.
- Use the noised_images.npy file, containing 200 noised images, to help reduce noise and create a cleaned image.

