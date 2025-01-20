import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter


def find_spatial_operation_mask(original_image,target_image,id):
    original_array = np.array(original_image)
    target_array=np.array(target_image)

    if id==1:
        width = original_image.shape[1]
        width=width*2

        kernel = np.ones((1, width)) / width

        filtered_image = cv2.filter2D(original_image, -1, kernel)

        # Calculate squared differences
        filtered_array=np.array(filtered_image)
        squared_diff = (target_array - filtered_array) ** 2
        mse_value = np.mean(squared_diff)
        print(f'Image_{id} Mse is {mse_value}')

        cv2.imwrite(f'image_{id}_recreated.jpg', filtered_image)

        '''
        # Display the original and filtered images
        cv2.imshow('Target Image', target_image)
        cv2.imshow('Result Image', filtered_image)
        '''

    if id==2:
        #There is two ways to get the target image:1.Blur, 2.Average with kernel 11*11
        #Both have equal MSE
        
        #1
        #filtered_image = cv2.blur(original_image, (11, 11))#kernel_size=11
        
        #2        
        # Define custom kernel for average filtering
        kernel_size = (11, 11)
        kernel = np.ones(kernel_size, np.float32) / (kernel_size[0] * kernel_size[1])

        #padding
        padding_height = kernel_size[0] 
        padding_width = kernel_size[1] 
        padded_image = cv2.copyMakeBorder(original_image, padding_height, padding_height, padding_width, padding_width, cv2.BORDER_CONSTANT, value=65)

        # Apply the filter 
        filtered_image = cv2.filter2D(padded_image, -1, kernel)

        # Ensure filtered image has the same dimensions as the original image
        filtered_image = filtered_image[padding_height:-padding_height, padding_width:-padding_width]

                
        # Calculate squared differences
        filtered_array=np.array(filtered_image)
        squared_diff = (target_array - filtered_array) ** 2
        mse_value = np.mean(squared_diff)
        print(f'Image_{id} Mse is {mse_value}')

        cv2.imwrite(f'image_{id}_recreated.jpg', filtered_image)

        '''    
        cv2.imshow('Target Image', target_image)
        cv2.imshow('Filtered Image', filtered_image)
        '''

    if id==3:
        filtered_image = cv2.medianBlur(original_image, 11)

        # Calculate squared differences
        filtered_array=np.array(filtered_image)
        squared_diff = (target_array - filtered_array) ** 2
        mse_value = np.mean(squared_diff)
        print(f'Image_{id} Mse is {mse_value}')

        cv2.imwrite(f'image_{id}_recreated.jpg', filtered_image)

        '''
        cv2.imshow('Target Image', target_image)
        cv2.imshow('Filtered Image', filtered_image)
        '''

    if id==4:
        # Define custom kernel for average filtering
        kernel_size = (15, 1)
        kernel = np.ones(kernel_size, np.float32) / (kernel_size[0] * kernel_size[1])

        padding_height = kernel_size[0] #// 2
        padding_width = kernel_size[0] #// 2
        
        # Apply zero padding
        padded_image = cv2.copyMakeBorder(original_image, padding_height, padding_height, padding_width, padding_width, cv2.BORDER_CONSTANT, value=65)

        # Apply the filter 
        filtered_image = cv2.filter2D(padded_image, -1, kernel)

        # Ensure filtered image has the same dimensions as the original image
        filtered_image = filtered_image[padding_height:-padding_height, padding_width:-padding_width]

        # Calculate squared differences
        target_array = np.array(target_image)
        squared_diff = (target_array - filtered_image) ** 2
        mse_value = np.mean(squared_diff)
        print(f'Image_{id} Mse is {mse_value}')

        cv2.imwrite(f'image_{id}_recreated.jpg', filtered_image)

        '''
        cv2.imshow('Target Image', target_image)
        cv2.imshow('Filtered Image', filtered_image)        
        '''

    if id==5:
        #padding
        padding_height = original_image.shape[0] #// 2
        padding_width = original_image.shape[1] #// 2
        padded_image = cv2.copyMakeBorder(original_image, padding_height, padding_height, padding_width, padding_width, cv2.BORDER_CONSTANT, value=65)

        filtered_image = cv2.blur(padded_image,(11,11))

        # Ensure filtered image has the same dimensions as the original image
        filtered_image = filtered_image[padding_height:-padding_height, padding_width:-padding_width]
        
        filtered_image=original_image.astype(np.int16)-filtered_image.astype(np.int16)
        filtered_image[(filtered_image <= 127) | (filtered_image < 0)] += 128                
        filtered_image=filtered_image.astype(np.uint8)


        # Calculate squared differences
        filtered_array=np.array(filtered_image)
        squared_diff = (target_array - filtered_array) ** 2
        mse_value = np.mean(squared_diff)
        print(f'Image_{id} Mse is {mse_value}')

        cv2.imwrite(f'image_{id}_recreated.jpg', filtered_image)

        '''
        # Display the original and filtered images
        cv2.imshow('Target Image', target_image)
        cv2.imshow('Filtered Image', filtered_image)
        '''
        
    if id==6:        
        laplacian_kernel = np.array([[0, -1, 0],
                                 [0,  0, 0],
                                 [0, 1, 0]])

        # Apply the filter 
        filtered_image = cv2.filter2D(original_image, -1, laplacian_kernel)

        # Calculate squared differences
        filtered_array=np.array(filtered_image)
        squared_diff = (target_array - filtered_array) ** 2
        mse_value = np.mean(squared_diff)
        print(f'Image_{id} Mse is {mse_value}')

        cv2.imwrite(f'image_{id}_recreated.jpg', filtered_image)

        '''
        # Display the original and filtered images
        cv2.imshow('Target Image', target_image)
        cv2.imshow('Filtered Image', filtered_image)
        '''

    if id==7:#wrap around                                 
        # Get the dimensions of the image
        rows, columns = original_image.shape

        # Apply wrap-around padding to the image
        wrapped_image = np.zeros((rows * 3, columns), dtype=np.uint8)
        wrapped_image[rows:rows*2, :] = original_image
        wrapped_image[:rows, :] = original_image[-rows:, :]
        wrapped_image[-rows:, :] = original_image[:rows, :]

        # Create an array filled with zeros of the same size as the image
        kernel = np.zeros((rows, columns), dtype=np.uint8)

        # Fill the specified rows and column with 1
        #kernel[rows-1, columns // 2] = 1
        kernel[0, columns // 2] = 1


        # Apply the filter using OpenCV's filter2D function
        filtered_image = cv2.filter2D(wrapped_image, -1, kernel)


        # Crop the wrapped_image to the size of the original image
        filtered_image = filtered_image[rows:rows*2, :]

        # Calculate squared differences
        filtered_array=np.array(filtered_image)
        squared_diff = (target_array - filtered_array) ** 2
        mse_value = np.mean(squared_diff)
        print(f'Image_{id} Mse is {mse_value}')
        
        cv2.imwrite(f'image_{id}_recreated.jpg', filtered_image)
        '''
        # Display the original and filtered images
        cv2.imshow('Target Image', target_image)
        cv2.imshow('Filtered Image', filtered_image)
        '''
        
    if id==8:
        filtered_image=original_image

        # Calculate squared differences
        filtered_array=np.array(filtered_image)        
        squared_diff = (target_array - filtered_array) ** 2
        mse_value = np.mean(squared_diff)
        print(f'Image_{id} Mse is {mse_value}')

        cv2.imwrite(f'image_{id}_recreated.jpg', filtered_image)
        '''
        cv2.imshow('Target Image', target_image)
        cv2.imshow('Filtered Image', filtered_image)
        '''
        
    if id==9:
        
        kernel = np.array([[0, -1, 0],
                             [-1,  5, -1],
                             [0, -1, 0]])

        # Apply the filter using OpenCV's filter2D function
        filtered_image = cv2.filter2D(original_image, -1, kernel)
        
        
        # Calculate squared differences
        filtered_array=np.array(filtered_image)
        squared_diff = (target_array - filtered_array) ** 2
        mse_value = np.mean(squared_diff)
        print(f'Image_{id} Mse is {mse_value}')

        cv2.imwrite(f'image_{id}_recreated.jpg', filtered_image)

        '''
        # Display the original and filtered images
        cv2.imshow('Target Image', target_image)
        cv2.imshow('Filtered Image', filtered_image)
        '''

    return


original_image_path = f'1.jpg'
original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
for id in range(1, 10):
    image_path = f'image_{id}.jpg'
    image_ = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    find_spatial_operation_mask(original_image,image_,id)
