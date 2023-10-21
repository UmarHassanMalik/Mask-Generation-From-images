
import cv2
import numpy as np
import os

image_dir = "D:/dataset/Malignantcases"
mask_dir = "D:/dataset/threshhold malig"
if not os.path.exists(mask_dir):
    os.makedirs(mask_dir)

for image_file in os.listdir(image_dir):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        _, mask = cv2.threshold(image, thresh=128, maxval=255, type=cv2.THRESH_BINARY_INV)

        image_height, image_width = image.shape
        center_width = image_width // 2
        center_height = image_height * 2 // 5 
        half_width = 190  
        
        shape_mask = np.zeros(mask.shape[:2], np.uint8)
        shape_mask[center_height - half_width:center_height + half_width,
                   center_width - half_width:center_width + half_width] = 255

        masked_image = cv2.bitwise_and(mask, mask, mask=shape_mask)
        
        # apply errosion
        kernel = np.ones((5, 5), np.uint8)
        eroded_mask = cv2.erode(masked_image, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours
        contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        
    
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  
                cv2.drawContours(contour_image, [contour], -1, (0, 0, 255), 2)

   
        contour_image_path = os.path.join(mask_dir, os.path.splitext(image_file)[0] + '_area_threshold.png')
        cv2.imwrite(contour_image_path, contour_image)

print("Mask generation complete.")
