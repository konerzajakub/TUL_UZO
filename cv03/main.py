import cv2
import numpy as np
import math
import os

#rotation_matrix = np.array([[np.cos(), 0], [0, 0]])

def read_image(path):

    # is a valid path?
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such file: '{path}'")
    
    image = cv2.imread(path)
    
    # no image found
    if image is None:
        raise ValueError(f"Failed to read image from: '{path}'")
    
    return image


def rotate_image(image, angle):
    h = image.shape[0]
    w = image.shape[1]
    
    #img_center = (w/2, h/2)
    img_center = np.array([w/2, h/2])
    x_center = w/2
    y_center = h/2
    print(img_center)
    
    angle_rad = math.radians(angle) # need radians
    print(f'Angle in radians: {angle_rad}')

    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])
    print(f'Rotation matrix: \n{rotation_matrix}')

    inverse_rotation_matrix = np.linalg.inv(rotation_matrix)
    print(f'Inverse matrix: \n{inverse_rotation_matrix}')

    # matrix for new image with same size
    rotated_image = np.zeros_like(image)

    # loop through lines
    for y in range(h):
        # loop through columns
        for x in range(w):

            # center coordinates
            original_coords = np.array([x - x_center, y - y_center])
            
            # rotate coordinates
            new_coords = np.dot(inverse_rotation_matrix, original_coords)
            
            # calculate new regular coordinates
            if(1):
                new_x = new_coords[0] + x_center
                new_y = new_coords[1] + y_center
            else:
                new_x = new_coords[0] #+ x_center
                new_y = new_coords[1] #+ y_center
            
            # check bounds
            if 0 <= new_x < w and 0 <= new_y < h:
                rotated_image[y, x] = image[int(new_y), int(new_x)]
                
    return rotated_image


def main():
    img = read_image("./res/cv03_robot.bmp")
    
    for i in range(0,361,45):
        print(i)
        rotated_img = rotate_image(img, i)
        cv2.imshow(f"Rotated by {i} degrees", rotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("-------")

    
if __name__ == "__main__":
    main()