import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def colors(image):
    """
        Vystupem funkce je matice s ciselnymi hodnotami jednotlivych oblasti
    """
    rows, cols = image.shape
    visited = np.zeros_like(image, dtype=bool)
    colors = np.zeros_like(image, dtype=np.uint8)
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    color_count = 1  # Začínáme od 1 kvůli maskám později

    for i in range(rows):
        for j in range(cols):
            if not visited[i, j] and image[i, j] == 1:
                stack = [(i, j)]
                visited[i, j] = True
                colors[i, j] = color_count

                while stack:
                    current_x, current_y = stack.pop()

                    for dx, dy in directions:
                        new_x, new_y = current_x + dx, current_y + dy
                        if 0 <= new_x < rows and 0 <= new_y < cols:
                            if not visited[new_x, new_y] and image[new_x, new_y] == 1:
                                stack.append((new_x, new_y))
                                visited[new_x, new_y] = True
                                colors[new_x, new_y] = color_count
                color_count += 1
    return colors


def mass_center(image):
    """
        Vezme nejvetsi hodnotu z matice a postupne projde vsechny oblasti, pro ktere vypocita teziste
    """
    points = []
    max_region = np.max(image)

    for i in range(1, max_region + 1):
        #zkopiruje jen oblast s danym indexme
        copy = np.zeros_like(image)
        copy[image == i] = 1

        moments = cv2.moments(copy, True) # vypocet momentu
        if moments["m00"] != 0:
            center_x = int(moments["m10"] / moments["m00"]) # m00 plocha objektu, m10 a m01 suma vsech x a y souradnic pixelu
            center_y = int(moments["m01"] / moments["m00"])
            points.append((center_x, center_y))
    return points


def main():
    img_path = Path("res/cv09_rice.bmp").resolve()

    origo_image = cv2.imread(str(img_path))
    origo_image = cv2.cvtColor(origo_image, cv2.COLOR_BGR2GRAY)

    origo_image_histogram = cv2.calcHist([origo_image], [0], None, [256], [0, 256])
    origo_normalized_histogram = origo_image_histogram / np.max(origo_image_histogram) * 255
    origo_flattened = np.ndarray.flatten(origo_normalized_histogram)
    origo_smoothed = np.convolve(origo_flattened, np.ones(3) / 3, mode='same')

    # # puvodni obrazek
    # plt.imshow(cv2.cvtColor(origo_image, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()

    # # Plot the smoothed histogram
    # plt.plot(origo_smoothed, color='blue')
    # plt.title('Smoothed Histogram')
    # plt.xlabel('Pixel Intensity')
    # plt.ylabel('Frequency')
    # plt.show()


    # predzpracovani pomoci top-hat
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    top_hat = cv2.morphologyEx(origo_image, cv2.MORPH_TOPHAT, kernel)
    top_hat_histogram = cv2.calcHist([top_hat], [0], None, [256], [0, 256])
    top_hat_normalized_histogram = top_hat_histogram / np.max(top_hat_histogram) * 255
    top_hat_flattened = np.ndarray.flatten(top_hat_normalized_histogram)
    top_hat_smoothed = np.convolve(top_hat_flattened, np.ones(3) / 3, mode='same')

    top_hat_segmentation = np.zeros_like(top_hat)
    target = 60
    top_hat_segmentation[top_hat < target] = 0
    top_hat_segmentation[top_hat >= target] = 1

    sections = colors(top_hat_segmentation)
    centers = mass_center(sections)
    without_zero = len(centers) - 1

    fig, ax = plt.subplots(2, 3, figsize=(18, 8))

    # origo_image
    ax[0, 0].imshow(origo_image, cmap='gray')
    ax[0, 0].set_title('Original Image')
    ax[0, 0].axis('off')

    # origo_image histogram
    ax[0, 1].plot(origo_smoothed, color='blue')
    ax[0, 1].set_title('Original Image Histogram')
    ax[0, 1].set_xlabel('Pixel Intensity')
    ax[0, 1].set_ylabel('Frequency')

    # top_hat image
    ax[1, 0].imshow(top_hat, cmap='gray')
    ax[1, 0].set_title('Top-Hat Image')
    ax[1, 0].axis('off')

    # top_hat image histogram
    ax[1, 1].plot(top_hat_smoothed, color='blue')
    ax[1, 1].set_title('Top-Hat Image Histogram')
    ax[1, 1].set_xlabel('Pixel Intensity')
    ax[1, 1].set_ylabel('Frequency')

    # origo_image s centry
    ax[0, 2].imshow(origo_image, cmap='gray')
    for center in centers:
        ax[0, 2].plot(center[0], center[1], 'ro')
    ax[0, 2].set_title('Original Image with Centers of Mass')
    ax[0, 2].axis('off')
    
    valid_centers = []
    for i, center in enumerate(centers, start=1):
        region_size = np.sum(sections == i)
        if region_size > 90:
            valid_centers.append(center)

    ax[0, 2].set_title(f'Pocet zrn {len(valid_centers)}')

    
    ax[1, 2].axis('off')


    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()