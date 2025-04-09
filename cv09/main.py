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

    image = cv2.imread(str(img_path))


    # puvodni obrazek
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


    # predzpracovani pomoci top-hat
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    plt.imshow(top_hat, cmap='gray')
    plt.axis('off')
    plt.show()
    return


if __name__ == "__main__":
    main()