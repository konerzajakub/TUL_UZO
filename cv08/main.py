import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def main():

    # Nacteni obrazku
    obrazek1_cesta = Path("cv08_im1.bmp").resolve()
    obrazek2_cesta = Path("cv08_im2.bmp").resolve()

    img1 = cv2.imread(str(obrazek1_cesta))
    img2 = cv2.imread(str(obrazek2_cesta))

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)


    # histogram
    #fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
    #ax_hist.hist(img1_gray.ravel(), bins=256, range=(0, 256), color='black', alpha=0.7)
    #ax_hist.set_title("Histogram obrázku 1")
    #ax_hist.set_xlabel("Hodnota pixelu")
    #ax_hist.set_ylabel("Počet pixelů")
    #plt.show()

    ## DRUHY OBRAZEK
    # Prahování obrázku - 150 vypada ok
    _, img1_thresh = cv2.threshold(img1_gray, 150,256 , cv2.THRESH_BINARY)
    _, img2_thresh = cv2.threshold(img2_gray, 150, 256, cv2.THRESH_BINARY)

    # Inverze - aby zmizely cerne puntiky
    img1_thresh = cv2.bitwise_not(img1_thresh)
    img2_thresh = cv2.bitwise_not(img2_thresh)

    # První okno s prvnim obrazkem
    fig1, axes1 = plt.subplots(2, 2, figsize=(10, 10))

    ## TRETI OBRAZEK
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    img1_opened = cv2.morphologyEx(img1_thresh, cv2.MORPH_OPEN, kernel)
    img2_opened = cv2.morphologyEx(img2_thresh, cv2.MORPH_OPEN, kernel)

    def colors(image):
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
        points = []
        max_region = np.max(image)

        for i in range(1, max_region + 1):
            copy = np.zeros_like(image)
            copy[image == i] = 1

            moments = cv2.moments(copy, True)
            if moments["m00"] != 0:
                center_x = int(moments["m10"] / moments["m00"])
                center_y = int(moments["m01"] / moments["m00"])
                points.append((center_x, center_y))
        return points

    # ---------- hlavní část: těžiště a kreslení ----------

    def draw_mass_centers(original_image, opened_image):
        # Prevod na binarni obrazek
        binary = (opened_image > 0).astype(np.uint8)

        # Barveni objektu
        labeled = colors(binary)

        # Vypocet teziste
        centers = mass_center(labeled)

        # Puvodni obrazek, kam vyznacim teziste
        output = original_image.copy()

        # Vykresleni tezisti objektu na puvodni obrazek
        for x, y in centers:
            cv2.drawMarker(output, (x, y), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

        return output

    img1_with_centers = draw_mass_centers(img1, img1_opened)
    img2_with_centers = draw_mass_centers(img2, img2_opened)

    # og prvni obrazek
    axes1[0, 0].imshow(img1)
    axes1[0, 0].set_title("Originalni obrazek (1,1)")
    axes1[0, 0].axis("off")

    # prahovany prvni obrazek
    axes1[0, 1].imshow(img1_thresh, cmap="gray")
    axes1[0, 1].set_title("Prahovany obrazek (1,2)")
    axes1[0, 1].axis("off")

    axes1[1, 0].imshow(img1_opened, cmap="gray")
    axes1[1, 0].set_title("Otevreni (2,1)")
    axes1[1, 0].axis("off")

    axes1[1, 1].imshow(img1_with_centers)
    axes1[1, 1].set_title("Puvodni obrazek s vyznacenimi tezisti objektu (2,2)")
    axes1[1, 1].axis("off")

    # Druhy okno s druhym obrazkem
    fig2, axes2 = plt.subplots(2, 2, figsize=(10, 10))

    axes2[0, 0].imshow(img2)
    axes2[0, 0].set_title("Originalni obrazek (1,1)")
    axes2[0, 0].axis("off")

    axes2[0, 1].imshow(img2_thresh, cmap="gray")
    axes2[0, 1].set_title("Prahovany obrazek (1,2)")
    axes2[0, 1].axis("off")

    axes2[1, 0].imshow(img2_opened, cmap="gray")
    axes2[1, 0].set_title("Otevreni (2,1)")
    axes2[1, 0].axis("off")

    axes2[1, 1].imshow(img2_with_centers)
    axes2[1, 1].set_title("Puvodni obrazek s vyznacenimi tezisti objektu (2,2)")
    axes2[1, 1].axis("off")

    plt.show()


main()
