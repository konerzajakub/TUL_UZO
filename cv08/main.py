import cv2
import matplotlib.pyplot as plt
from pathlib import Path


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



    ## Segmantace prahovani
    # histogram
    #fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
    #ax_hist.hist(img1_gray.ravel(), bins=256, range=(0, 256), color='black', alpha=0.7)
    #ax_hist.set_title("Histogram obrázku 1")
    #ax_hist.set_xlabel("Hodnota pixelu")
    #ax_hist.set_ylabel("Počet pixelů")
    #plt.show()

    # Prahování obrázku - 150 vypada ok
    _, img1_thresh = cv2.threshold(img1_gray, 150,256 , cv2.THRESH_BINARY)
    # Inverze - aby zmizely cerne puntiky
    img1_thresh = cv2.bitwise_not(img1_thresh)

    # První okno s prvnim obrazkem
    fig1, axes1 = plt.subplots(2, 2, figsize=(10, 10))

    # og prvni obrazek
    axes1[0, 0].imshow(img1)
    axes1[0, 0].set_title("Originalni obrazek (1,1)")
    axes1[0, 0].axis("off")

    # prahovany prvni obrazek
    axes1[0, 1].imshow(img1_thresh, cmap="gray")
    axes1[0, 1].set_title("Prahovany obrazek (1,2)")
    axes1[0, 1].axis("off")

    axes1[1, 0].imshow(img1)
    axes1[1, 0].set_title("Obrázek 1 (2,1)")
    axes1[1, 0].axis("off")

    axes1[1, 1].imshow(img1)
    axes1[1, 1].set_title("Obrázek 1 (2,2)")
    axes1[1, 1].axis("off")

    # Druhy okno s druhym obrazkem
    fig2, axes2 = plt.subplots(2, 2, figsize=(10, 10))

    axes2[0, 0].imshow(img2)
    axes2[0, 0].set_title("Obrázek 2 (1,1)")
    axes2[0, 0].axis("off")

    axes2[0, 1].imshow(img2)
    axes2[0, 1].set_title("Obrázek 2 (1,2)")
    axes2[0, 1].axis("off")

    axes2[1, 0].imshow(img2)
    axes2[1, 0].set_title("Obrázek 2 (2,1)")
    axes2[1, 0].axis("off")

    axes2[1, 1].imshow(img2)
    axes2[1, 1].set_title("Obrázek 2 (2,2)")
    axes2[1, 1].axis("off")

    plt.show()


main()
