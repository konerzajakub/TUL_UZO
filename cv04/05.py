import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.fft import dctn, idctn

# ZKOPIROVANE CVICENI 1 PRO BUDOUCI UPRAVU

# Načtení obrázků
def ziskej_obrazky(slozka_dir="res/C01_IT_new"):
    obrazky_ve_slozce = os.listdir(slozka_dir)
    obrazky_ve_slozce.sort() # aby 01-09
    obrazky_seznam = []
    for obrazky in obrazky_ve_slozce:
        # nacteni obrazku
        img = cv2.imread(os.path.join(slozka_dir, obrazky), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # obrazky jsou jinak modre idk
        obrazky_seznam.append(img)
    return obrazky_seznam


def vypocet_grayscale_histogramu(img):
    # převod do grayscale
    grayscale_obrazek = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # vytvoření histogramu
    histogram = cv2.calcHist([grayscale_obrazek], [0], None, [256], [0, 256])
    return histogram

def vypocet_dct(img):
    # převod do grayscale
    #plt.imshow(img)
    #plt.show()
    grayscale_obrazek = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #plt.imshow(grayscale_obrazek)
    #plt.show()
    dct = dctn(grayscale_obrazek, norm='ortho')
    # beru jen to omezene spektrum, flatten pro 1D vektor
    dct = dct[0:5, 0:5].flatten()
    return dct


def main():
    # Načtení všech obrázků
    obrazky = ziskej_obrazky()

    # Výpočet histogramů pro všechny obrázky
    histogramy = []
    dct_vectors = []
    for obrazek in obrazky:
        #histogram = vypocet_grayscale_histogramu(obrazek)
        #histogramy.append(histogram)
        dct = vypocet_dct(obrazek)
        dct_vectors.append(dct)


    # Pro každý obrázek (řádek) určení pořadí ostatních obrázků podle vzdálenosti histogramu
    sortovane_indexy_obrazku = []
    for i in range(len(obrazky)):
        #porovnany_histogram = histogramy[i]
        porovnany_dct = dct_vectors[i]
        pole_vzdalenost = []

        for j in range(len(obrazky)):
            # Výpočet vzdálenosti mezi histogramy (Bhattacharyya distance)
            #vzdalenost = cv2.compareHist(porovnany_histogram, histogramy[j], cv2.HISTCMP_BHATTACHARYYA)
            vzdalenost = np.linalg.norm(porovnany_dct - dct_vectors[j]) # euklidovska vzdalenost
            pole_vzdalenost.append((vzdalenost, j))

        # Seřazení od nejmenší hodnoty (0.0) po největší vzdálenost
        pole_vzdalenost.sort(key=lambda x: x[0])

        # Uložení indexů po seřazení
        serazene_indexy = []
        for (_, j) in pole_vzdalenost:
            serazene_indexy.append(j)

        sortovane_indexy_obrazku.append(serazene_indexy)

    # Vytvoření řádků s obrázky seřazenými podle vzdálenosti
    radky = []
    for serazene_indexy in sortovane_indexy_obrazku:
        radek = []
        for j in serazene_indexy:
            radek.append(obrazky[j])
        radky.append(radek)

    # Vytvoření a zobrazení mřížky 9x9
    fig, axes = plt.subplots(nrows=9, ncols=9, figsize=(15, 15))
    for i in range(9):
        for j in range(9):
            axes[i, j].imshow(radky[i][j])
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.gcf().canvas.manager.set_window_title('05 s DCT')
    plt.show()


if __name__ == "__main__":
    main()