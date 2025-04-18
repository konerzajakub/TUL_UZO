import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft2, fftshift

# nacteni obrazku
obrazek = cv2.imread('cv04c_robotC.bmp', cv2.IMREAD_GRAYSCALE)
if obrazek is None:
    raise FileNotFoundError("Obrázek není ve složce.")


# Vypocet spektra
def vypocti_spektrum(obrazek):
    fft_obrazek = fft2(obrazek) # Výpočet FFT
    fft_posunuto = fftshift(fft_obrazek) # Posunutí nulové frekvence do středu
    magnitudove_spektrum = np.log(1 + np.abs(fft_posunuto)) # Výpočet magnitudy a převod na logaritmické měřítko
    return magnitudove_spektrum


# Funkce pro zobrazení výsledků v matici 2x2
def zobraz_vysledky(puvodni, detekovane_hrany, nazev):
    plt.figure(figsize=(12, 10))

    # Původní obrázek (vlevo nahoře)
    plt.subplot(221)
    plt.imshow(puvodni, cmap='gray')
    plt.title('Původní obrázek')
    plt.axis('off')

    # Spektrum původního obrázku (vpravo nahoře)
    plt.subplot(222)
    spektrum_puvodniho = vypocti_spektrum(puvodni)
    plt.imshow(spektrum_puvodniho, cmap='jet')
    plt.title('Spektrum původního obrázku')
    plt.axis('off')

    # Detekované hrany (vlevo dole) - nyní také v 'jet' barevné mapě
    plt.subplot(223)
    plt.imshow(detekovane_hrany, cmap='jet')
    plt.title(f'{nazev} detektor hran')
    plt.axis('off')

    # Spektrum detekovaných hran (vpravo dole)
    plt.subplot(224)
    spektrum_hran = vypocti_spektrum(detekovane_hrany)
    plt.imshow(spektrum_hran, cmap='jet')
    plt.title(f'Spektrum {nazev}')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# 1. Laplaceův hranový detektor
# aproximuje 2. derivaci
def laplaceuv_detektor(obrazek):
    vyska, sirka = obrazek.shape
    vystup = np.zeros_like(obrazek, dtype=np.float64)

    # Laplaceův operátor - maska 3x3
    maska = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=np.float64)

    # Aplikace masky
    for y in range(1, vyska - 1):
        for x in range(1, sirka - 1):
            suma = 0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    suma += obrazek[y + i, x + j] * maska[i + 1, j + 1]
            vystup[y, x] = suma



    # Normalizace pro zobrazení
    vystup -= vystup.min()  # Posunutí minima na 0
    vystup = (vystup / vystup.max()) * 255  # Přepočet na rozsah 0–255
    vystup = vystup.astype(np.uint8)

    return vystup


# 2. Sobelův hranový detektor
def sobeluv_detektor(obrazek):
    vyska, sirka = obrazek.shape
    vystup = np.zeros_like(obrazek, dtype=np.float64)

    # Sobelovy masky pro detekci hran ve směru x a y
    maska_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    maska_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    # Aplikace masek
    for y in range(1, vyska - 1):
        for x in range(1, sirka - 1):
            suma_x = 0
            suma_y = 0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    suma_x += obrazek[y + i, x + j] * maska_x[i + 1, j + 1]
                    suma_y += obrazek[y + i, x + j] * maska_y[i + 1, j + 1]

            # velikost gradientu
            vystup[y, x] = np.sqrt(suma_x ** 2 + suma_y ** 2)

    # Normalizace pro zobrazení
    vystup -= vystup.min()  # Posunutí minima na 0
    vystup = (vystup / vystup.max()) * 255  # Přepočet na rozsah 0–255
    vystup = vystup.astype(np.uint8)

    #vystup = np.clip(vystup, 0, 255)
    #vystup = vystup.astype(np.uint8)

    return vystup


# 3. Kirschův hranový detektor
def kirschuv_detektor(obrazek):
    vyska, sirka = obrazek.shape
    vystup = np.zeros_like(obrazek, dtype=np.float64)

    # Kirschovy masky pro 8 směrů
    masky = [
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),  # Sever
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),  # Severovýchod
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),  # Východ
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),  # Jihovýchod
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),  # Jih
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),  # Jihozápad
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),  # Západ
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]])  # Severozápad
    ]

    # Aplikace masek
    for y in range(1, vyska - 1):
        for x in range(1, sirka - 1):
            max_hodnota = 0
            for maska in masky:
                suma = 0
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        suma += obrazek[y + i, x + j] * maska[i + 1, j + 1]
                max_hodnota = max(max_hodnota, abs(suma))
            vystup[y, x] = max_hodnota

    # Normalizace pro zobrazení
    #vystup = vystup / vystup.max() * 255
    #vystup = np.clip(vystup, 0, 255)
    #vystup = vystup.astype(np.uint8)

    vystup -= vystup.min()  # Posunutí minima na 0
    vystup = (vystup / vystup.max()) * 255  # Přepočet na rozsah 0–255
    vystup = vystup.astype(np.uint8)

    return vystup


# Aplikace detektorů
laplace_hrany = laplaceuv_detektor(obrazek)
sobel_hrany = sobeluv_detektor(obrazek)
kirsch_hrany = kirschuv_detektor(obrazek)

# Zobrazení výsledků pro každý detektor
zobraz_vysledky(obrazek, laplace_hrany, "Laplace")
zobraz_vysledky(obrazek, sobel_hrany, "Sobel")
zobraz_vysledky(obrazek, kirsch_hrany, "Kirsch")

print("Detekce hran dokončena.")