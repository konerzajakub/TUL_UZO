import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_spectrum(img):
    """Zobrazí amplitudové spektrum obrázku"""
    # Převod obrázku na float32
    f = np.float32(img)

    # Aplikace FFT
    f_shift = np.fft.fftshift(np.fft.fft2(f))

    # Výpočet amplitudového spektra (v logaritmickém měřítku pro lepší vizualizaci)
    magnitude_spectrum =  np.log(np.abs(f_shift) + 1)

    # Normalizace pro zobrazení
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Aplikace barevné mapy pro lepší vizualizaci
    magnitude_spectrum_color = cv2.applyColorMap(magnitude_spectrum, cv2.COLORMAP_JET)

    return magnitude_spectrum_color


def show_histogram(img):
    """Zobrazí histogram obrázku"""
    if len(img.shape) == 3:
        # Pro barevný obrázek převedeme na grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])

    # Vytvoření obrázku histogramu
    hist_img = np.zeros((300, 256, 3), np.uint8)
    cv2.normalize(hist, hist, 0, 300, cv2.NORM_MINMAX)

    for i in range(256):
        cv2.line(hist_img, (i, 300), (i, 300 - int(hist[i])), (255, 255, 255), 1)

    return hist_img


def simple_averaging_filter(img):
    height, width = img.shape
    output = np.copy(img)

    # Procházení každého pixelu, který má okolí 3x3 (vynecháme 1px okraje)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # Extrahujeme 3x3 okolo pixelu
            window = img[y - 1:y + 2, x - 1:x + 2]
            # Vypočteme průměr a zaokrouhlíme na celé číslo
            avg = np.round(np.mean(window)).astype(np.uint8)
            # Uložení do výstupu
            output[y, x] = avg

    return output.astype(img.dtype)


def rotating_mask_filter(img):
    # Získat rozměry obrázku
    H, W = img.shape[:2]

    # Vytvořit výstupní obrázek s typem float pro přesnější výpočty
    result = img.copy().astype(np.float32)

    # Seznam všech možných pozic (8 pozic kromě středu)
    positions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1)
    ]

    # Procházet validní pixely (s okrajem 2px)
    for i in range(2, H - 2):
        for j in range(2, W - 2):
            min_var = np.inf
            best_mask = None

            # Pro každou možnou pozici masky
            for dr, dc in positions:
                # Vypočítat střed masky
                x = i - dr
                y = j - dc

                # Extrahovat 3x3 oblast kolem středu
                mask = img[x - 1:x + 2, y - 1:y + 2]

                # Výpočet rozptylu
                current_var = np.var(mask)

                # Aktualizovat nejlepší masku
                if current_var < min_var:
                    min_var = current_var
                    best_mask = mask

            # Aplikovat průměr nejlepší masky
            result[i, j] = np.mean(best_mask)

    # Konverze zpět na původní datový typ
    return result.astype(img.dtype)

def median_filter_manual(img, kernel_size=3):
    """
    Metoda mediánové filtrace - vlastní implementace
    Pro každý pixel vybere medián z okolí
    """
    # Výška a šířka obrázku
    height, width = img.shape

    # Vytvoření výstupního obrázku
    result = np.zeros_like(img)

    # Velikost "okraje" (padding) kolem centrálního pixelu
    padding = 1

    # Rozšíření obrazu o okraje pro zpracování hraničních pixelů
    padded_img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_REFLECT)

    # Pro každý pixel v originálním obrázku
    for y in range(height):
        for x in range(width):
            # Extrakce oblasti kolem pixelu
            window = padded_img[y:y + kernel_size, x:x + kernel_size]

            # Převedení 2D okna na 1D pole
            values = window.flatten()

            # Výpočet mediánu hodnot v okně
            median_value = np.median(values)

            # Přiřazení mediánu do výsledného obrázku
            result[y, x] = median_value

    return result.astype(np.uint8)


def process_image(image_path, fig_title):
    # Načtení obrázku
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Kontrola, zda byl obrázek správně načten
    if original is None:
        print(f"Chyba: Obrázek {image_path} nebyl nalezen.")
        return

    # Aplikace filtrů
    avg_filtered = simple_averaging_filter(original)
    rotating_filtered = rotating_mask_filter(original)
    median_filtered = median_filter_manual(original, kernel_size=3)

    # Příprava obrázků, spekter a histogramů
    images = [
        ("Originál", original),
        ("Prosté průměrování", avg_filtered),
        ("Rotující maska", rotating_filtered),
        ("Medián", median_filtered)
    ]

    # Vytvoření výsledné vizualizace
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    fig.suptitle(fig_title, fontsize=16)

    for i, (title, img) in enumerate(images):
        # Obrázek
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f"{title}")
        axes[i, 0].axis('off')

        # Spektrum
        spectrum = show_spectrum(img)
        axes[i, 1].imshow(cv2.cvtColor(spectrum, cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title(f"Spektrum - {title}")
        axes[i, 1].axis('off')

        # Histogram
        hist = show_histogram(img)
        axes[i, 2].imshow(cv2.cvtColor(hist, cv2.COLOR_BGR2RGB))
        axes[i, 2].set_title(f"Histogram - {title}")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()


# Zpracování obou obrázků
process_image("cv05_robotS.bmp", "Zpracování obrázku cv05_robotS.bmp")
process_image("cv05_PSS.bmp", "Zpracování obrázku cv05_PSS.bmp")