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


def simple_averaging_filter(img, kernel_size=3):
    """
    Metoda prostého průměrování - implementace pomocí ruční konvoluce
    s maskou obsahující stejné váhy pro všechny pixely.
    """
    # Získání rozměrů obrazu
    height, width = img.shape[:2]

    # Vytvoření výstupního obrazu (kopie původního)
    output = np.zeros((height, width), dtype=np.uint8)

    # Velikost okolí (poloměr kernelu)
    offset = kernel_size // 2

    # Procházení obrazu (mimo okraje, aby kernel nepřetékal)
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            # Vybrání okolí pixelu
            neighborhood = img[y - offset:y + offset + 1, x - offset:x + offset + 1]

            # Výpočet průměru hodnot
            new_value = np.mean(neighborhood)

            # Uložení do výstupního obrazu
            output[y, x] = int(new_value)

    return output


def rotating_mask_filter(img, kernel_size=3, threshold=30):
    """
    Metoda s rotující maskou - pro každý pixel vybere nejlepší masku
    na základě nejmenšího rozptylu hodnot v okolí
    """
    result = np.copy(img)
    padding = kernel_size // 2

    # Rozšíření obrazu o okraje
    padded_img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_REFLECT)

    height, width = img.shape

    # Definice různých masek (horizontální, vertikální, diagonální)
    masks = [
        np.zeros((kernel_size, kernel_size)),  # horizontální
        np.zeros((kernel_size, kernel_size)),  # vertikální
        np.zeros((kernel_size, kernel_size)),  # diagonální 1
        np.zeros((kernel_size, kernel_size))  # diagonální 2
    ]

    # Vytvoření masek
    masks[0][kernel_size // 2, :] = 1  # horizontální
    masks[1][:, kernel_size // 2] = 1  # vertikální
    for i in range(kernel_size):  # diagonální 1 (levý horní -> pravý dolní)
        masks[2][i, i] = 1
    for i in range(kernel_size):  # diagonální 2 (pravý horní -> levý dolní)
        masks[3][i, kernel_size - 1 - i] = 1

    # Normalizace masek
    for i in range(4):
        masks[i] = masks[i] / np.sum(masks[i])

    # Pro každý pixel v obraze
    for y in range(height):
        for x in range(width):
            # Extrakce oblasti kolem pixelu
            region = padded_img[y:y + kernel_size, x:x + kernel_size]

            # Výpočet průměrů a rozptylů pro každou masku
            averages = []
            variances = []
            for mask in masks:
                # Výpočet váženého průměru
                weighted_sum = np.sum(region * mask)
                averages.append(weighted_sum)

                # Výpočet rozptylu pro danou masku
                variance = np.sum(((region - weighted_sum) ** 2) * mask)
                variances.append(variance)

            # Výběr masky s nejmenším rozptylem
            min_var_idx = np.argmin(variances)
            result[y, x] = averages[min_var_idx]

    return result


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
    padding = kernel_size // 2

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
    avg_filtered = simple_averaging_filter(original, kernel_size=5)
    rotating_filtered = rotating_mask_filter(original, kernel_size=5)
    median_filtered = median_filter_manual(original, kernel_size=5)

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