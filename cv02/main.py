import cv2
import numpy as np


def zpetna_projekce(hsv_image, hist):
    hue = hsv_image[:, :, 0] # získání jen HUE kanálu z HSV obrázku => [všechny řádky, všechny sloupce, kanál 0]
    projekce = np.zeros(hue.shape, dtype=np.float32) # nulová matice o stejný velikosti obrázek

    # průchod každým pixelem v obrázku
    for i in range(hue.shape[0]):  # pro řádky
        for j in range(hue.shape[1]):  # pro sloupce
            hue_value = hue[i, j]  # hodnoty HUE kanálu v pixelu (i, j)
            projekce[i, j] = hist[hue_value, 0]  # Do projekce uložíme hodnotu z histogramu odpovídající této barvě

    return projekce


def vypocet_teziste(projekce):
    vyska, sirka = projekce.shape  # rozměr - shape vrací (počet řádků, počet sloupců)

    y_souradnice, x_souradnice = np.indices((vyska, sirka)) # vytvoření mřížky souřadnic prvků v matici

    # vážené součtů
    sum_P = np.sum(projekce)  # suma všech pravděpodobností
    sum_xP = np.sum(x_souradnice * projekce)
    sum_yP = np.sum(y_souradnice * projekce)

    # x souřadnice těžiště
    if sum_P != 0:
        xt = sum_xP / sum_P
    else:
        xt = 0

    # y souřadnice těžiště
    if sum_P != 0:
        yt = sum_yP / sum_P
    else:
        yt = 0

    return (xt, yt)

# Načtení videa a vzoru hrníčku
cap = cv2.VideoCapture('cv02_hrnecek.mp4')
vzor_hrnecek = cv2.imread('cv02_vzor_hrnecek.bmp')

# velikost obrázku vzoru hrníčku
w, h = 104, 145  # Šířka a výška v pixelech

# Histogram ze vzoru hrníčku
hsv_vzor = cv2.cvtColor(vzor_hrnecek, cv2.COLOR_BGR2HSV)  # převod do HSV
hist = cv2.calcHist([hsv_vzor], [0], None, [180], [0, 180])  # histogram pro Hue kanál
hist = hist / hist.max()  # normalizce 0-1

# První frame pro inicializaci
ret, frame = cap.read()  # první frame videa
hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # převedeme si frame na HSV
projekce = zpetna_projekce(hsv_frame, hist)  # vypočítáme projekci prvního framu

# najdeme těžiště prvního framu
xt, yt = vypocet_teziste(projekce)
# okno hrníčku prvního framu
track_window = (int(xt - w / 2), int(yt - h / 2), w, h)  # (x, y, šířka, výška)

while True:
    ret, frame = cap.read()
    if not ret: break

    # hledáme další těžiště jen v oblasti hrníčku
    x, y, rw, rh = track_window
    roi = frame[y:y + rh, x:x + rw]

    # projekce ve vybrané oblasti
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    projekce_roi = zpetna_projekce(hsv_roi, hist)

    # aktualizace nového těžiště
    xt_roi, yt_roi = vypocet_teziste(projekce_roi)

    if not np.isnan(xt_roi) and not np.isnan(yt_roi):
        # převod souřadnic do kontextu celýho obrázku
        xt_global = x + xt_roi
        yt_global = y + yt_roi

        # změna vybraný oblasti
        track_window = (
            max(0, int(xt_global - w / 2)),  # x-souřadnice levého horního rohu
            max(0, int(yt_global - h / 2)),  # y-souřadnice
            w, h  # pořád chceme jen původní velikost
        )

    # vykreslení obdélníku kolem hrníčku
    x, y, _, _ = track_window
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Zelený obdélník
    cv2.imshow('Hrnecek trackovani', frame)  # Zobrazení obrázku

    if cv2.waitKey(30) == 27: break

cv2.destroyAllWindows()