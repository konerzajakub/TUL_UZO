# Import knihoven
import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt

# --- Trénovací část ---
print("Trénovací část")

seznam_pXX_souboru = sorted(glob.glob("cv10_PCA/p??.bmp"))  # serazeno

# Seznam pro uložení obrazových dat jako vektorů
vektory_obrazku = []

# --- Krok 1 ---
print("Krok 1: Načítání šedotónových obrázků, převod na vektory")
for image_path in seznam_pXX_souboru:
    img_sedy = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # načtení obrázku v šedotónech
    img_vektor = img_sedy.flatten()                         # zploštění matice obrázku do jednoho vektoru (řádky za sebou)
    vektory_obrazku.append(img_vektor)
print()

# --- Krok 2 ---
print("Krok 2: Vytvoření matice Wp")
Wp = np.column_stack(vektory_obrazku)           # Spojení vektorů do matice – každý sloupec je jeden obrázek
num_pixels, num_known_images = Wp.shape         # Počet pixelů & počet obrázků
print(f"   Matice Wp, rozměry: {Wp.shape} (pixely, obrázky)")
print()

# --- Krok 3 ---
print("Krok 3: Výpočet průměrného vektoru wp")
wp = np.mean(Wp, axis=1, keepdims=True)         # průměrný vektor (průměr po řádcích)
print(f"   Průměrný vektor wp, rozměry: {wp.shape}")
print()

# --- Krok 4 ---
print("Krok 4: Vytvoření matice W – od sloupců Wp odečten wp")
W = Wp - wp                                     # od sloupců Wp odečten wp
print(f"   Matice W, rozměry: {W.shape}")
print()

# --- Krok 5 ---
print("Krok 5: Vytvoření kovarianční matice C = WT * W – velikost 9 x 9")
C = W.T @ W                                     # Malá kovarianční matice (velikost: obrázky × obrázky)
print(f"    Kovarianční matice C, rozměry: {C.shape}")
print()

# --- Krok 6 ---
print("Krok 6: Z matice C spočítány vlastní čísla a jím náležející vlastní vektory")
vlastni_cisla, vlastni_vektory = np.linalg.eig(C)   # Vypočet vlastních čísel
vlastni_cisla = np.real(vlastni_cisla)              # Vezme jen reálnou část vlastních čísel
vlastni_vektory = np.real(vlastni_vektory)          # Vezme jen reálnou část vlastních vektorů
print(f"   Vlastní čísla a vlastní vektory matice C.")
print(f"      Rozměry D: {vlastni_cisla.shape}")
print(f"      Rozměry Epom: {vlastni_vektory.shape}")
print()

# --- Krok 7 ---
print("Krok 7: Z vlastních vektorů vytvořena matice Ep")
serazene_indexy = np.argsort(vlastni_cisla)[::-1]                 # Seřazení indexů vlastních čísel sestupně
vlastni_cisla_serazene = vlastni_cisla[serazene_indexy]           # Seřazená vlastní čísla
vlastni_vektory_serazene = vlastni_vektory[:, serazene_indexy]    # Seřazené vlastní vektory
print(f"   Vlastní vektory seřazeny do matice Ep, rozměry: {vlastni_vektory_serazene.shape}")
print()

# --- Krok 8 ---
print("Krok 8: Vytvoření matice (vlastní prostor – EigenSpace) E = W * Ep – velikost 4096 x 9")
vlastni_prostor_EigenSpace = W @ vlastni_vektory_serazene         # Vlastní vektory v původním prostoru pomocí W @ Ep (rekonstrukce eigenfaces)
print(f"   Matice vlastního prostoru E vytvořena, rozměry: {vlastni_prostor_EigenSpace.shape}")
# E = E / np.linalg.norm(E, axis=0) # norma??
print()

# --- Krok 9 ---
print("Krok 9: Projekce známých vektorů do vlastního prostoru PI = ET * W")
PI = vlastni_prostor_EigenSpace.T @ W
print(f"   Projekce PI trénovacích dat vypočítána, rozměry: {PI.shape}")
print()

print("Trénovací část dokončena.\n")

# --- Testovací část ---
print("Testovací část.")

# --- Krok 1 (Test) ---
print("Krok 1: Převedení neznámého obrázku do stupně šedi a vytvoření vektoru wpu")
unknown_img_gray = cv2.imread("cv10_PCA/unknown.bmp", cv2.IMREAD_GRAYSCALE)
wpu = unknown_img_gray.flatten() # Zploštění na vektor
print(f"   Wpu, délka: {wpu.shape[0]}")
print()

# --- Krok 2 (Test) ---
print("Krok 2 (test): Vektor wu = wpu – wp")
wu = wpu - wp.flatten()
print(f"   Wu, délka: {wu.shape[0]}")
print()

# --- Krok 3 (Test) ---
print("Krok 3 (test): Projekce neznámého vektoru PT = ET * wu")
PT = vlastni_prostor_EigenSpace.T @ wu
print(f"   Projekce PT neznámého vektoru vypočítána, rozměry: {PT.shape}")
print()

# --- Krok 4 (test) - Euklidovská vzdálenost ---
print("Krok 4 (test): Porovnání známých příznakových vektorů PI(i) a neznámého PT – např. dle minimální vzdálenosti")

pocet_znamych_obrazku = PI.shape[1] # počet známých obrázků

min_vzdalenost = float('inf')  # nastavíme na nekonečno pro první porovnání
index_nejlepsi_shody = None      # index nejlepší shody (-1 = zatím žádná)

# Pro každý sloupec PI
for i in range(pocet_znamych_obrazku):
    vektor_projekce_znamy = PI[:, i] # Vezmeme se jeden vektor

    # Výpočet Euklidovské vzdálenosti mezi PT a jedním vektorem PI
    vzdalenost = np.linalg.norm(PT - vektor_projekce_znamy) # np.linalg.norm(a - b) = Euklid. vzdálenost

    # Pokud je aktuální vzdálenost menší než dosavadní minimum (TATO PODMÍNKA ZŮSTÁVÁ)
    if vzdalenost < min_vzdalenost:
        min_vzdalenost = vzdalenost    # Aktualizujeme minimální vzdálenost
        index_nejlepsi_shody = i       # Aktualizujeme index nejlepší shody


cesta_identifikovaneho_obrazku = seznam_pXX_souboru[index_nejlepsi_shody] # Cestu k souboru nejlépe odpovídajícího obrázku

print(f"   Nejbližší obrázek z datasetu: {index_nejlepsi_shody}")
print(f"   Vzdálenost: {min_vzdalenost:.4f}")

# --- Zobrazení výsledku ---

# Načtení originálního neznámého obrázku (barevně)
puvodni_neznamy_img_bgr = cv2.imread("cv10_PCA/unknown.bmp", cv2.IMREAD_COLOR) # Načtení jako BGR
puvodni_neznamy_img_rgb = cv2.cvtColor(puvodni_neznamy_img_bgr, cv2.COLOR_BGR2RGB) # Převod BGR -> RGB

# Načtení identifikovaného obrázku (barevně)
identifikovany_img_bgr = cv2.imread(cesta_identifikovaneho_obrazku, cv2.IMREAD_COLOR)
identifikovany_img_rgb = cv2.cvtColor(identifikovany_img_bgr, cv2.COLOR_BGR2RGB) # Převod BGR -> RGB

# Vytvoření okna pro zobrazení dvou obrázků vedle sebe
fig, osy = plt.subplots(1, 2, figsize=(10, 5)) # 1 řádek, 2 sloupce

# Zobrazení originálního neznámého obrázku vlevo
osy[0].imshow(puvodni_neznamy_img_rgb)
osy[0].set_title(f"Neznámý obrázek\n({'unknown.bmp'})")
osy[0].axis('off') # Skrytí os

# Zobrazení identifikovaného (nejpodobnějšího) obrázku vpravo
osy[1].imshow(identifikovany_img_rgb)
osy[1].set_title(f"Identifikovaný obrázek\n({os.path.basename(cesta_identifikovaneho_obrazku)})")
osy[1].axis('off') # Skrytí os

plt.show()