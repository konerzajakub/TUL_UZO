# Import knihoven
import numpy as np         # Pro numerické operace, zejména s poli a maticemi
import cv2                 # OpenCV pro načítání a manipulaci s obrázky
import os                  # Pro práci se souborovým systémem (nalezení souborů)
import glob                # Pro snadnější vyhledávání souborů podle vzoru
import matplotlib.pyplot as plt # Pro zobrazení obrázků

# --- Konfigurace ---
#'unknown.bmp' = 'unknown.bmp'

# --- Trénovací část ---
print("Trénovací část")

seznam_pXX_souboru = sorted(glob.glob("cv10_PCA/p??.bmp"))  # serazeno

# Seznam pro uložení obrazových dat jako vektorů
vektory_obrazku = []
rozmery_obrazku = None # rozměry obrázku

# --- Krok 1 ---
print("Krok 1: Načítání šedotónových obrázků, převod na vektory")
for image_path in seznam_pXX_souboru:
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # načtení obrázku v šedotónech
    rozmery_obrazku = img_gray.shape                        # uložení rozměru obrázku
    img_vector = img_gray.flatten()                         # zploštění matice obrázku do jednoho vektoru (řádky za sebou)
    vektory_obrazku.append(img_vector)
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
# E = E / np.linalg.norm(E, axis=0)
print()

# --- Krok 9 ---
print("Krok 9: Projekce známých vektorů do vlastního prostoru PI = ET * W")
PI = vlastni_prostor_EigenSpace.T @ W
print(f"   Projekce PI trénovacích dat vypočítána, rozměry: {PI.shape}")
print()

print("Trénovací část dokončena.\n")

# --- Testovací část ---
print("Spouštím testovací část...")

# 1) Načtení neznámého obrázku, převod na šedotónový, vytvoření vektoru wpu
print("Krok 1 (test): Načítání neznámého obrázku a převod na vektor wpu...")
# Načtení neznámého obrázku v šedotónovém režimu
unknown_img_gray = cv2.imread("cv10_PCA/unknown.bmp", cv2.IMREAD_GRAYSCALE)

# Zkontrolujeme rozměry neznámého obrázku
if unknown_img_gray.shape != rozmery_obrazku:
     raise ValueError(
         f"Neznámý obrázek '{'unknown.bmp'}' má jiné rozměry ({unknown_img_gray.shape}) "
         f"než trénovací obrázky ({rozmery_obrazku})."
     )

# Zploštění na vektor
wpu = unknown_img_gray.flatten()
print(f"   Neznámý obrázek načten a převeden na vektor wpu, délka: {wpu.shape[0]}")
print()

# 2) Centrování neznámého vektoru wu
print("Krok 2 (test): Centrování neznámého vektoru (wu = wpu - wp)...")
wu = wpu - wp.flatten()
print(f"   Neznámý vektor centrován do wu, délka: {wu.shape[0]}")
print()

# 3) Projekce neznámého vektoru PT do vlastního prostoru
print("Krok 3 (test): Projekce neznámého vektoru wu do vlastního prostoru (výpočet PT)...")
PT = vlastni_prostor_EigenSpace.T @ wu
print(f"   Projekce PT neznámého vektoru vypočítána, rozměry: {PT.shape}")
print()

# 4) Porovnání neznámého příznakového vektoru PT se známými PI
print("Krok 4 (test): Porovnání PT s PI pomocí Euklidovské vzdálenosti...")

# Efektivní výpočet všech vzdáleností najednou pomocí broadcastingu
# PT potřebujeme jako sloupcový vektor (N_eigenvect, 1)
PT_col = PT.reshape(-1, 1)
# PI má rozměry (N_eigenvect, N_images)
# PI - PT_col odečte PT od každého sloupce PI
# np.linalg.norm(..., axis=0) spočítá normu (délku) každého výsledného sloupce
distances = np.linalg.norm(PI - PT_col, axis=0)

# Najdeme index minimální vzdálenosti
best_match_index = np.argmin(distances)
# Získáme název souboru nejlépe odpovídajícího obrázku z původního seznamu
identified_image_path = seznam_pXX_souboru[best_match_index]
# Získáme minimální vzdálenost
min_distance = distances[best_match_index]

print(f"   Nalezen nejbližší známý obrázek:")
print(f"      Index: {best_match_index}")
print(f"      Soubor: {os.path.basename(identified_image_path)}") # Zobrazíme jen název souboru
print(f"      Vzdálenost: {min_distance:.4f}")

print("Testovací část dokončena.\n")

# --- Zobrazení výsledku ---
print("Zobrazuji výsledek...")

# Načtení originálního neznámého obrázku (tentokrát barevně, pokud existuje)
# OpenCV načítá ve formátu BGR (Blue-Green-Red)
original_unknown_img_bgr = cv2.imread("cv10_PCA/unknown.bmp", cv2.COLOR_BGR2RGB)
original_unknown_img_rgb = cv2.cvtColor(original_unknown_img_bgr, cv2.COLOR_BGR2RGB)


# Načtení identifikovaného obrázku (barevně, pokud existuje)
identified_img_bgr = cv2.imread(identified_image_path, cv2.IMREAD_COLOR)
if identified_img_bgr is None:
    print(f"Varování: Nepodařilo se načíst barevný identifikovaný obrázek {identified_image_path} pro zobrazení.")
    # Zkusíme načíst šedotónově a převést
    identified_img_gray = cv2.imread(identified_image_path, cv2.IMREAD_GRAYSCALE)
    if identified_img_gray is not None:
        identified_img_rgb = cv2.cvtColor(identified_img_gray, cv2.COLOR_GRAY2RGB)
    else:
        # Pokud selže i šedotónové načtení, vytvoříme černý placeholder
        h, w = rozmery_obrazku
        identified_img_rgb = np.zeros((h, w, 3), dtype=np.uint8)
else:
    # Převedení BGR na RGB
    identified_img_rgb = cv2.cvtColor(identified_img_bgr, cv2.COLOR_BGR2RGB)


# Vytvoření okna pro zobrazení dvou obrázků vedle sebe
fig, axes = plt.subplots(1, 2, figsize=(10, 5)) # 1 řádek, 2 sloupce

# Zobrazení originálního neznámého obrázku vlevo
axes[0].imshow(original_unknown_img_rgb)
axes[0].set_title(f"Neznámý obrázek\n({'unknown.bmp'})")
axes[0].axis('off')

# Zobrazení identifikovaného (nejpodobnějšího) obrázku vpravo
axes[1].imshow(identified_img_rgb)
axes[1].set_title(f"Identifikovaný obrázek\n({os.path.basename(identified_image_path)})")
axes[1].axis('off')

plt.tight_layout()
plt.show()

print("Hotovo.")