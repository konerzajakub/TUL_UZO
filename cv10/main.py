# -*- coding: utf-8 -*-
"""
Identifikace neznámého obrazu pomocí PCA (Analýza hlavních komponent).

Tento skript implementuje postup identifikace obrazu ('unknown.bmp')
porovnáním s množinou známých obrazů ('p11.bmp' ... 'p33.bmp')
s využitím metody PCA a Euklidovské vzdálenosti.
"""

# Import potřebných knihoven
import numpy as np  # Pro numerické operace, zejména s poli a maticemi
from PIL import Image  # Pro načítání a manipulaci s obrázky
import os  # Pro práci se souborovým systémem (nalezení souborů)
import glob  # Pro snadnější vyhledávání souborů podle vzoru
import matplotlib.pyplot as plt  # Pro zobrazení obrázků

# --- Konfigurace ---
# Cesta ke složce s obrázky (předpokládáme, že skript je ve stejné složce)
image_folder = 'cv10_PCA'
# Vzor pro názvy známých (trénovacích) obrázků
known_image_pattern = 'p??.bmp'
# Název neznámého (testovacího) obrázku
unknown_image_name = 'unknown.bmp'

# --- Trénovací část ---
print("Spouštím trénovací část...")

# Seznam pro uložení cest k trénovacím obrázkům
known_image_files = []
# Najdi všechny soubory odpovídající vzoru v dané složce
# Použijeme glob.glob pro nalezení všech souborů p??.bmp
pattern_path = os.path.join(image_folder, known_image_pattern)
known_image_files = sorted(glob.glob(pattern_path))  # Seřadíme pro konzistentní pořadí

# Zkontrolujeme, zda byly nalezeny nějaké trénovací obrázky
if not known_image_files:
    raise FileNotFoundError(
        f"Nebyly nalezeny žádné trénovací obrázky odpovídající vzoru '{pattern_path}'. Zkontrolujte cestu a vzor.")

# Seznam pro uložení obrazových dat jako vektorů
image_vectors = []
image_shape = None  # Pro uložení rozměrů obrázku (pro pozdější rekonstrukci/zobrazení)

# 1) Načtení obrázků, převod na šedotónové, vytvoření vektorů
print("Krok 1: Načítání trénovacích obrázků a převod na vektory...")
for image_path in known_image_files:
    # Otevření obrázku pomocí Pillow
    img = Image.open(image_path)
    # Převedení obrázku na šedotónový ('L' mode)
    img_gray = img.convert('L')
    # Převedení obrázku na NumPy pole (matici pixelů)
    img_array = np.array(img_gray)

    # Uložení rozměrů prvního obrázku (předpokládáme, že všechny mají stejné)
    if image_shape is None:
        image_shape = img_array.shape
        print(f"   Detekované rozměry obrázku: {image_shape}")

    # Zploštění matice obrázku do jednoho vektoru (řádky za sebou)
    # img_array.flatten() vytvoří 1D pole z 2D matice
    img_vector = img_array.flatten()
    # Přidání vektoru do seznamu
    image_vectors.append(img_vector)

# 2) Vytvoření matice Wp ze známých vektorů
print("Krok 2: Vytvoření trénovací matice Wp...")
# np.vstack vytvoří matici, kde každý řádek je jeden vektor obrázku
# Potřebujeme ale matici, kde sloupce jsou vektory, proto transpozice .T
# Alternativně a efektivněji lze použít np.column_stack
Wp = np.column_stack(image_vectors)
# Wp má rozměry (počet_pixelů, počet_obrázků)
# Např. pro 64x64 obrázky je počet_pixelů = 4096
# Pokud máme 9 obrázků p11-p33, bude rozměr (4096, 9)
num_pixels, num_known_images = Wp.shape
print(f"   Matice Wp vytvořena, rozměry: {Wp.shape}")

# 3) Spočítání průměrného vektoru wp
print("Krok 3: Výpočet průměrného vektoru wp...")
# Spočítá průměr podél řádků (pro každý pixel přes všechny obrázky)
# axis=1 znamená průměrování přes sloupce (tj. přes všechny obrázky)
# keepdims=True zachová výsledek jako sloupcový vektor (počet_pixelů, 1)
wp = np.mean(Wp, axis=1, keepdims=True)
# wp má rozměry (počet_pixelů, 1), např. (4096, 1)
print(f"   Průměrný vektor wp vypočítán, rozměry: {wp.shape}")

# 4) Vytvoření matice W (centrované údaje)
print("Krok 4: Centrování dat - vytvoření matice W...")
# Od každého sloupce (vektoru obrázku) v Wp odečteme průměrný vektor wp
# Díky NumPy broadcasting se wp automaticky odečte od každého sloupce
W = Wp - wp
# W má stejné rozměry jako Wp, např. (4096, 9)
print(f"   Matice W (centrovaná data) vytvořena, rozměry: {W.shape}")

# 5) Vytvoření kovarianční matice C
print("Krok 5: Výpočet kovarianční matice C...")
# Použijeme "trik" pro výpočet menší kovarianční matice C = W^T * W
# Výsledná matice C má rozměry (počet_obrázků, počet_obrázků), např. (9, 9)
# Je to výpočetně mnohem efektivnější než počítat C' = W * W^T (4096x4096)
# Znak '@' je operátor pro maticové násobení v NumPy (od Python 3.5+)
C = W.T @ W
print(f"   Kovarianční matice C vypočítána, rozměry: {C.shape}")

# 6) Výpočet vlastních čísel a vlastních vektorů matice C
print("Krok 6: Výpočet vlastních čísel a vektorů matice C...")
# np.linalg.eig spočítá vlastní čísla (D) a vlastní vektory (Epom) matice C
# D je pole vlastních čísel (např. 9 prvků)
# Epom je matice, kde sloupce jsou vlastní vektory C (např. 9x9)
# Vlastní vektory Epom odpovídají vlastním číslům v D ve stejném pořadí
D, Epom = np.linalg.eig(C)
print(f"   Vlastní čísla (D) a vlastní vektory (Epom) matice C nalezeny.")
print(f"      Rozměry D: {D.shape}")
print(f"      Rozměry Epom: {Epom.shape}")

# 7) Seřazení vlastních vektorů podle velikosti vlastních čísel
print("Krok 7: Seřazení vlastních vektorů Epom podle vlastních čísel...")
# Získáme indexy, které by seřadily vlastní čísla D sestupně (od největšího)
# np.argsort vrací indexy pro vzestupné řazení, [::-1] obrátí pořadí
sorted_indices = np.argsort(D)[::-1]
# Seřadíme vlastní čísla (pro informaci, není nutně potřeba dál)
D_sorted = D[sorted_indices]
# Seřadíme sloupce matice Epom podle těchto indexů -> dostaneme Ep
Ep = Epom[:, sorted_indices]
print(f"   Vlastní vektory seřazeny do matice Ep, rozměry: {Ep.shape}")

# 8) Vytvoření matice vlastního prostoru E (EigenSpace)
print("Krok 8: Vytvoření báze vlastního prostoru E (Eigenfaces)...")
# Vlastní vektory původní velké kovarianční matice (W @ W.T) získáme jako E = W @ Ep
# Tyto vektory se často nazývají "eigenfaces"
# E má rozměry (počet_pixelů, počet_vlastních_vektorů), např. (4096, 9)
E = W @ Ep
print(f"   Matice vlastního prostoru E vytvořena, rozměry: {E.shape}")
# Normalizace vektorů v E (často se dělá, i když není v kuchařce explicitně)
# Každý sloupec (eigenface) by měl mít jednotkovou délku (normu)
# E = E / np.linalg.norm(E, axis=0)
# print(f"   (Volitelné: Vlastní vektory v E normalizovány)")


# 9) Projekce známých vektorů do vlastního prostoru (výpočet příznaků PI)
print("Krok 9: Projekce trénovacích dat do vlastního prostoru (výpočet PI)...")
# Pro každé známé centrované pozorování (sloupec W) spočítáme jeho souřadnice
# v bázi E. To odpovídá projekci vektoru na jednotlivé eigenfaces.
# PI = E^T * W
# PI má rozměry (počet_vlastních_vektorů, počet_obrázků), např. (9, 9)
# Každý sloupec PI je příznakový vektor jednoho známého obrázku
PI = E.T @ W
print(f"   Projekce PI trénovacích dat vypočítána, rozměry: {PI.shape}")

print("Trénovací část dokončena.\n")

# --- Testovací část ---
print("Spouštím testovací část...")

# Cesta k neznámému obrázku
unknown_image_path = os.path.join(image_folder, unknown_image_name)

# Zkontrolujeme existenci souboru
if not os.path.exists(unknown_image_path):
    raise FileNotFoundError(f"Neznámý obrázek '{unknown_image_path}' nebyl nalezen.")

# 1) Načtení neznámého obrázku, převod na šedotónový, vytvoření vektoru wpu
print("Krok 1 (test): Načítání neznámého obrázku a převod na vektor wpu...")
# Otevření obrázku
unknown_img = Image.open(unknown_image_path)
# Převod na šedotónový
unknown_img_gray = unknown_img.convert('L')
# Převod na NumPy pole
unknown_img_array = np.array(unknown_img_gray)
# Zploštění na vektor
# Musíme zajistit, že vektor má stejný počet prvků jako trénovací vektory
wpu = unknown_img_array.flatten()
# Zkontrolujeme rozměry
if wpu.shape[0] != num_pixels:
    raise ValueError(
        f"Neznámý obrázek má jiné rozměry ({wpu.shape[0]} pixelů) než trénovací obrázky ({num_pixels} pixelů).")
print(f"   Neznámý obrázek načten a převeden na vektor wpu, délka: {wpu.shape[0]}")

# 2) Centrování neznámého vektoru wu
print("Krok 2 (test): Centrování neznámého vektoru (wu = wpu - wp)...")
# Od vektoru neznámého obrázku odečteme průměrný vektor z trénovacích dat
# wp je sloupcový vektor (N, 1), wpu je 1D vektor (N,). Flatten wp pro snadné odečtení.
wu = wpu - wp.flatten()
# wu je 1D vektor stejné délky jako wpu
print(f"   Neznámý vektor centrován do wu, délka: {wu.shape[0]}")

# 3) Projekce neznámého vektoru PT do vlastního prostoru
print("Krok 3 (test): Projekce neznámého vektoru wu do vlastního prostoru (výpočet PT)...")
# Vypočteme souřadnice neznámého centrovaného vektoru wu v bázi E
# PT = E^T * wu
# PT bude vektor souřadnic (příznakový vektor) neznámého obrázku
# PT má rozměry (počet_vlastních_vektorů,), např. (9,)
PT = E.T @ wu
print(f"   Projekce PT neznámého vektoru vypočítána, rozměry: {PT.shape}")

# 4) Porovnání neznámého příznakového vektoru PT se známými PI
print("Krok 4 (test): Porovnání PT s PI pomocí Euklidovské vzdálenosti...")
# Chceme najít, který sloupec v PI (příznakový vektor známého obrázku)
# je nejblíže vektoru PT (příznakový vektor neznámého obrázku).

# Seznam pro uložení vzdáleností
distances = []
# Projdeme všechny sloupce matice PI (každý sloupec je projekce jednoho známého obr.)
for i in range(num_known_images):
    # Vezmeme i-tý sloupec z PI
    known_projection = PI[:, i]
    # Spočítáme Euklidovskou vzdálenost mezi PT a tímto sloupcem
    # np.linalg.norm(a - b) počítá Euklidovskou vzdálenost mezi vektory a a b
    distance = np.linalg.norm(PT - known_projection)
    # Uložíme vzdálenost
    distances.append(distance)

# Efektivnější výpočet všech vzdáleností najednou:
# PT potřebujeme jako sloupcový vektor pro broadcasting (N_eigenvect, 1)
# PT_col = PT.reshape(-1, 1)
# distances = np.linalg.norm(PI - PT_col, axis=0) # axis=0 počítá normu pro každý sloupec

# Najdeme index minimální vzdálenosti
# np.argmin vrátí index prvního výskytu minimální hodnoty v poli/seznamu
best_match_index = np.argmin(distances)
# Získáme název souboru nejlépe odpovídajícího obrázku z původního seznamu
identified_image_path = known_image_files[best_match_index]
# Získáme minimální vzdálenost (pro informaci)
min_distance = distances[best_match_index]

print(f"   Nalezen nejbližší známý obrázek:")
print(f"      Index: {best_match_index}")
print(f"      Soubor: {identified_image_path}")
print(f"      Vzdálenost: {min_distance:.4f}")

print("Testovací část dokončena.\n")

# --- Zobrazení výsledku ---
print("Zobrazuji výsledek...")

# Načtení originálního neznámého obrázku (barevného, pokud takový je)
# Načteme soubor znovu, abychom měli jistotu, že máme původní verzi
try:
    original_unknown_img_color = Image.open(unknown_image_path)
    print(f"   Načten původní soubor: {unknown_image_path} (režim: {original_unknown_img_color.mode})")
except FileNotFoundError:
    print(f"   CHYBA: Původní soubor {unknown_image_path} nenalezen pro zobrazení.")
    # Můžeme případně použít šedotónovou verzi jako zálohu
    original_unknown_img_color = unknown_img_gray
    print(f"   Používám šedotónovou verzi pro zobrazení neznámého obrázku.")


# Načtení identifikovaného obrázku (barevného, pokud takový je)
# Načteme soubor znovu, abychom měli jistotu, že máme původní verzi
try:
    identified_img_color = Image.open(identified_image_path)
    print(f"   Načten původní soubor: {identified_image_path} (režim: {identified_img_color.mode})")
except FileNotFoundError:
    print(f"   CHYBA: Původní soubor {identified_image_path} nenalezen pro zobrazení.")
    # Můžeme případně použít šedotónovou verzi jako zálohu (kterou jsme již načetli dříve)
    identified_img_color = Image.open(identified_image_path).convert('L') # Načteme znovu a převedeme na L pro jistotu
    print(f"   Používám šedotónovou verzi pro zobrazení identifikovaného obrázku.")


# Vytvoření okna pro zobrazení dvou obrázků vedle sebe
fig, axes = plt.subplots(1, 2, figsize=(10, 5)) # 1 řádek, 2 sloupce

# Zobrazení originálního neznámého obrázku vlevo
# Použijeme nově načtený barevný obrázek
# Matplotlib by měl automaticky správně zobrazit RGB(A) obrázky z Pillow
axes[0].imshow(original_unknown_img_color)
axes[0].set_title(f"Neznámý obrázek\n({unknown_image_name})") # Název levého podgrafu
axes[0].axis('off') # Skryjeme osy

# Zobrazení identifikovaného (nejpodobnějšího) obrázku vpravo
# Použijeme nově načtený barevný obrázek
axes[1].imshow(identified_img_color)
axes[1].set_title(f"Identifikovaný obrázek\n({os.path.basename(identified_image_path)})") # Název pravého podgrafu
axes[1].axis('off') # Skryjeme osy

# Upraví layout, aby se titulky nepřekrývaly
plt.tight_layout()
# Zobrazí okno s obrázky
plt.show()

print("Hotovo.")