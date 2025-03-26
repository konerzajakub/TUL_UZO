import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path



def get_histogram(image, smoothing):
    # ziskani vyhlazeneho histogramu obrazku
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    normalized = histogram / np.max(histogram) * 255
    
    flattened = np.ndarray.flatten(normalized) # transformace do 1D pole

    smoothing_filter = np.ones(smoothing) / smoothing
    smoothed = np.convolve(flattened, smoothing_filter, mode='same')
    
    return smoothed



def get_threshold(array):
    # Najdi prvky co jsou mensi nez oba sousedi
    # kazdy prvek se porovna s prvkem nalevo a napravo
    is_smaller_than_left = array[1:-1] < array[:-2] # prvni a posledni prvek nemaji sousedy
    is_smaller_than_right = array[1:-1] < array[2:]
    are_local_minima = is_smaller_than_left & is_smaller_than_right

    # Ziskam indexy vsech lokalnich minim
    local_minima_indices = np.where(are_local_minima)[0]

    # Vratim prvni lokalni minimum
    first_threshold = local_minima_indices[0]
    return first_threshold



def colors(image):
    rows, cols = image.shape
    visited = np.zeros_like(image, dtype=bool)  # navstivene pixely
    colors = np.zeros_like(image, dtype=np.uint8)  # output matice s oblastmi
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # smery pro 4sousedni okoli hledani
    
    color_count = 0  # unikatni oblasti

    for i in range(rows):
        for j in range(cols):
            if not visited[i, j] and image[i, j] == 1:
                color_count += 1
                stack = [(i, j)]
                visited[i, j] = True
                colors[i, j] = color_count

                # DFS
                while stack:
                    current_x, current_y = stack.pop()  # posledni pridany pixel

                    for dx, dy in directions:  # ctyri smery
                        new_x, new_y = current_x + dx, current_y + dy

                        # zda je v mezich
                        if 0 <= new_x < rows and 0 <= new_y < cols:
                            # pokud neni navstiveny a je to objekt
                            if not visited[new_x, new_y] and image[new_x, new_y] == 1:
                                stack.append((new_x, new_y))
                                visited[new_x, new_y] = True  # navstiveny
                                colors[new_x, new_y] = color_count  # stejna barva
    return colors



def mass_center(image):
    # vypocet teziste oblasti
    points = []
    max_region = np.max(image)
    
    for i in range(2, max_region + 1):
        copy = np.zeros_like(image)
        copy[image == i] = 1
        
        moments = cv2.moments(copy, True)
        if moments["m00"] != 0:
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
            points.append([center_x, center_y])
    
    return points



def coin_values(image, points):
    values = []
    
    for point in points:
        x, y = point[0], point[1]
        region_number = image[y][x]

        region_pixels = np.argwhere(image == region_number) # binarni mapa stejnych pixelu
        number_of_pixels = len(region_pixels) # pocet pixelu pro porovnani minci
        
        # pokud ma vic jak 4000 pixelu, tak je to petikoruna
        if number_of_pixels > 4000:
            value = 5
        else:
            value = 1
        
        values.append({"point": point, "value": value})
    
    return values



def main():
    image_path = Path("./res/cv07_segmentace.bmp").as_posix()
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    red, green, blue = np.float32(img[:, :, 0]), np.float32(img[:, :, 1]), np.float32(img[:, :, 2])
    
    # uprava green dle zadani 
    g = 255 - ((green * 255) / (red + green + blue))
    g_hist = get_histogram(g, 10)
    
    # prah pro segmentaci
    threshold = get_threshold(g_hist)
    
    # segmentace 
    segmented_image = np.where(g < threshold, 0, 1)

    # ziskani oblasti
    regions = colors(segmented_image)
    
    # teziste oblasti
    points = mass_center(regions)
    
    # rozeznani mince
    values = coin_values(regions, points)


    fig, axes = plt.subplots(2, 4, figsize=(14, 8))
    
    # puvodni obrazek
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Img")
    
    # zelena slozka
    axes[0, 1].imshow(g, cmap="gray")
    axes[0, 1].set_title("Zelena")
    
    # segmentace
    axes[0, 2].imshow(segmented_image, cmap="gray")
    axes[0, 2].set_title("Segmentace")

    # histogram zelene
    axes[0, 3].plot(g_hist)
    #axes[0, 3].axvline(threshold, color="red")
    axes[0, 3].set_xlim(0, 255)
    axes[0, 3].set_ylim(0, 255)
    axes[0, 3].set_title("Green hist")
    
    # vykresleni oblasti
    axes[1, 0].imshow(regions, cmap='jet')
    axes[1, 0].set_title("Oblasti")
    
    #vykresleni tezist
    axes[1, 1].imshow(img)
    axes[1, 1].scatter(*zip(*points), marker="+", color="red")
    axes[1, 1].set_title("Teziste")
    
    # vykresleni hodnot minci
    axes[1, 2].imshow(img)
    total_val = 0
    for coin in values:
        x, y, value = coin.get("point")[0], coin.get("point")[1], coin.get("value")
        axes[1, 2].text(x, y, value, color="red", fontsize=16)
        print(f'x,y {{{x};{y}}}, mince {value}') # vypis do konzole
        total_val += value
    print(f'Celkova hodnota: {total_val} Kc')
    axes[1, 2].set_title("Hodnota oblasti")
    
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()