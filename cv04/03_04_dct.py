import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn

# Diskretni cos transformace

def limit_dct_spectrum(dct, size):
    dct_limited = np.zeros_like(dct)  # Vytvoříme nulové spektrum stejné velikosti
    
    #center_y = dct.shape[0] // 2
    #center_x = dct.shape[1] // 2
    #half = size // 2
#
    #for y in range(center_y - half, center_y + half):
    #    for x in range(center_x - half, center_x + half):
    #        dct_limited[y, x] = dct[y, x]

    for y in range(size):
        for x in range(size):
            dct_limited[y, x] = dct[y, x]

    return dct_limited


def main():
    # nacteni obrazku
    original_image = cv2.imread("./res/cv04c_robotC.bmp", cv2.IMREAD_GRAYSCALE)
    #plt.imshow(original_image)
    #plt.show()
    # origo dct spektrum
    original_dct_spectrum = dctn(original_image, norm='ortho')

    # pozadovane velikosti
    sizes = [10, 30, 50]
    print(len(sizes))

    dct = []
    img = []
    for size in sizes:
        # pro kazdy obrazek vytvoreni omezeneho spektra a zpetna transformace
        dct_limited = limit_dct_spectrum(original_dct_spectrum, size)
        inverse_image = idctn(dct_limited, norm='ortho')

        dct.append(dct_limited)
        img.append(inverse_image)

    # vyploteni vysledku
    plt.figure(figsize=(20, 20))

    plt.subplot(len(dct)+1, 2, 1)
    plt.imshow(np.log(np.abs(original_dct_spectrum)), cmap='jet')
    plt.title('Original DCT')
    plt.colorbar()

    plt.subplot(len(dct)+1, 2, 2)
    plt.imshow(original_image, cmap='grey')


    for x in range(len(dct)):
        plt.subplot(len(dct)+1, 2, (x+1)*2 + 1)
        plt.imshow(np.log(np.abs(dct[x])), cmap='jet')
        plt.title(f'Size: {sizes[x]}')
        plt.colorbar()

        plt.subplot(len(dct)+1, 2, (x+1)*2 + 2)
        plt.imshow(img[x], cmap='grey')

    plt.tight_layout()
    plt.gcf().canvas.manager.set_window_title('03 04 DCT')
    plt.show()

if __name__ == "__main__":
    main()