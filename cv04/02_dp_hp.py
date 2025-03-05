import cv2
import numpy as np
import matplotlib.pyplot as plt

# Filtrovani obrazu dolni a horni propusti

def main():
    image = cv2.imread("./res/cv04c_robotC.bmp", cv2.IMREAD_GRAYSCALE)
    # spektrum DFT a presun do stredu
    robot = np.fft.fftshift(np.fft.fft2(image))
    #plt.imshow(np.log(np.abs(robot)), cmap='gray')
    #plt.title("Robot Image Spectrum")
    #plt.show()

    # nacteni filtru
    filtrDP = cv2.imread("./res/cv04c_filtDP.bmp", cv2.IMREAD_GRAYSCALE) # propousti nizke f
    filtrDP1 = cv2.imread("./res/cv04c_filtDP1.bmp", cv2.IMREAD_GRAYSCALE)
    filtrHP = cv2.imread("./res/cv04c_filtHP.bmp", cv2.IMREAD_GRAYSCALE) # vysoke f
    filtrHP1 = cv2.imread("./res/cv04c_filtHP1.bmp", cv2.IMREAD_GRAYSCALE)
    # ulozeni pro snazsi algoritmus
    filters = [filtrDP, filtrDP1, filtrHP, filtrHP1]

    # vytvoreni spekter, potlaceni frekvenci dle filtru
    spectrums = []
    for filtr in filters:
        spectrums.append(robot * filtr)

    # zpetna transformace
    images = []
    for spectrum in spectrums:
        # nizke frekvence zpet do rohu, inverzi dft 
        images.append(np.abs(np.fft.ifft2(np.fft.ifftshift(spectrum))))


    rows, cols = 4, 4
    figure, axes = plt.subplots(rows, cols, figsize=(10, 10))

    for x in range(4):
        axes[x, 0].imshow(np.log(np.abs(spectrums[x])), cmap='gray')
        axes[x, 1].imshow(image, cmap="gray")
        axes[x, 2].imshow(np.log(np.abs(spectrums[x])), cmap='jet')
        axes[x, 3].imshow(images[x], cmap='gray')

    plt.gcf().canvas.manager.set_window_title('02 DP HP')
    plt.show()

if __name__ == "__main__":
    main()