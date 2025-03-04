import cv2
import numpy as np
import matplotlib.pyplot as plt

# Filtrovani obrazu dolni a horni propusti

def main():
    image = cv2.imread("./res/cv04c_robotC.bmp", cv2.IMREAD_GRAYSCALE) / 255
    # spektrum
    robot = np.fft.fftshift(np.fft.fft2(image))
    #plt.imshow(np.log(np.abs(robot)), cmap='gray')
    #plt.title("Robot Image Spectrum")
    #plt.show()

    # nacteni filtru
    filtrDP = cv2.imread("./res/cv04c_filtDP.bmp", cv2.IMREAD_GRAYSCALE) / 255
    filtrDP1 = cv2.imread("./res/cv04c_filtDP1.bmp", cv2.IMREAD_GRAYSCALE) / 255
    filtrHP = cv2.imread("./res/cv04c_filtHP.bmp", cv2.IMREAD_GRAYSCALE) / 255
    filtrHP1 = cv2.imread("./res/cv04c_filtHP1.bmp", cv2.IMREAD_GRAYSCALE) / 255
    # ulozeni pro snazsi algoritmus
    filters = [filtrDP, filtrDP1, filtrHP, filtrHP1]

    # vytvoreni spekter
    spectrums = []
    for filtr in filters:
        spectrums.append(robot * filtr)

    # zpetna transformace
    images = []
    for spectrum in spectrums:
        images.append(np.abs(np.fft.ifft2(np.fft.ifftshift(spectrum))))


    rows, cols = 4, 4
    figure, axes = plt.subplots(rows, cols, figsize=(10, 10))

    for x in range(4):
        axes[x, 0].imshow(np.log(np.abs(spectrums[x])), cmap='gray')
        axes[x, 1].imshow(image, cmap="gray")
        axes[x, 2].imshow(np.log(np.abs(spectrums[x])), cmap='jet')
        axes[x, 3].imshow(images[x], cmap='gray')

    plt.show()

if __name__ == "__main__":
    main()