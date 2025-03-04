import cv2
import numpy as np
import matplotlib.pyplot as plt

# Disketni Fourierova transformace

def main():
    # ./obrazky/cv04c_robotC.bmp
    image = cv2.imread("./res/cv04c_robotC.bmp", cv2.IMREAD_GRAYSCALE)
    print(image.shape)

    # ze zadani
    fft2 = np.fft.fft2(image)
    #plt.imshow(np.log(np.abs(fft2)))
    spektrum = np.abs(fft2)
    upravene_spektrum = np.fft.fftshift(spektrum)

    # vykresleni 
    plt.subplot(1, 2, 1)
    plt.imshow(np.log(spektrum), cmap='jet')
    plt.title('Spektrum')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(np.log(upravene_spektrum), cmap='jet')
    plt.title('Posunute kvadranty')
    plt.colorbar()

    plt.show()
    plt.savefig("./output/01_dft_spectrum.png", dpi=300)


if __name__ == "__main__":
    main()