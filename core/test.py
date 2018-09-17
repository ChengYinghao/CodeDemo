from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def run():
    captcha_text_str = '3681'
    image = ImageCaptcha()
    captcha_image = image.generate(captcha_text_str)
    captcha_image = Image.open(captcha_image)
    captcha_image = np.array(captcha_image)
    plt.imshow(captcha_image)
    plt.show()

if __name__ == '__main__':
    run()
