from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def run():
    captcha_text_str = '3681'
    image = ImageCaptcha()
    captcha_image = image.generate(captcha_text_str)
    captcha_image = Image.open(captcha_image)
    captcha_image = np.array(captcha_image)
    plt.imshow(captcha_image)
    plt.show()


def text2vec(text, char_set_len=len(number), max_captcha=4):
    text_len = len(text)
    if text_len > max_captcha:
        raise ValueError('验证码最长4个字符')

    vector = np.zeros(max_captcha * char_set_len)

    def char2pos(char):
        if char == '_':
            k = 62
            return k
        k = ord(char) - 48
        if k > 9:
            k = ord(char) - 55
            if k > 35:
                k = ord(char) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k

    for i, c in enumerate(text):
        idx = i * char_set_len + char2pos(c)
        vector[idx] = 1
    return vector


if __name__ == '__main__':
    text = ['w', '2', 'T', '3']
    for i, c in enumerate(text):
        print(i, c);
