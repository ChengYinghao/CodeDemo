from captcha.image import ImageCaptcha
import numpy as np
from PIL import Image

# Content used to generate a captcha starts with numbers only
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

number_and_alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                       'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                       'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R','S', 'T',
                       'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
                       'e', 'f', 'g', 'h', 'i','j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r','s', 't',
                       'u', 'v', 'w', 'x', 'y', 'z']


def random_captcha_text(captcha_content, captcha_size):
    captcha_text = []
    for i in range(captcha_size):
        captcha_text.append(np.random.choice(captcha_content))
    return captcha_text


# Randomly generate a verification code image and text
def gen_captcha_text_image(captcha_content=number_and_alphabet, captcha_size=4):
    image = ImageCaptcha()
    captcha_text = random_captcha_text(captcha_content, captcha_size)
    captcha_text_str = ''.join(captcha_text)
    captcha_image = image.generate(captcha_text_str)
    captcha_image = Image.open(captcha_image)
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image


def text2vec(text, char_set_len=len(number_and_alphabet), max_captcha=4):
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


def convert2gray(img):
    if len(img.shape) > 2:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img


def get_next_batch(batch_size, image_height, image_width, char_set_len, max_captcha):
    batch_x = np.zeros([batch_size, image_height * image_width])
    batch_y = np.zeros([batch_size, max_captcha * char_set_len])

    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_text_image()
            if image.shape == (60, 160, 3):
                return text, image

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        image = convert2gray(image)

        batch_x[i, :] = image.flatten() / 255
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y
