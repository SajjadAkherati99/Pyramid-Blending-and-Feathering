import numpy as np
import cv2

def equalize_size(img1, img2):
    if np.prod(img1.shape) < np.prod(img2.shape):
        shape = img1.shape
        img2 = cv2.resize(img2, (shape[1], shape[0]))
    else:
        shape = img2.shape
        img1 = cv2.resize(img1, (shape[1], shape[0]))
    return img1, img2


def alpha_blending(img1, img2, dc=0, d=0):
    mask1 = np.zeros(img1.shape)
    mask2 = np.zeros(img1.shape)
    x_start = np.int(img1.shape[0]/2)+dc
    D = np.int(d/2)
    mask1[0:x_start-D, :, :] = 1
    mask2[x_start+D:, :, :] = 1
    for i in range(0,d):
        if (x_start + i - D < img1.shape[0]):
            mask1[i+x_start-D, :, :] = (d-i-1)/d
        if (x_start-i+D < img1.shape[0]):
            mask2[x_start-i+D, :, :] = (d-i-1)/d
    img_out = np.multiply(img1, mask1) + np.multiply(img2, mask2)
    return img_out.astype('int')


def make_pyramid(img, k_size=(5, 5), sigma_x=0, sigma_y=0, num_of_levels=5):
    img1 = cv2.GaussianBlur(img.astype('uint8'), k_size, sigma_x, sigmaY=sigma_y)
    img_level0 = img-img1.astype('int')

    if num_of_levels == 1:
        return img

    shape = img1.shape
    img2 = cv2.resize(img1, (np.int(shape[1] / 2), np.int(shape[0] / 2)))
    img1 = cv2.GaussianBlur(img2.astype('uint8'), k_size, sigma_x, sigmaY=sigma_y)
    img_level2 = img2-img1.astype('int')

    if num_of_levels == 2:
        return  img_level0, img1

    shape = img1.shape
    img2 = cv2.resize(img1, (np.int(shape[1] / 2), np.int(shape[0] / 2)))
    img1 = cv2.GaussianBlur(img2.astype('uint8'), k_size, sigma_x, sigmaY=sigma_y)
    img_level4 = img2 - img1.astype('int')

    if num_of_levels == 3:
        return  img_level0, img_level2, img1

    shape = img1.shape
    img2 = cv2.resize(img1, (np.int(shape[1] / 2), np.int(shape[0] / 2)))
    img1 = cv2.GaussianBlur(img2.astype('uint8'), k_size, sigma_x, sigmaY=sigma_y)
    img_level8 = img2 - img1.astype('int')

    if num_of_levels == 4:
        return  img_level0, img_level2, img_level4, img1

    shape = img1.shape
    img2 = cv2.resize(img1, (np.int(shape[1] / 2), np.int(shape[0] / 2)))
    img_level16 = img2.astype('int')

    return img_level0, img_level2, img_level4, img_level8, img_level16


def pyramid_blending(img1, img2, dc=0, d=0, num_of_level=4):
    if (num_of_level==1):
        img1_level0 = make_pyramid(img1, num_of_levels=num_of_level)
        img2_level0 = make_pyramid(img2, num_of_levels=num_of_level)
        img_out = alpha_blending(img1_level0, img2_level0, dc=dc, d=d)

        return img_out

    elif (num_of_level == 2):
        img1_level0, img1_level2 = make_pyramid(img1, num_of_levels=num_of_level)
        img2_level0, img2_level2 = make_pyramid(img2, num_of_levels=num_of_level)

        img_out = alpha_blending(img1_level2, img2_level2, dc=int(dc / 2), d=d)
        shape = img1_level0.shape
        img_out = cv2.resize(img_out, (shape[1], shape[0]))

        img_out += alpha_blending(img1_level0, img2_level0, dc=dc, d=d)

        return img_out

    elif (num_of_level == 3):
        img1_level0, img1_level2, img1_level4 = make_pyramid(img1, num_of_levels=num_of_level)
        img2_level0, img2_level2, img2_level4 = make_pyramid(img2, num_of_levels=num_of_level)

        img_out = alpha_blending(img1_level4, img2_level4, dc=int(dc / 4), d=d)
        shape = img1_level2.shape
        img_out = cv2.resize(img_out, (shape[1], shape[0]))

        img_out += alpha_blending(img1_level2, img2_level2, dc=int(dc / 2), d=d)
        shape = img1_level0.shape
        img_out = cv2.resize(img_out, (shape[1], shape[0]))

        img_out += alpha_blending(img1_level0, img2_level0, dc=dc, d=d)

        return img_out

    elif (num_of_level==4):
        img1_level0, img1_level2, img1_level4, img1_level8 = make_pyramid(img1, num_of_levels=num_of_level)
        img2_level0, img2_level2, img2_level4, img2_level8 = make_pyramid(img2, num_of_levels=num_of_level)

        img_out = alpha_blending(img1_level8, img2_level8, dc=int(dc/8), d=d)
        shape = img1_level4.shape
        img_out = cv2.resize(img_out, (shape[1], shape[0]))

        img_out += alpha_blending(img1_level4, img2_level4, dc=int(dc/4), d=d)
        shape = img1_level2.shape
        img_out = cv2.resize(img_out, (shape[1], shape[0]))

        img_out += alpha_blending(img1_level2, img2_level2, dc=int(dc/2), d=d)
        shape = img1_level0.shape
        img_out = cv2.resize(img_out, (shape[1], shape[0]))

        img_out += alpha_blending(img1_level0, img2_level0, dc=dc, d=d)

        return img_out

    elif (num_of_level == 5):
        img1_level0, img1_level2, img1_level4, img1_level8, img1_level16 = \
            make_pyramid(img1, num_of_levels=num_of_level, sigma_x=0, sigma_y=0)
        img2_level0, img2_level2, img2_level4, img2_level8, img2_level16 = \
            make_pyramid(img2, num_of_levels=num_of_level, sigma_x=0, sigma_y=0)

        img_out = alpha_blending(img1_level16, img2_level16, dc=int(dc / 16), d=d)
        shape = img1_level8.shape
        img_out = img_out.astype('uint8')
        img_out = cv2.resize(img_out, (shape[1], shape[0]))
        img_out = img_out.astype('int')

        img_out += alpha_blending(img1_level8, img2_level8, dc=int(dc / 8), d=d)
        shape = img1_level4.shape
        img_out = img_out.astype('uint8')
        img_out = cv2.resize(img_out, (shape[1], shape[0]))
        img_out = img_out.astype('int')

        img_out += alpha_blending(img1_level4, img2_level4, dc=int(dc / 4), d=d)
        shape = img1_level2.shape
        img_out = img_out.astype('uint8')
        img_out = cv2.resize(img_out, (shape[1], shape[0]))
        img_out = img_out.astype('int')

        img_out += alpha_blending(img1_level2, img2_level2, dc=int(dc / 2), d=d)
        shape = img1_level0.shape
        img_out = img_out.astype('uint8')
        img_out = cv2.resize(img_out, (shape[1], shape[0]))
        img_out = img_out.astype('int')

        img_out += alpha_blending(img1_level0, img2_level0, dc=dc, d=d)
        img_out[img_out>255] = 255
        img_out[img_out < 0] = 0
        img_out = img_out.astype('uint8')

        return img_out

    else:
        raise Exception("num_of_level must be an integer between 1 to 5")

img1 = cv2.imread('1.source.jpg')
img2 = cv2.imread('2.target.jpg')
img1, img2 = equalize_size(img1, img2)
img3 = pyramid_blending(img1.astype('int'), img2.astype('int'), dc=20, d=30, num_of_level=5)
cv2.imwrite('res2.jpg', img3)
