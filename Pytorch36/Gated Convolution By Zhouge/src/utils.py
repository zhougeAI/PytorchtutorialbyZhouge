import os
from PIL import Image
import numpy as np

def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return

def tensor2image(tensor):
    '''
    Convert a tensor to a image.
    :param tensor: the tensor needed to be convert to images
    :return:
    '''
    return (( tensor + 1 ) * 127.5).transpose(1,2).transpose(2,3).detach().cpu().numpy()


def save_tensor_as_image(tensor, save_dir):
        '''
        Save an image tensor to be an image file.
        :param tensor: a tensor to be saved as image file, size B X C X W X H
        :param save_dir: save dir, example: 'xxx.png'
        :return:
        '''
        tensorarrary = tensor2image(tensor).astype(np.uint8)
        for singleimage in tensorarrary:
            image = Image.fromarray(singleimage) # size B X W X H X C
            image.save(save_dir)


def stitch_images(inputs, *outputs, img_per_row=2):
    gap = 5
    columns = len(outputs) + 1

    width, height = inputs[0][:, :, 0].shape
    img = Image.new('RGB', (width * img_per_row * columns + gap * (img_per_row - 1), height * int(len(inputs) / img_per_row)))
    images = [inputs, *outputs]

    for ix in range(len(inputs)):
        xoffset = int(ix % img_per_row) * width * columns + int(ix % img_per_row) * gap
        yoffset = int(ix / img_per_row) * height

        for cat in range(len(images)):
            im = np.array((images[cat][ix]).cpu()).astype(np.uint8).squeeze()
            im = Image.fromarray(im)
            img.paste(im, (xoffset + cat * width, yoffset))

    return img


def imsave(img, path):
    im = Image.fromarray(img.cpu().numpy().astype(np.uint8).squeeze())
    im.save(path)
