import math
from PIL import Image
import numpy as np

def visualize_sample(inputs, *outputs, images_per_row = 2):
    '''
    :param inputs: torch.tensor,  B x C x H x W
    :param outputs: torch.tensor,  B x C x H x W, may contain different sizes of outputs
    :return:
    '''
    images = [inputs, *outputs]
    group_images_num = len(images)
    single_image_width, single_image_height = inputs.shape[2], inputs.shape[3]
    image_board_width = int(group_images_num * images_per_row * single_image_width )
    image_board_height = int( math.ceil( inputs.shape[0] / images_per_row ) * single_image_height )
    image_board = Image.new(mode='RGB', size=(image_board_width, image_board_height))

    for num in range(len(inputs)):
        start_width = int ( num % images_per_row ) * single_image_width * len(images)
        start_height =  int( num / images_per_row )  * single_image_height

        for j in range(len(images)):
            image_board.paste(convert_tensor_to_img(postprocess(images[j])[num]), ( start_width + j * single_image_width, start_height))

    return image_board

def convert_tensor_to_img(tensor):
    image = Image.fromarray(tensor.detach().cpu().numpy().astype(np.uint8).squeeze()).resize((100,100))
    return image

def postprocess(img):
    # [0, 1] => [0, 255]
    img = img * 255.0
    img = img.permute(0, 2, 3, 1)
    return img.int()