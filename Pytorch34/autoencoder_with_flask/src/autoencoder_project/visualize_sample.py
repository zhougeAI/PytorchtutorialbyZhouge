import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from network import Autoencoder
from dataset import MyDataset

def visualize_sample( inputs, outputs, embedding, images_per_row=2):
    '''
    :param inputs: torch.tensor,  B x C x H x W
    :param outputs: torch.tensor,  B x C x H x W
    :param embedding: torch.tensor,  B x C x H x W
    :return:
    '''
    single_image_height, single_image_width = inputs.shape[2], inputs.shape[3]
    image_board_height = (int(inputs.shape[0] / images_per_row) * single_image_height)
    image_board_width = (int(images_per_row * 3) * single_image_width)
    image_board = Image.new(mode='RGB', size=(image_board_width, image_board_height))
    total_tensor = torch.stack((inputs, outputs, embedding), dim=1)
    for num, item in enumerate(total_tensor):
        start_width = int((num % images_per_row) * single_image_width* 3)
        start_height = (int(num / images_per_row) * single_image_height)
        image_board.paste(convert_tensor_to_img(item[0]), (start_width, start_height))
        image_board.paste(convert_tensor_to_img(item[1]),
                          (start_width + single_image_width, start_height))
        image_board.paste(convert_tensor_to_img(item[2]),
                          (start_width + 2 * single_image_width, start_height ))

    image_board.save('2.png', format='png')

def convert_tensor_to_img(tensor):
    image = Image.fromarray(tensor.permute(1, 2, 0).int().numpy().astype(np.uint8))
    return image

if __name__ == '__main__':
    dataloader = DataLoader(dataset=MyDataset(), batch_size=20, shuffle=False)
    for item in dataloader:
        images = item * 255.
        visualize_sample(images,images, images,images_per_row=6)
        break

