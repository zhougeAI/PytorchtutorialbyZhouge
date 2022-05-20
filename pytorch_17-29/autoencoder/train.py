import numpy as np
import torch
import torchvision.utils

from network import Autoencoder
from dataset import MyDataset
from torch.utils.data import DataLoader
from utils import visualize_sample, postprocess
import torchvision.transforms.functional as F
import tqdm
from PIL import Image


# 我们把 整个训练过程写成一个类，方便管理
class Autoencoder_training():
    def __init__(self):
        # 初始化里面，定义网络， 数据集
        self.autoencoder = Autoencoder()
        # 读取模型, load_state_dict 方法就是用来读取存储好的模型状态的
        saved_state_dict = torch.load('epoch_4.pth')
        # 从saved_state_dict 键值对中取出模型和训练轮数
        self.model = saved_state_dict['model']
        self.epoch = saved_state_dict['epoch']
        self.autoencoder.load_state_dict(self.model)
        self.train_dataset = MyDataset(dataset_path=r'D:\dataset\fruit_fresh_rotten\train\freshapples')
        self.eval_dataset = MyDataset(dataset_path=r'D:\dataset\fruit_fresh_rotten\test\freshapples')
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=16, shuffle=True)
        self.eval_dataloader = DataLoader(self.eval_dataset, batch_size=1, shuffle=True)
        # sample iterator 是从eval_dataset得到的一个迭代器，方便用来采样
        self.sample_num = 10
        self.sample_iterator = self.eval_dataset.get_iterator()
        # 训练部分还需要什么？
        # 1. 是优化器
        # 2. 损失函数
        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=0.001)
        self.loss_fn = torch.nn.L1Loss()
        self.max_train_epochs = 100
        self.save_model_interval = 2
        self.sample_interval = 1
        self.eval_interval = 1


    def train(self):
        # train函数， 训练，第一步是把数据从数据导入器中获取，
        # 第二步， 开始训练
        # 模型搬到GPU上
        self.autoencoder.to('cuda:0')
        for current_train_epochs in range(self.epoch, self.max_train_epochs):
            # current_train_epochs 当前的轮数
            # 遍历的方式把train_dataloader中的数据取出来
            # ，原理：dataloader是一个生成器（可迭代对象），python支持遍历地取出生成器中的东西
            with tqdm.tqdm(total=self.train_dataset.__len__(),desc=f'Epoch{current_train_epochs} / {self.max_train_epochs}', unit='img') as pbar:
                for item in self.train_dataloader:
                        # 将取出的项赋值给images变量，并使用.cuda()方法，搬运到GPU上
                        images = item.cuda()
                        # 首先把模型调成训练模式
                        self.autoencoder.train()
                        # 走一个前向传播
                        outputs, embeddings = self.autoencoder(images)
                        # 注意： outputs 和 embeddings 都是在 一个计算图里面的，
                        # 它们有 gradient 属性，这个gradient属性和计算图有关，也就是前面的模型
                        # 因此 如果要把它当值来用， 就要用到 outputs.detach()方法
                        # 计算损失函数
                        loss = self.loss_fn(images, outputs)
                        # 接下来，最关键的一步，反向传播，使用优化器更新参数
                        # 清零 优化器的梯度记录
                        self.optimizer.zero_grad()
                        # 用loss.backward()方法反向传播，求损失函数的导数
                        loss.backward()
                        # 根据导数来更新参数， 使用optimizer.step()方法
                        self.optimizer.step()
                        # 训练完一些数据，使文本进度条更新这么多数据数
                        pbar.update(item.shape[0])
                        pbar.set_postfix(loss_batch=loss.item())
                # print("epoch:", current_train_epochs, '\n', "loss:", loss.item())

            # 在特定的 轮数 存储模型
            if current_train_epochs % self.save_model_interval == 0:
                # torch.save方法 可以保存模型， model.state_dict()可以以dict的形式保存模型当前的信息
                torch.save({'model': self.autoencoder.state_dict(),
                            'epoch': current_train_epochs,
                            'loss': loss.item()}, "epoch_{}.pth".format(current_train_epochs))

            # 定义一个sample方法
            if current_train_epochs % self.sample_interval == 0:
                # sample
                self.sample(current_train_epochs)

            # 定义一个验证方法
            if current_train_epochs % self.eval_interval == 0 :
                self.eval(current_train_epochs)

    def sample(self, current_train_epochs):
        self.autoencoder.eval()
        items, outputs, embeddings = [], [], []
        for i in range(0, self.sample_num):
            item = next(self.sample_iterator).cuda()
            output, embedding = self.autoencoder(item)
            items.append(item)
            outputs.append(output)
            embeddings.append(embedding)
        items = torch.stack(items, dim=0).squeeze(1)
        outputs = torch.stack(outputs, dim=0).squeeze(1)
        embeddings = torch.stack(embeddings, dim=0).view(-1,1,10,10)

        image_board = visualize_sample(items, outputs, embeddings,images_per_row=3)
        image_board.save('{}.png'.format(current_train_epochs))
        print("saving sample {}.png at xxx".format(current_train_epochs))

    def eval(self, current_train_epochs):
        # eval方法用于训练过程中验证，在验证集上跑一轮结果，然后计算Loss和评价指标
        # 从验证集提取数据
        total_loss = []
        # 使用tqdm 添加文本进度条
        with tqdm.tqdm(total=len(self.eval_dataset), desc=f'Epoch{current_train_epochs} / {self.max_train_epochs}',
                       unit='img', colour='green') as pbar:
            for item in self.eval_dataloader:
                    self.autoencoder.eval()
                    item = item.cuda()
                    # 前向传播
                    output, embedding = self.autoencoder(item)
                    # 使用output计算损失函数
                    loss = self.loss_fn(item, output)
                    total_loss.append(loss.item())
                    pbar.update(item.shape[0])
                    pbar.set_postfix(loss_batch= loss.item())

        # 计算平均Loss,psnr
        eval_loss = sum(total_loss) / self.eval_dataset.__len__()
        print("epoch:", current_train_epochs, "eval loss:", eval_loss)

        # 计算 一个评价指标 比如 Cosine Similarity
        # Peak Signal Noise Ratio
        # 如果显存不够，怎么节约显存？
        # 1. 使用cpu()来验证
        #  2. 删除掉一些用完了的变量，来节约显存

    def test(self, input_path, save_result_path):
        # 模型训练完后，使用训练好的模型处理数据，得到结果
        # 在 测试的时候，模型设置为eval()模式
        print("Start testing ...")
        self.autoencoder.eval()
        # 等价于self.autoencoder.train(False)
        # 读入数据
        # 如果想用GPU，将images = images.cuda(), self.autoencoder.to("cuda:0")就可以了
        image = self.read_image(input_path)
        # 利用模型进行前向传播
        output, embedding = self.autoencoder(image)
        # 得到了输出结果，接下来的事情就是处理输出结果
        # 将结果保存为图像.png
        self.save_results(output, save_result_path)
        print("End testing ....")

    def read_image(self, path):
        image = Image.open(path)
        # 预处理
        import torchvision
        transform = torchvision.transforms.Compose([torchvision.transforms.Resize((100,100)),
                                                torchvision.transforms.ToTensor()])
        image = transform(image)
        image = image.unsqueeze(0)

        return image

    def save_results(self, output, path):
        #  tensor 变为 PIL.Image 就可以了
        # output = output.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        # output = output * 255.
        # output = Image.fromarray(output.astype(np.uint8))
        # output.save(path)
        # 直接用torchvision.utils 提供的save_image 方法，可以直接将 tensor保存为Image
        torchvision.utils.save_image(output,path)

if __name__ == '__main__':
    autoencoder_training = Autoencoder_training()
    autoencoder_training.test(input_path='1.png', save_result_path='1_result.png')
