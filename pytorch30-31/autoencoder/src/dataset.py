import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import glob
from PIL import Image
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
# 定义实现我们自己想要的数据集的类，MyDataset
# 这个类要继承 torch.utils.data.Dataset 类， 同时需要重写（实现）两个方法：__getitem__ 和 __len__
# 这个类很高效，它一批一批读取数据，在数据要用的时候，才加载到内存中。
#  与其把所有的数据打开再装入list里，不如直接把这些数据的路径装入list中
class MyDataset(Dataset):
    # 初始化函数, 这里定义一些我们所需要的变量
    def __init__(self, dataset_path):
        super(MyDataset, self).__init__()
        # 写一些东西，数据集的文件夹
        data_dir = dataset_path
        self.data_path_list = []
        for name in glob.glob(data_dir + '/*'):
            self.data_path_list.append(name)
        # 预处理， 写预处理方法, 预处理方法可以自己实现，我们用到的是torchvision提供的方法
        # 实现了图像的尺寸调整，和转化为张量
        self.transform = torchvision.transforms.Compose([torchvision.transforms.Resize((100,100)),
                                                         torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.RandomRotation(degrees=(0,360)),
                                                         torchvision.transforms.GaussianBlur(kernel_size=7)
                                                         ])
        # functional 把 所有 transforms类可以做的事情做成了函数，类似于nn.functional把所有层
        # 做的事情做成了函数，我们可以用类，也可以直接调用函数F

    # 需要重写的方法， 这个方法，做什么的：通过给定的标签，获得数据样本
    # item 是标签， 表示第几个数据，训练时会自动变化，增加
    def __getitem__(self, item):
        # 打开图片
        image = Image.open(self.data_path_list[item]).convert('RGB')

        # 对图像进行预处理
        image = self.transform(image)
        # 也可以直接用函数处理
        # image = F.resize(image,size=(100,100))
        # image = F.to_tensor(image)
        # image = F.rotate(image, angle=180)
        # image = F.gaussian_blur(image,kernel_size=7)

        return image

    # 需要重写，做什么的： 给出数据集中所有数据的总数
    def __len__(self):
        return len(self.data_path_list)

   # 用于采样，相当于是给出了一个迭代器，方便我们每次取数据来采样
    def get_iterator(self, batch_size = 1):
        while True:
            for items in DataLoader(dataset=self, batch_size=batch_size, shuffle=False, drop_last=True):
                yield items


if __name__ == '__main__':
    mydataset = MyDataset()
    # 实现完了数据集类，实例化之后，还不够，我们还需要用 DataLoader 类来使它可用
    # Pytorch提供了DataLoader类, 实例化
    # num_workers参数指定了数据导入器用到的线程数（可以理解为多开，开几个程序来帮你读数据），
    #一般电脑性能较好可以设置num_workers较大，不好则设为0
    mydataloader = DataLoader(dataset= mydataset, batch_size=16,
                              shuffle=True, num_workers=0)
    # 创建好了一个mydataloader对象，就可以使用了，DataLoader是一个可迭代对象
    data_iter = iter(mydataloader)
    for j in range(1):
        m = next(data_iter)
        # 使用matplotlib可视化
        m = m.permute(0, 2, 3, 1).numpy()
        for i in range(0, 16):
            plt.subplot(4, 4, i + 1)
            plt.axis('off')
            plt.imshow(m[i])
        plt.show()
        plt.close()
