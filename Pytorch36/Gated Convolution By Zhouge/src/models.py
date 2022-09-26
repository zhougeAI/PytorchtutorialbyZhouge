import torch
import torch.nn as nn
from .networks import InpaintSANet, InpaintSADiscriminator
from .loss import ReconLoss, SNGenLoss, SNDisLoss
import os

class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0

        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')

    def load(self):
        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']

        # load discriminator only when training
        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)

            self.discriminator.load_state_dict(data['discriminator'])

    def save(self):
        print('\nsaving %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, self.gen_weights_path)

        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, self.dis_weights_path)



class GatedConv(BaseModel):
    def __init__(self,config):
        super(GatedConv, self).__init__('GatedConv',config)
        # networks
        generator = InpaintSANet(config)
        discriminator = InpaintSADiscriminator(config)

        # multi-GPU settings
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator, config.GPU)

        # losses
        recon_loss = ReconLoss(*(config.L1_LOSS_ALPHA))
        generator_loss = SNGenLoss(config.GAN_LOSS_ALPHA)
        discriminator_loss = SNDisLoss()

        # add_modules and losses to the model
        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)
        self.add_module('recon_loss', recon_loss)
        self.add_module('generator_loss', generator_loss)
        self.add_module('discriminator_loss', discriminator_loss)

        # optimizers
        self.gen_optimizer = torch.optim.Adam(generator.parameters(), lr=float(config.LR), weight_decay=self.config.WEIGHT_DECAY)
        self.dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=float(self.config.D2G_LR) * float(config.LR), weight_decay=self.config.WEIGHT_DECAY)

    def forward(self, images, masks, img_exs=None):
        coarse_output, fine_output = self.generator(images,masks,img_exs)

        return coarse_output, fine_output


    def backward(self, gen_loss=None, dis_loss=None ):
        dis_loss.backward(retain_graph=True)
        gen_loss.backward(retain_graph=True)
        self.dis_optimizer.step()
        self.gen_optimizer.step()


    def process(self, images, masks):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        # process outputs
        gen_loss = 0
        dis_loss = 0

        # inference
        coarse_imgs, recon_imgs = self.generator(images, masks)
        # conbine and concate
        complete_imgs = recon_imgs * masks + images * (1 - masks)
        outputs = complete_imgs.detach()

        pos_imgs = torch.cat([images, masks, torch.full_like(masks, 1.)], dim=1)
        neg_imgs = torch.cat([complete_imgs, masks, torch.full_like(masks, 1.)], dim=1)
        pos_neg_imgs = torch.cat([pos_imgs, neg_imgs], dim=0)

        # calculate the loss
        pred_pos_neg = self.discriminator(pos_neg_imgs)
        pred_pos, pred_neg = torch.chunk(pred_pos_neg, 2, dim=0)

        # pred_neg = self.discriminator(neg_imgs)

        # d_loss
        d_loss = self.discriminator_loss(pred_pos, pred_neg)
        dis_loss += d_loss

        # gan loss
        gan_loss = self.generator_loss(pred_neg)
        gen_loss += gan_loss

        # reconstruction loss
        r_loss = self.recon_loss(images, coarse_imgs, recon_imgs, masks)
        gen_loss += r_loss


        return outputs, gen_loss, dis_loss