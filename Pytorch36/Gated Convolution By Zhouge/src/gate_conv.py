from tqdm import tqdm
from .models import GatedConv
from .dataset import Dataset
from torch.utils.data import  DataLoader
from .utils import *
from .metrics import PSNR, MAE
import time


class gated_conv():
    def __init__(self, config):
        self.config = config
        self.debug = False
        self.model_name = 'GatedConv'
        self.gatedconvmodel = GatedConv(config).to(config.DEVICE)

        # evalution metric
        self.psnr = PSNR(255.0).to(config.DEVICE)
        self.mae = MAE()

        # test mode
        if self.config.MODE == 2:
            self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_MASK_FLIST,
                                        augment=False, training=False)
        else:
            self.train_dataset = Dataset(config, config.TRAIN_FLIST, config.TRAIN_MASK_FLIST,
                                         augment=True, training=True)
            self.val_dataset = Dataset(config, config.VAL_FLIST, config.VAL_MASK_FLIST,
                                       augment=False, training=True)
            self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG == True:
            self.debug = True

    def load(self):
        self.gatedconvmodel.load()

    def save(self):
        self.gatedconvmodel.save()

    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=0,
            drop_last=True,
            shuffle=True
            # shuffle = True
        )

        epoch = 0
        keep_training = True
        max_iteration = int(float(self.config.MAX_ITERS))
        total = len(self.train_dataset)

        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        # progbar
        while(keep_training):
            epoch += 1
            # print('\n\n Training epoch: %d ' %epoch)

            with tqdm(total=total, desc=f'Epoch{epoch + int(self.gatedconvmodel.iteration / (total / self.config.BATCH_SIZE)) } / {int((max_iteration) / (total / self.config.BATCH_SIZE))}',
                      unit='img', ncols=150) as pbar:
                for items in train_loader:
                    self.gatedconvmodel.train()

                    if self.config.DEBUG == True:
                        # get data
                        images, masks = items
                    else:
                        images, masks = self.cuda(*items)

                    # process
                    outputs, gen_loss, dis_loss = self.gatedconvmodel.process(images,masks)

                    # logs
                    pbar.set_postfix({'iteration':'{}/{}'.format(self.gatedconvmodel.iteration,self.config.MAX_ITERS),
                                      'gen_loss':'{:.4f}'.format(gen_loss.item()),
                                      'dis_loss':'{:.4f}'.format(dis_loss.item())})
                    pbar.update(images.shape[0])

                    # backward
                    self.gatedconvmodel.backward(gen_loss,dis_loss)

                    iteration = self.gatedconvmodel.iteration

                    if iteration >= max_iteration:
                        keep_training = False
                        break

                    # logs
                    # if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    #     self.log(logs)

                    # sample model at checkpoints
                    if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0:
                        self.sample()

                    # evaluate model at checkpoints
                    if self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 0:
                        print('\nstart eval...\n')
                        self.eval(iteration_=iteration)

                    # save model at checkpoints
                    if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                        self.save()

        print('\nEnd training....')


    def eval(self, iteration_):
        # val_dataloader
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            drop_last=True,
            shuffle=True
        )

        total = len(self.val_dataset)
        self.gatedconvmodel.eval()

        iteration = 0
        eval_loss = {}
        eval_loss['gen_loss'] = []
        eval_loss['dis_loss'] = []
        eval_loss['psnr'] = []
        eval_loss['mae']  = []

        for items in val_loader:
            iteration += 1
            # inference
            images, masks = self.cuda(*items)
            outputs, gen_loss, dis_loss = self.gatedconvmodel.process(images, masks)
            outputs_merged = (outputs * masks) + (images * ( 1 - masks))
            # metrics
            psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
            mae = self.mae(images,outputs_merged)
            # append losses metrics to eval_loss
            eval_loss['gen_loss'].append(gen_loss.item())
            eval_loss['dis_loss'].append(dis_loss.item())
            eval_loss['psnr'].append(psnr.item())
            eval_loss['mae'].append(mae.item())

        # logs
        # log losses and metrics to val_log.dat
        with open('eval_log.txt','a')  as f:
            f.write('epoch: {} ,  eval_loss: gen_loss: {} , dis_loss: {} ,  psnr: {} , mae: {} '.format(iteration_,  sum(eval_loss['gen_loss'])/iteration , sum(eval_loss['dis_loss'])/iteration, sum(eval_loss['psnr'])/iteration, sum(eval_loss['mae']) / iteration) + '\n')
            f.close()
        # print eval log information
        print("Saving eval results at eval_log.txt......")


    def test(self):
        self.gatedconvmodel.eval()

        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1
        )

        total = len(test_loader)
        index = 0
        with tqdm(total=total, desc='Test',unit='img',colour='green') as pbar:
            for items in test_loader:
                name = self.test_dataset.load_name(index)
                images, masks = self.cuda(*items)
                index += 1
                coarse_outputs, fine_outputs = self.gatedconvmodel(images, masks)
                # coarse_outputs_merged = (coarse_outputs * masks) + (images * (1 - masks))
                fine_outputs_merged = ( fine_outputs * masks) + (images * (1 - masks))

                # coarse_output = self.postprocess(coarse_outputs_merged)[0]
                fine_output = self.postprocess(fine_outputs_merged)[0]
                path = os.path.join(self.results_path, name)
                print(index, name)
                # save
                imsave(fine_output, path)
                pbar.update(images.shape[0])

                if self.debug:
                    masked_images = self.postprocess(images * (1 - masks) + masks)[0]
                    fname, fext = name.split('.')

                    imsave(masked_images, os.path.join(self.results_path, fname + '_masked.' + fext))

            print('\nEnd test....')

    def sample(self, it=None):
        #When the validation set is empty do not sample
        if len(self.val_dataset) == 0:
            return

        self.gatedconvmodel.eval()

        items = next(self.sample_iterator)
        images, masks = self.cuda(*items)

        iteration = self.gatedconvmodel.iteration
        inputs = ( images * (1 - masks)) + masks
        coarse_outputs, fine_outputs = self.gatedconvmodel(images,masks)
        coarse_outputs_merged = ( coarse_outputs * masks ) + ( images * (1-masks) )
        fine_outputs_merged = ( fine_outputs * masks ) + ( images * (1-masks) )

        if it is not None:
            iteration = it

        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1

        images = stitch_images(
            self.postprocess(images),
            self.postprocess(inputs),
            self.postprocess(coarse_outputs_merged),
            self.postprocess(fine_outputs_merged),
            img_per_row=image_per_row
        )

        path = os.path.join(self.samples_path, self.model_name)
        name = os.path.join(path, str(iteration).zfill(5) + ".png")
        # if not exist, create
        if not os.path.exists(path):
            os.makedirs(path)

        print('\nsaving sample ' + name)
        images.save(name)

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()




