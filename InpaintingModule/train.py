import torch
import torch.optim as optim
import gc
from src.model.generator import Generator
from src.model.discriminator import Discriminator
from src.utils.logger import Logger
from src.utils.trainer import Trainer
from src.utils.notify_me import *
from src.dataset.dataloader import *
from torchvision.transforms import transforms
from torch.nn import init
from env_var import *  # Ensure this file contains the necessary variables
import argparse

def weights_init(init_type='kaiming', init_gain=0.2):
    """Initialize network weights.
    Parameters:
        init_type (str)  -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float) -- scaling factor for normal, xavier, and orthogonal.
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError(f'Initialization method [{init_type}] is not implemented')

        elif 'BatchNorm2d' in classname:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

        elif 'InstanceNorm2d' in classname:
            if m.weight is not None:
                init.normal_(m.weight.data, 1.0, 0.02)
            if m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        elif 'Linear' in classname:
            init.normal_(m.weight.data, 0, 0.01)
            init.constant_(m.bias.data, 0)


    return init_func

def train(train_loader, start_epoch=1, num_epochs=2, device='cuda'):
    # Initialize Generator and Discriminator
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    if start_epoch == 1:
        generator.apply(weights_init())
        discriminator.apply(weights_init())
        print("Weights initialized")

    # Optimizers with learning rates
    opt_coarse = optim.Adam(generator.coarse_generator.parameters(), lr=5e-5, betas=(0.5, 0.99))
    opt_refine = optim.Adam(generator.refine_generator.parameters(), lr=5e-5, betas=(0.5, 0.99))
    opt_disc = optim.Adam(discriminator.parameters(), lr=5e-5, betas=(0.5, 0.99))

    sched_coarse = optim.lr_scheduler.LambdaLR(opt_coarse, lambda epoch: 0.95 ** (epoch // 3))
    sched_refine = optim.lr_scheduler.LambdaLR(opt_refine, lambda epoch: 0.95 ** (epoch // 3))
    sched_disc = optim.lr_scheduler.LambdaLR(opt_disc, lambda epoch: 0.90 ** (epoch // 2))
    
    # Initialize Logger
    logger = Logger(TRAIN_CSV_PATH, 'train', CHKPT_DIR)
    # logger.load_checkpoint(start_epoch - 1, generator, discriminator)

    if start_epoch > 1:
        logger.load_checkpoint(start_epoch - 1, generator, discriminator, 
                               opt_coarse, opt_refine, opt_disc, 
                               sched_coarse, sched_refine, sched_disc)
        
    # Initialize Trainer and start training
    trainer = Trainer(generator, discriminator, 
                      opt_coarse, opt_refine, opt_disc, 
                      sched_coarse, sched_refine, sched_disc,
                      train_loader, logger, device)
    trainer.train(start_epoch, num_epochs)

    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_epoch', type = int, default = -1, help = 'Training starts from epoch [start_epoch]')
    parser.add_argument('--num_epochs', type = int, default = 0, help = 'Training continues for [num_epochs] epochs')
    parser.add_argument('--device', type = str, default = 'cuda', help = 'Training [device]')
    parser.add_argument('--batch_size', type = int, default = 16, help = 'Define Batch Size')
    parser.add_argument('--img_size', type = int, default = 512, help = 'Define Image Size')
    parser.add_argument('--notify', type = bool, default = False, help = "Send notification to telegram")
    opt = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((opt.img_size, opt.img_size)),
        transforms.ToTensor()
    ])

    num_workers = os.cpu_count()
    train_dataset = ObjectRemovalDataset(data_dir=f"{DATA_DIR}", data_type="train", transform=transform, mask_threshold=1.0)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=num_workers)
    # print(len(train_loader))
    train(train_loader, opt.start_epoch, opt.num_epochs, opt.device)

    if opt.notify:    
        send_notification(0, 0, 0, False, True, False, False)