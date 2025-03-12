from src.model.generator import *
from src.model.discriminator import *
from src.utils.logger import *
from src.utils.validator import *
from src.utils.notify_me import *
from src.dataset.dataloader import *
from torchvision.transforms import transforms
from env_var import *
import argparse

def validate(chkpt_no, val_loader, device='cuda'):

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Initialize Logger
    logger = Logger(VAL_CSV_PATH, 'val', CHKPT_DIR)

    # Validate the model after the last epoch
    validator = Validator(generator, discriminator, logger, device)
    logger.load_checkpoint(chkpt_no, generator, discriminator)
    validator.validate(chkpt_no, val_loader)

    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--chkpt_no', type = int, default = -1, help = 'validate checkpoint no x')
    parser.add_argument('--num_chkpts', type = int, default = 1, help = 'number of checkpoint to validate with step of 2')
    parser.add_argument('--device', type = str, default = 'cuda', help = 'Training [device]')
    parser.add_argument('--batch_size', type = int, default = 16, help = 'Define Batch Size')
    parser.add_argument('--img_size', type = int, default = 512, help = 'image size')
    parser.add_argument('--notify', type = bool, default = False, help = "Send notification to telegram")
    opt = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((opt.img_size, opt.img_size)),
        transforms.ToTensor()
    ])

    num_workers = os.cpu_count()
    val_dataset = ObjectRemovalDataset(data_dir=f"{DATA_DIR}", data_type="val", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=num_workers)
    # print(len(val_loader))
    for chkpt_no in range(opt.chkpt_no, opt.chkpt_no + opt.num_chkpts, 2):
        validate(chkpt_no, val_loader, opt.device)
        
    if opt.notify:
        send_notification(0, 0, 0, False, False, True, False)