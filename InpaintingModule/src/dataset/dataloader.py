import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class ObjectRemovalDataset(Dataset):
    def __init__(self, data_dir, data_type="train", transform=None, mask_threshold=1.0):
        self.data_type = data_type
        self.transform = transform
        self.mask_threshold = mask_threshold

        data_dir = os.path.abspath(os.path.expanduser(data_dir))
        self.img_dir = os.path.join(data_dir, self.data_type, "img")
        self.mask_dir = os.path.join(data_dir, self.data_type, "mask")
        self.gt_dir = os.path.join(data_dir, self.data_type, "gt")

        self.folders_li = self.filter_files(os.listdir(self.img_dir))
        self.files_li = self.get_basename(self.img_dir)
        
        self.img_files = [f"{file}.jpg" for file in self.files_li]
        self.mask_files = [f"{file}_M.png" for file in self.files_li]
        self.gt_files = [f"{file}.jpg" for file in self.files_li]

        # Filter out images where the mask covers more than 50%
        self.img_files, self.mask_files, self.gt_files = self.filter_valid_images()

    def __len__(self):
        return len(self.img_files)

    def filter_files(self, files_li):
        return [f for f in files_li if not f.startswith("._")]

    def get_basename(self, dir_path):
        files_li = []
        for folder in self.folders_li:
            files = self.filter_files(os.listdir(os.path.join(dir_path, folder)))
            for _file in files:
                file_path = f"{folder}/{_file}".split('.')[0]
                files_li.append(file_path)
        return files_li

    def is_valid_mask(self, mask):
        mask_tensor = transforms.ToTensor()(mask)  # Convert mask to tensor (range 0-1)
        mask_ratio = mask_tensor.sum() / mask_tensor.numel()
        return mask_ratio < self.mask_threshold

    def filter_valid_images(self):
        valid_img_files, valid_mask_files, valid_gt_files = [], [], []
        for img_file, mask_file, gt_file in zip(self.img_files, self.mask_files, self.gt_files):
            mask_path = os.path.join(self.mask_dir, mask_file)
            mask = Image.open(mask_path).convert("L")
            if self.is_valid_mask(mask):
                valid_img_files.append(img_file)
                valid_mask_files.append(mask_file)
                valid_gt_files.append(gt_file)
        return valid_img_files, valid_mask_files, valid_gt_files

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        gt_path = os.path.join(self.gt_dir, self.gt_files[idx])

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        gt = Image.open(gt_path).convert("RGB")

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
            gt = self.transform(gt)

        mask = 1 - mask  # Invert the mask: 1 represents hole
        
        return img, mask, gt
