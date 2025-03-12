from env_var import *
from src.utils.logger import *
import os
from zipfile import ZipFile
from validate import *
from src.utils.notify_me import *

from zipfile import ZipFile
def unzip_data(file_path, extract_path):
    with ZipFile(file_path, 'r') as zip:
        zip.extractall(extract_path)
        zip.close()

def remove_stale_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.startswith("._") or file.startswith("_"):
                os.remove(os.path.join(root, file))
                # print(f"Removed: {os.path.join(root, file)}")
import os
import zipfile

def zip_items(zip_filename, items):
    """Create a ZIP archive containing the specified files and directories."""
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for item in items:
            if os.path.isfile(item):
                # Add the file with its basename.
                zipf.write(item, arcname=os.path.basename(item))
            elif os.path.isdir(item):
                # Walk the directory and add files with a structure starting at the folder name.
                for root, dirs, files in os.walk(item):
                    for file in files:
                        filepath = os.path.join(root, file)
                        # Create an archive name that preserves the folder structure.
                        arcname = os.path.join(os.path.basename(item), os.path.relpath(filepath, item))
                        zipf.write(filepath, arcname)
            else:
                print(f'Warning: {item} does not exist or is not a valid file/folder.')

# if __name__ == '__main__':
#     file_list = ['project.zip']
#     run_async_tasks(file_list, message="Uploading ground truths and results...")
#     print("Notification Sent!!")

if __name__ == '__main__':
    # List the files and directories you want to include in the ZIP archive.
    items_to_zip = [
        'Checkpoints',
        'Logs',
        'Results',
        'src',
        '__init__.py',
        'env_var.py',
        'quick.py',
        'test.py',
        'train.py',
        'util.py',
        'validate.py',
        'commands.txt',
        'requirements.txt'
        'test_code.ipynb'
    ]
    
    zip_filename = 'project.zip'
    zip_items(zip_filename, items_to_zip)
    print(f'Successfully created {zip_filename}')


# # Run the function on your target directory
# if __name__ == "__main__":
#     # unzip_data("./data/RORD-small-dataset/gt.zip", "./data/train")
#     # unzip_data("./data/RORD-small-dataset/img.zip", "./data/train")
#     # unzip_data("./data/RORD-small-dataset/mask.zip", "./data/train")
#     # unzip_data("./data/RORD-small-dataset/val.zip", "./data")
    
#     # Run this on your dataset directories
#     # remove_stale_files("./data")

#     transform = transforms.Compose([
#         transforms.Resize((512, 512)),
#         transforms.ToTensor()
#     ])

#     num_workers = os.cpu_count()
#     val_dataset = ObjectRemovalDataset(data_dir=f"{DATA_DIR}", data_type="val", transform=transform)
#     val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=num_workers)
    
#     for chkpt_no in range(2, 31, 2):
#         # print(len(val_loader))
#         validate(chkpt_no, val_loader=val_loader)