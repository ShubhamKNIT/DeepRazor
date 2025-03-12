import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)

import asyncio
import telegram
import argparse
from env_var import *

# Load environment variables from ~/.bashrc
bashrc_path = os.path.expanduser("~/.zshrc")
if os.path.exists(bashrc_path):
    with open(bashrc_path) as f:
        for line in f:
            if line.startswith("export "):  # Extract variables from `export VAR=VALUE`
                key, value = line.replace("export ", "").strip().split("=", 1)
                os.environ[key] = value

# Fetch environment variables
TELEBOT = os.getenv("TELEBOT")
CHAT_ID = os.getenv("CHAT_ID")

if not TELEBOT or not CHAT_ID:
    raise ValueError("TELEBOT or CHAT_ID environment variables are not set. Ensure ~/.bashrc contains them.")

# Initialize the bot
bot = telegram.Bot(token=TELEBOT)

# Define an async function to send a message
async def send_hello_messages(message):
    await bot.send_message(chat_id=CHAT_ID, text=message)
    await asyncio.sleep(1)

# Function to send a single document (image, etc.)
async def send_document(file_path):
    try:
        with open(file_path, 'rb') as doc:
            await bot.send_document(chat_id=CHAT_ID, document=doc)
    except Exception as e:
        print(f"Failed to send {file_path}: {e}")

# Function to send multiple files
async def send_files(file_list, message=None):
    if message:
        await send_hello_messages(message)
    for file_path in file_list:
        await send_document(file_path)
        await asyncio.sleep(1)  # Short delay between file sends

# Function to run async tasks safely
def run_async_tasks(file_list, message="Uploading files..."):
    asyncio.run(send_files(file_list, message))

# Function to send notifications with result images and logs
def send_notification(chkpt_no, num_chkpts, num_samples, send_imgs, send_train_logs, send_val_logs, send_chart):
    file_list = []
    
    if send_imgs:
        file_list.extend([
            f"{VAL_IMG_DIR}/epoch_{chkpt}_result_{idx}.png" 
            for chkpt in range(chkpt_no, chkpt_no + num_chkpts)
            for idx in range(num_samples) 
        ])
    if send_train_logs:
        file_list.append(TRAIN_CSV_PATH)
    if send_val_logs:
        file_list.append(VAL_CSV_PATH)
    if send_chart:
        file_list.append(f'{VAL_IMG_DIR}/comparison_plot.png')

    run_async_tasks(file_list, message="Uploading ground truths and results...")
    print("Notification Sent!!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--chkpt_no', type = int, default = -1, help = 'Checkpoint number for notification')
    parser.add_argument('--num_chkpts', type = int, default = 1, help = 'Enter number of chkpt')
    parser.add_argument('--num_samples', type = int, default = 2, help = 'Number of samples to send')
    parser.add_argument('--send_imgs', type = bool, default = True, help = 'Send images option')
    parser.add_argument('--send_train_logs', type = bool, default = True, help = 'Send Train Logs option')
    parser.add_argument('--send_val_logs', type = bool, default = True, help = 'Send Val Logs option')
    parser.add_argument('--send_chart', type = bool, default = True, help = 'Send Comparision Plot option')
    opt = parser.parse_args()
    
    send_notification(opt.chkpt_no, opt.num_chkpts, opt.num_samples, opt.send_imgs, opt.send_train_logs, opt.send_val_logs, opt.send_chart)