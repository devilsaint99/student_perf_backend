import logging
import os
from datetime import datetime

LOG_FILE_DIR = f"{datetime.now().strftime('%d_%m_%Y')}"
logs_path=os.path.join(os.getcwd(), 'logs', LOG_FILE_DIR)
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path,datetime.now().strftime('%d_%m_%Y_%H_%M_%S'))

logging.basicConfig(
    filename=f'{LOG_FILE_PATH}.log',
    format="[%(asctime)s ]  %(lineno)s %(name)s - %(levelname)s - %(message)s]",
    level=logging.INFO
)
