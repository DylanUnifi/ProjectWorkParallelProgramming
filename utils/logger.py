import os

def init_logger(log_dir, fold):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"fold_{fold}.log")
    log_file = open(log_path, "a")
    return log_path, log_file

def write_log(log_file, text):
    log_file.write(text + "\n")
    log_file.flush()
