import numpy as np
import torch
import random
import os
import datetime

def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_filename(model, desc, hparams):
        hparams_formatted = {k: f"{v}".replace('-', 'm').replace('.', 'p') for k, v in hparams.items()}
        current_datetime = datetime.datetime.now().strftime("%d%m%Y_%H%M")
        filename = (f"{model}_{desc}_"
            f"{'_'.join([f'{k}{v}' for k, v in hparams_formatted.items()])}"
            f"_{current_datetime}")
        return f"{filename}"

