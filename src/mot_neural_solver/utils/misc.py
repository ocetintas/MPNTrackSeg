import os
import os.path as osp
import pickle
import random
from datetime import datetime
import pandas as pd

import numpy as np
import torch
#from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import Callback

from mot_neural_solver.path_cfg import OUTPUT_PATH, DATA_PATH


def make_deterministic(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    # random.seed(seed)
    # np.random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    #
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False

def _init_fn(worker_id):
    """
    Make dataloaders deterministic
    """
    np.random.seed(12 + worker_id)

def load_pickle(path):
    with open(path, 'rb') as file:
        ob = pickle.load(file)
    return ob

def save_pickle(ob, path):
    with open(path, 'wb') as file:
        pickle.dump(ob, file, protocol=pickle.HIGHEST_PROTOCOL)


def get_run_str(run_id, cross_val_split, add_date):
    if cross_val_split is None:
        run_str = run_id
    else:
        run_str = run_id + f"_split_{cross_val_split}"

    if add_date:
        date = '{date:%m-%d_%H:%M:%S.%f}'.format(date=datetime.now())
        run_str = date + '_' + run_str

    return run_str


def get_run_str_and_save_dir(run_id, cross_val_split, add_date):
    run_str = get_run_str(run_id, cross_val_split,
                          add_date=add_date)
    unique_id_assert = f"Run ID string {run_str} already exists, try setting add_date_to_run_str=True"
    save_dir = osp.join(OUTPUT_PATH, 'experiments', run_str)

    assert not osp.exists(save_dir), unique_id_assert

    return run_str, save_dir


def load_gt_df(seq_name, dataset_name='MOTS20', split='train'):
    seq_gt_path = osp.join(DATA_PATH, dataset_name, split, seq_name, 'processed_data', 'gt', 'gt_df.pkl')
    seq_gt_df = pd.read_pickle(seq_gt_path).sort_values(by=['frame', 'id'])
    return seq_gt_df


class ModelCheckpoint(Callback):
    """
    Callback to allow saving models on every epoch, even if there's no validation loop
    """
    def __init__(self, save_epoch_start = 0, save_every_epoch=False):
        super(ModelCheckpoint, self).__init__()
        self.save_every_epoch = save_every_epoch
        self.save_epoch_start = save_epoch_start

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch + 1 >= self.save_epoch_start and self.save_every_epoch:
            filepath = osp.join(trainer.default_save_path,'checkpoints', f"epoch_{trainer.current_epoch+1:03}.ckpt")
            os.makedirs(osp.dirname(filepath), exist_ok = True)
            trainer.save_checkpoint(filepath)
            print(f"Saving model at {filepath}")