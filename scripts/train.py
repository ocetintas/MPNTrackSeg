import sacred
from sacred import Experiment

from mot_neural_solver.utils.evaluation import MOTMetricsLogger
from mot_neural_solver.utils.misc import make_deterministic, get_run_str_and_save_dir, ModelCheckpoint

from mot_neural_solver.path_cfg import OUTPUT_PATH
import os.path as osp

from mot_neural_solver.pl_module.pl_module import MOTNeuralSolver

import pytorch_lightning
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
#from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG=False

ex = Experiment()
ex.add_config('configs/tracking_cfg.yaml')
ex.add_config({'run_id': 'train_w_default_config',
               'add_date': True,
               'cross_val_split': None})

@ex.config
def cfg(cross_val_split, eval_params, dataset_params, graph_model_params, data_splits):

    # Training requires the use of precomputed embeddings
    assert dataset_params['precomputed_embeddings'], "Training without precomp. embeddings is not supp"

    # Only use tracktor for postprocessing if tracktor was used for preprocessing
    if 'tracktor' not in dataset_params['det_file_name']:
        eval_params['add_tracktor_detects'] = False

    # Make sure that the edges encoder MLP input dim. matches the number of edge features used.
    graph_model_params['encoder_feats_dict']['edge_in_dim'] = len(dataset_params['edge_feats_to_use'])

    data_splits['train'] = ['mots20_train_split']
    data_splits['val'] = ['mots20_val_split']


@ex.automain
def main(_config, _run):

    pytorch_lightning.trainer.seed.seed_everything(_config['seed'])
    pytorch_lightning.seed_everything(_config['seed'])
    make_deterministic(_config['seed'])
    sacred.commands.print_config(_run)

    model = MOTNeuralSolver(hparams=dict(_config))

    run_str, save_dir = get_run_str_and_save_dir(_config['run_id'], _config['cross_val_split'], _config['add_date'])
    print("EXPERIMENT ID: ", run_str)

    if _config['train_params']['tensorboard']:
        logger = TensorBoardLogger(OUTPUT_PATH, name='experiments', version=run_str)

    else:
        logger = None

    ckpt_callback = ModelCheckpoint(save_epoch_start = _config['train_params']['save_epoch_start'],
                                    save_every_epoch = _config['train_params']['save_every_epoch'])

    trainer = Trainer(gpus=1,
                      callbacks=[MOTMetricsLogger(compute_oracle_results = _config['eval_params']['normalize_mot_metrics'],
                                                  compute_pred_oracles=_config['eval_params']['compute_pred_oracles']), ckpt_callback],
                      weights_summary = None,
                      checkpoint_callback=False,
                      max_epochs=_config['train_params']['num_epochs'],
                      val_percent_check = _config['eval_params']['val_percent_check'],
                      check_val_every_n_epoch=_config['eval_params']['check_val_every_n_epoch'],
                      nb_sanity_val_steps=0,
                      logger =logger,
                      default_save_path=osp.join(OUTPUT_PATH, 'experiments', run_str),
                      accumulate_grad_batches=_config['train_params']['accumulate_grad_batches'],
                      deterministic=True)
    trainer.fit(model)





