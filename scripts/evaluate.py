import sacred
from sacred import Experiment

from mot_neural_solver.utils.misc import make_deterministic, get_run_str_and_save_dir

from mot_neural_solver.path_cfg import OUTPUT_PATH, DATA_PATH
import os.path as osp

from mot_neural_solver.pl_module.pl_module import MOTNeuralSolver
from mot_neural_solver.utils.evaluation import compute_mots_metrics
from shutil import copyfile
from TrackEval.scripts.run_kitti_mots import eval_kitti_mots

import pandas as pd
import time

from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG=False

ex = Experiment()
ex.add_config('configs/tracking_cfg.yaml')
ex.add_config({'ckpt_path': 'trained_models/mots/mots20.ckpt',
               'run_id': 'eval_test',
               'add_date': True,
               'precomputed_embeddings': True})

@ex.automain
def main(_config, _run):

    sacred.commands.print_config(_run)
    make_deterministic(12345)

    run_str, save_dir = get_run_str_and_save_dir(_config['run_id'], None, _config['add_date'])
    out_files_dir = osp.join(save_dir, 'mots_files')
    print("EXPERIMENT ID: ", run_str)

    # Load model from checkpoint and update config entries that may vary from the ones used in training
    model = MOTNeuralSolver.load_from_checkpoint(checkpoint_path=_config['ckpt_path'] if osp.exists(_config['ckpt_path'])  else osp.join(OUTPUT_PATH, _config['ckpt_path']))

    model.hparams.update({'eval_params':_config['eval_params'],
                          'data_splits':_config['data_splits']})
    model.hparams['dataset_params']['precomputed_embeddings'] = _config['precomputed_embeddings']

    # Get output MOT results files
    test_dataset = model.test_dataset()
    t_processing = time.time()
    constr_satisf_rate = model.track_all_seqs(dataset=test_dataset,
                                              output_files_dir = out_files_dir,
                                              use_gt = False,
                                              verbose=True)

    if all('KITTI' in s for s in test_dataset.seq_names):
        tracker_id = run_str
        # Copy the tracking output file with KITTI output name format
        for seq in test_dataset.seq_names:
            seq_num = int(seq[-2:])
            original_file = osp.join(out_files_dir, seq + '.txt')
            kitti_file = osp.join(out_files_dir, f'{seq_num:04}.txt')
            copyfile(original_file, kitti_file)
        try:
            eval_kitti_mots(tracker_id, split='training_valid')

        except:
            print('KITTI - results can not be evaluated')

    # If there's GT available (e.g. if testing on train sequences) try to compute MOT metrics
    try:
        mot_metrics_summary = compute_mots_metrics(gt_path=osp.join(DATA_PATH, 'MOTS20'),
                                                  out_mots_files_path=out_files_dir,
                                                  seqs=test_dataset.seq_names,)
        mot_metrics_summary['constr_sr'] = constr_satisf_rate

        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'expand_frame_repr', False):
            cols = [col for col in mot_metrics_summary.columns if col in _config['eval_params']['mot_metrics_to_log']]
            print("\n" + str(mot_metrics_summary[cols]))

    except:
        print("Could not evaluate the given results")
