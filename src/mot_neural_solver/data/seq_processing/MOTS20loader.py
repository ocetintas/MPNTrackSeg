from mot_neural_solver.path_cfg import DATA_PATH
import os.path as osp
import os
import shutil
import pandas as pd
import configparser
import pycocotools.mask as rletools
import time
import numpy as np





MOV_CAMERA_DICT = { 'MOTS20-02': False,
                    'MOTS20-05': True,
                    'MOTS20-09': False,
                    'MOTS20-11': True,
                    'MOTS20-01': False,
                    'MOTS20-06': True,
                    'MOTS20-07': True,
                    'MOTS20-12': True}


# Class ids:
#   1: car (not considered in the evaluation) / Not sure about 1, they seem to be labeled as 10
#   2: pedestrian

DET_COL_NAMES = ('frame', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'label', 'img_height', 'img_width')
GT_COL_NAMES = ('frame', 'id', 'label', 'img_height', 'img_width', 'rle')

def _add_frame_path(det_df, seq_name, data_root_path, seq_info_dict):
    # Add each image's path from  MOTS20 data dir
    add_frame_path = lambda frame_num: osp.join(data_root_path, seq_name, f'img1/{frame_num:06}'+seq_info_dict['file_ext'])
    det_df['frame_path'] = det_df['frame'].apply(add_frame_path)

def _build_scene_info_dict_mots20(seq_name, data_root_path, dataset_params):
    info_file_path = osp.join(data_root_path, seq_name, 'seqinfo.ini')
    cp = configparser.ConfigParser()
    cp.read(info_file_path)

    seq_info_dict = {'seq': seq_name,
                     'seq_path': osp.join(data_root_path, seq_name),
                     'det_file_name': dataset_params['det_file_name'],

                     'frame_height': int(cp.get('Sequence', 'imHeight')),
                     'frame_width': int(cp.get('Sequence', 'imWidth')),

                     'seq_len': int(cp.get('Sequence', 'seqLength')),
                     'fps': int(cp.get('Sequence', 'frameRate')),
                     'mov_camera': MOV_CAMERA_DICT[seq_name],
                     'file_ext': str(cp.get('Sequence', 'imExt')),

                     'has_gt': osp.exists(osp.join(data_root_path, seq_name, 'gt'))}
    return seq_info_dict


def _make_cocotools_compatible(df):
    # Manipulate the df columns so that we can directly use cocotools on them
    df['coco_mask'] = [{'size': [h, w], 'counts': r.encode(encoding='UTF-8')} for h, w, r in
                       df[['img_height', 'img_width', 'rle']].values]
    return df


def _add_bbox_coords_to_gt_df(gt_df):
    # Initialize the columns
    gt_df['bb_left'] = -1
    gt_df['bb_top'] = -1
    gt_df['bb_width'] = -1
    gt_df['bb_height'] = -1

    # Iterate over frames to save memory
    for frame in gt_df['frame'].unique():
        frame_detects = gt_df[gt_df.frame == frame]

        rle_masks = frame_detects['coco_mask'].values.tolist()
        boxes = rletools.toBbox(rle_masks)  # x, y, w, h

        # Update the dataframe
        gt_df.loc[frame_detects.index, ['bb_left', 'bb_top', 'bb_width', 'bb_height']] = boxes

    gt_df['bb_bot'] = (gt_df['bb_top'] + gt_df['bb_height']).values
    gt_df['bb_right'] = (gt_df['bb_left'] + gt_df['bb_width']).values

    return gt_df

def get_mots20_det_df(seq_name, data_root_path, dataset_params):
    seq_path = osp.join(data_root_path, seq_name)
    detections_file_path = osp.join(seq_path, f"det/{dataset_params['det_file_name']}.txt")
    det_df = pd.read_csv(detections_file_path, header=None, sep=' ')

    if dataset_params['det_file_name'] == 'tracktor_prepr_det':
        DET_COL_NAMES = ('frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'label', 'img_height',
                         'img_width')
    else:
        DET_COL_NAMES = ('frame', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'label', 'img_height',
                         'img_width')

    # Number and order of columns is always assumed to be the same
    det_df = det_df[det_df.columns[:len(DET_COL_NAMES)]]
    det_df.columns = DET_COL_NAMES
    det_df = det_df[det_df['label'].isin([2])].copy()

    det_df = det_df[det_df['conf'].ge(dataset_params['confidence_threshold'])].copy()
    det_df['id'] = -1

    # det_df['bb_left'] -= 1 # Coordinates are 1 based
    # det_df['bb_top'] -= 1
    det_df['bb_bot'] = (det_df['bb_top'] + det_df['bb_height']).values
    det_df['bb_right'] = (det_df['bb_left'] + det_df['bb_width']).values

    # if len(det_df['id'].unique()) > 1:
    #     det_df['tracktor_id'] = det_df['id']

    det_df['gt_rle'] = ''    # Placeholder for gt masks

    seq_info_dict = _build_scene_info_dict_mots20(seq_name, data_root_path, dataset_params)
    seq_info_dict['is_gt'] = False
    _add_frame_path(det_df, seq_name, data_root_path, seq_info_dict)

    if seq_info_dict['has_gt']:  # Return the corresponding ground truth, if available, for the ground truth assignment
        gt_file_path = osp.join(seq_path, f"gt/gt.txt")
        gt_df = pd.read_csv(gt_file_path, header=None, sep=' ')
        gt_df = gt_df[gt_df.columns[:len(GT_COL_NAMES)]]
        gt_df.columns = GT_COL_NAMES
        gt_df = gt_df[gt_df['label'].isin([2])].copy()  # Only the pedestrians are considered in the MOTS challenge
        gt_df = _make_cocotools_compatible(gt_df)
        gt_df = _add_bbox_coords_to_gt_df(gt_df)


        # Store the gt file in the common evaluation path
        gt_to_eval_path = osp.join(DATA_PATH, 'MOTS_eval_gt', seq_name, 'gt')
        os.makedirs(gt_to_eval_path, exist_ok=True)
        shutil.copyfile(gt_file_path, osp.join(gt_to_eval_path, 'gt.txt'))

    else:
        gt_df = None

    return det_df, seq_info_dict, gt_df


def get_mots20_det_df_from_gt(seq_name, data_root_path, dataset_params):
    # Create a dir to store Ground truth data in case if does not exist yet
    seq_path = osp.join(data_root_path, seq_name)
    if not osp.exists(seq_path):
        os.mkdir(seq_path)
        non_gt_seq_path = osp.join(data_root_path, seq_name[:-3])
        shutil.copytree(osp.join(non_gt_seq_path, 'gt'), osp.join(seq_path, 'gt'))

    detections_file_path = osp.join(data_root_path, seq_name, f"gt/gt.txt")
    det_df = pd.read_csv(detections_file_path, header=None, sep=' ')

    # Number and order of columns is always assumed to be the same
    det_df = det_df[det_df.columns[:len(GT_COL_NAMES)]]
    det_df.columns = GT_COL_NAMES

    det_df = det_df[det_df['label'].isin([2])].copy()

    # Add bboxes
    det_df = _add_bbox_coords_to_gt_df(det_df)
    det_df['gt_rle'] = det_df['coco_mask'].values

    seq_info_dict = _build_scene_info_dict_mots20(seq_name[:-3], data_root_path, dataset_params)
    _add_frame_path(det_df, seq_name[:-3], data_root_path, seq_info_dict)

    # Correct the detections file name to contain the 'gt' as name
    seq_info_dict['seq'] = seq_name
    seq_info_dict['seq_path'] = osp.join(data_root_path, seq_name)
    seq_info_dict['det_file_name'] = 'gt'
    seq_info_dict['is_gt'] = True

    # Create gt df
    gt_file_path = osp.join(seq_path, f"gt/gt.txt")
    gt_df = pd.read_csv(gt_file_path, header=None, sep=' ')
    gt_df = gt_df[gt_df.columns[:len(GT_COL_NAMES)]]
    gt_df.columns = GT_COL_NAMES
    gt_df = gt_df[gt_df['label'].isin([2])].copy()  # Only the pedestrians are considered in the MOTS challenge
    gt_df = _make_cocotools_compatible(gt_df)
    gt_df = _add_bbox_coords_to_gt_df(gt_df)

    # Store the gt file in the common evaluation path
    gt_file_path = osp.join(seq_path, f"gt/gt.txt")
    gt_to_eval_path = osp.join(DATA_PATH, 'MOTS_eval_gt', seq_name, 'gt')
    os.makedirs(gt_to_eval_path, exist_ok=True)
    shutil.copyfile(gt_file_path, osp.join(gt_to_eval_path, 'gt.txt'))

    return det_df, seq_info_dict, gt_df

def MOTS20Wrapper(dataset_name):
    train_sequences = ['MOTS20-02', 'MOTS20-05', 'MOTS20-09', 'MOTS20-11']
    test_sequences = ['MOTS20-01', 'MOTS20-06', 'MOTS20-07', 'MOTS20-12']

    if dataset_name == 'mots20_train':
        return train_sequences, 'train'
    elif dataset_name == 'mots20_test':
        return test_sequences, 'test'
