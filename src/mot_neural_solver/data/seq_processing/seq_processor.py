"""
This file contains MOTSeqProcessor, which does all the necessary work to prepare tracking data (that is, detections,
imgs for every frame, sequence metainfo, and possibly ground truth files) for training and evaluation.

MOT Sequences from different datasets (e.g. MOT15 and MOT17) might have different storage structure, this is why we
define different 'sequence types', and map different sequences to them in _SEQ_TYPES.

For each type in _SEQ_TYPES, we define a different function to load a pd.DataFrame with their detections, a dictionary
with sequence metainfo (frames per second, img resolution, static/moving camera, etc.), and another pd.DataFrame with
ground truth boxes information. See e.g. MOT17loader.py as an example.

Once these three objects have been loaded, the rest of the sequence processing (e.g. matching ground truth boxes to
detections, storing embeddings, etc.) is performed in common, by the methods in MOTSeqProcessor

If you want to add new/custom sequences:
    1) Store with the same structure as e.g. MOT challenge mot_seqs (one directory per sequence):
    2) Add its sequences' names (dir names) and sequence type to the corresponding/new 'seq_type' in _SEQ_TYPES
    3) Modify / write a det_df loader function for the new 'seq_type' (see MOT17loader.py as an example)
    If you had to write a new loader function:
        4) Add the new (seq_type, det_df loader function) to SEQ_TYPE_DETS_DF_LOADER
    Make sure that 'fps' and other metadata is available in the scene_info_dict returned by your loader
"""
import pandas as pd
import numpy as np

from lapsolver import solve_dense

from mot_neural_solver.data.seq_processing.KITTIloader import get_kittimots_det_df, get_kittimots_det_df_from_gt
from mot_neural_solver.data.seq_processing.MOTS20loader import get_mots20_det_df, get_mots20_det_df_from_gt
from mot_neural_solver.data.seq_processing.MOT17loader import get_mot17_det_df, get_mot17_det_df_from_gt
from mot_neural_solver.data.seq_processing.MOT15loader import get_mot15_det_df, get_mot15_det_df_from_gt
from mot_neural_solver.utils.iou import iou, iou_mask
from mot_neural_solver.utils.rgb import BoundingBoxDataset, NodeEmbeddingDataset

import os
import os.path as osp

import shutil

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import MultiScaleRoIAlign, RoIAlign, roi_align
from tracktor_masked.maskrcnn_fpn import MaskRCNN_FPN
from torchvision.models.utils import load_state_dict_from_url
from tqdm import tqdm

import matplotlib.pyplot as plt
from mot_neural_solver.models.resnet import resnet50_fc256, load_pretrained_weights
from mot_neural_solver.path_cfg import OUTPUT_PATH

import pycocotools.mask as rletools

##########################################################
# Definition of available Sequences
##########################################################

# We define 'sequence types' for different MOT sequences, depending on the kind of processing they require (e.g. file
# storage structure, etc.). Each different type requires a different loader function that returns a pandas DataFrame
# with the right format from its detection file (see e.g. MOT17loader.py).

# Assign a loader func to each Sequence Type
_SEQ_TYPE_DETS_DF_LOADER = {'MOT17': get_mot17_det_df,
                            'MOT17_gt': get_mot17_det_df_from_gt,
                            'MOT15': get_mot15_det_df,
                            'MOT15_gt': get_mot15_det_df_from_gt,
                            'MOTS20': get_mots20_det_df,
                            'MOTS20_gt': get_mots20_det_df_from_gt,
                            'KITTIMOTS': get_kittimots_det_df,
                            'KITTIMOTS_gt': get_kittimots_det_df_from_gt}

# Determines whether boxes are allowed to have some area outside the image (all GT annotations in MOT15 are inside img
# hence we crop its detections to also be inside it)
_ENSURE_BOX_IN_FRAME = {'MOTS20': False,
                        'MOTS20_gt': False,
                        'KITTIMOTS': False,
                        'KITTIMOTS_gt': False,
                        'MOT17': False,
                        'MOT17_gt': False,
                        'MOT15': True,
                        'MOT15_gt': False}


# We now map each sequence name to a sequence type in _SEQ_TYPES
_SEQ_TYPES = {}

# MOTS20 Sequences
mots20_seqs = [f'MOTS20-{seq_num:02}' for seq_num in (2, 5, 9, 11, 22, 25, 29, 31, 63, 66, 67, 69)]
mots20_seqs += [f'MOTS20-{seq_num:02}' for seq_num in (1, 6, 7, 12)]
mots20_seqs += [f'MOTS20-{seq_num:02}-GT' for seq_num in (2, 5, 9, 11, 22, 25, 29, 31, 63, 66, 67, 69)]
for seq_name in mots20_seqs:
    if 'GT' in seq_name:
        _SEQ_TYPES[seq_name] = 'MOTS20_gt'

    else:
        _SEQ_TYPES[seq_name] = 'MOTS20'

# KITTI Sequences
kitti_seqs = [f'KITTIMOTS-{seq_num:02}' for seq_num in range(0, 29)]
for seq_name in kitti_seqs:
    if 'GT' in seq_name:
        _SEQ_TYPES[seq_name] = 'KITTIMOTS_gt'

    else:
        _SEQ_TYPES[seq_name] = 'KITTIMOTS'

# MOT17 Sequences
mot17_seqs = [f'MOT17-{seq_num:02}-{det}' for seq_num in (2, 4, 5, 9, 10, 11, 13) for det in ('DPM', 'SDP', 'FRCNN', 'GT')]
mot17_seqs += [f'MOT17-{seq_num:02}-{det}' for seq_num in (1, 3, 6, 7, 8, 12, 14) for det in ('DPM', 'SDP', 'FRCNN')]
for seq_name in mot17_seqs:
    if 'GT' in seq_name:
        _SEQ_TYPES[seq_name] = 'MOT17_gt'

    else:
        _SEQ_TYPES[seq_name] = 'MOT17'

# MOT15 Sequences 
mot15_seqs = ['KITTI-17', 'KITTI-13', 'ETH-Sunnyday', 'ETH-Bahnhof', 'PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte']
mot15_seqs += ['ADL-Rundle-6', 'ADL-Rundle-8', 'Venice-2', 'ETH-Pedcross2']
mot15_seqs += [seq_name + '-GT' for seq_name in mot15_seqs]
mot15_seqs += ['Venice-1', 'KITTI-16', 'KITTI-19', 'ADL-Rundle-1', 'ADL-Rundle-3', 'AVG-TownCentre']
mot15_seqs += ['ETH-Crossing', 'ETH-Linthescher', 'ETH-Jelmoli', 'PETS09-S2L2', 'TUD-Crossing']
for seq_name in mot15_seqs:
    if 'GT' in seq_name:
        _SEQ_TYPES[seq_name] = 'MOT15_gt'

    else:
        _SEQ_TYPES[seq_name] = 'MOT15'

##########################################################################################
# Classes used to store and process detections for every sequence
##########################################################################################

class DataFrameWSeqInfo(pd.DataFrame):
    """
    Class used to store each sequences's processed detections as a DataFrame. We just add a metadata atribute to
    pandas DataFrames it so that sequence metainfo such as fps, etc. can be stored in the attribute 'seq_info_dict'.
    This attribute survives serialization.
    This solution was adopted from:
    https://pandas.pydata.org/pandas-docs/stable/development/extending.html#define-original-properties
    """
    _metadata = ['seq_info_dict']

    @property
    def _constructor(self):
        return DataFrameWSeqInfo

class MOTSeqProcessor:
    """
    Class to process detections files coming from different mot_seqs.
    Main method is process_detections. It does the following:
    - Loads a DataFrameWSeqInfo (~pd.DataFrame) from a  detections file (self.det_df) via a the 'det_df_loader' func
    corresponding to the sequence type (mapped via _SEQ_TYPES)
    - Adds Sequence Info to the df (fps, img size, moving/static camera, etc.) as an additional attribute (_get_det_df)
    - If available, assigns ground truth identities to detection boxes via bipartite matching (_assign_gt)
    - Stores the df in disk (_store_det_df)
    - If required, precomputes CNN embeddings for every detected box and stores them to disk (_store_embeddings)

    The stored information assumes that each MOT sequence has its own directory. Inside it all processed data is
    stored as follows:
        +-- <Sequence name>
        |   +-- processed_data
        |       +-- det
        |           +-- <dataset_params['det_file_name']>.pkl # pd.DataFrame with processed detections and metainfo
        |       +-- embeddings
        |           +-- <dataset_params['det_file_name']> # Precomputed embeddings for a set of detections
        |               +-- <CNN Name >
        |                   +-- {frame1}.jpg
        |                   ...
        |                   +-- {frameN}.jpg
    """
    def __init__(self, dataset_path, seq_name, dataset_params, logger = None):
        self.seq_name = seq_name
        self.dataset_path = dataset_path
        self.seq_type = _SEQ_TYPES[seq_name]

        self.det_df_loader = _SEQ_TYPE_DETS_DF_LOADER[self.seq_type]
        self.dataset_params = dataset_params

        self.logger = logger

    def _ensure_boxes_in_frame(self):
        """
        Determines whether boxes are allowed to have some area outside the image (all GT annotations in MOT15 are inside
        the frame hence we crop its detections to also be inside it)
        """
        initial_bb_top = self.det_df['bb_top'].values.copy()
        initial_bb_left = self.det_df['bb_left'].values.copy()

        self.det_df['bb_top'] = np.maximum(self.det_df['bb_top'].values, 0).astype(int)
        self.det_df['bb_left'] = np.maximum(self.det_df['bb_left'].values, 0).astype(int)

        bb_top_diff = self.det_df['bb_top'].values - initial_bb_top
        bb_left_diff = self.det_df['bb_left'].values - initial_bb_left

        self.det_df['bb_height'] -= bb_top_diff
        self.det_df['bb_width'] -= bb_left_diff

        img_height, img_width = self.det_df.seq_info_dict['frame_height'], self.det_df.seq_info_dict['frame_width']
        self.det_df['bb_height'] = np.minimum(img_height - self.det_df['bb_top'], self.det_df['bb_height']).astype(int)
        self.det_df['bb_width'] = np.minimum(img_width - self.det_df['bb_left'], self.det_df['bb_width']).astype(int)

    def _assign_gt(self):
        """
        Assigns a GT identity to every detection in self.det_df, based on the ground truth boxes in self.gt_df.
        The assignment is done frame by frame via bipartite matching.
        """
        if self.det_df.seq_info_dict['has_gt'] and not self.det_df.seq_info_dict['is_gt']:
            print(f"Assigning ground truth identities to detections to sequence {self.seq_name}")
            for frame in self.det_df['frame'].unique():
                frame_detects = self.det_df[self.det_df.frame == frame]
                frame_gt = self.gt_df[self.gt_df.frame == frame]

                if self.dataset_params['mask_priority']:
                    # Compute IoU for each pair of detected / GT mask
                    iou_matrix = iou_mask(frame_detects['coco_mask'].values,
                                          frame_gt['coco_mask'].values)
                else:
                    # Compute IoU for each pair of detected / GT bounding box
                    iou_matrix = iou(frame_detects[['bb_top', 'bb_left', 'bb_bot', 'bb_right']].values,
                                     frame_gt[['bb_top', 'bb_left', 'bb_bot', 'bb_right']].values)

                iou_matrix[iou_matrix < self.dataset_params['gt_assign_min_iou']] = np.nan
                dist_matrix = 1 - iou_matrix
                assigned_detect_ixs, assigned_detect_ixs_ped_ids = solve_dense(dist_matrix)
                unassigned_detect_ixs = np.array(list(set(range(frame_detects.shape[0])) - set(assigned_detect_ixs)))

                assigned_detect_ixs_index = frame_detects.iloc[assigned_detect_ixs].index
                assigned_detect_ixs_masks = frame_gt.iloc[assigned_detect_ixs_ped_ids]['coco_mask'].values
                assigned_detect_ixs_ped_ids = frame_gt.iloc[assigned_detect_ixs_ped_ids]['id'].values
                unassigned_detect_ixs_index = frame_detects.iloc[unassigned_detect_ixs].index

                self.det_df.loc[assigned_detect_ixs_index, 'id'] = assigned_detect_ixs_ped_ids
                self.det_df.loc[unassigned_detect_ixs_index, 'id'] = -1  # False Positives

                self.det_df.loc[assigned_detect_ixs_index, 'gt_rle'] = assigned_detect_ixs_masks
                self.det_df.loc[unassigned_detect_ixs_index, 'gt_rle'] = None  # False Positives

    def _get_det_df(self):
        """
        Loads a pd.DataFrame where each row contains a detections bounding box' coordinates information (self.det_df),
        and, if available, a similarly structured pd.DataFrame with ground truth boxes.
        It also adds seq_info_dict as an attribute to self.det_df, containing sequence metainformation (img size,
        fps, whether it has ground truth annotations, etc.)
        """
        self.det_df, seq_info_dict, self.gt_df = self.det_df_loader(self.seq_name, self.dataset_path, self.dataset_params)

        self.det_df = DataFrameWSeqInfo(self.det_df)
        self.det_df.seq_info_dict = seq_info_dict

        # Some further processing
        if self.seq_type in _ENSURE_BOX_IN_FRAME and _ENSURE_BOX_IN_FRAME[self.seq_type]:
            self._ensure_boxes_in_frame()

        # Add some additional box measurements that might be used for graph construction
        # self.det_df['bb_bot'] = (self.det_df['bb_top'] + self.det_df['bb_height']).values
        # self.det_df['bb_right'] = (self.det_df['bb_left'] + self.det_df['bb_width']).values
        self.det_df['feet_x'] = self.det_df['bb_left'] + 0.5 * self.det_df['bb_width']
        self.det_df['feet_y'] = self.det_df['bb_top'] + self.det_df['bb_height']

        # Just a sanity check. Sometimes there are boxes that lay completely outside the frame
        frame_height, frame_width = self.det_df.seq_info_dict['frame_height'], self.det_df.seq_info_dict['frame_width']
        conds = (self.det_df['bb_width'] > 0) & (self.det_df['bb_height'] > 0)
        conds = conds & (self.det_df['bb_right'] > 0) & (self.det_df['bb_bot'] > 0)
        conds  =  conds & (self.det_df['bb_left'] < frame_width) & (self.det_df['bb_top'] < frame_height)
        self.det_df = self.det_df[conds].copy()

        self.det_df.sort_values(by = 'frame', inplace = True)
        self.det_df['detection_id'] = np.arange(self.det_df.shape[0]) # This id is used for future tastks

        return self.det_df

    def _store_df(self):
        """
        Stores processed detections DataFrame in disk.
        """
        processed_dets_path = osp.join(self.det_df.seq_info_dict['seq_path'], 'processed_data', 'det')
        os.makedirs(processed_dets_path, exist_ok = True)
        det_df_path = osp.join(processed_dets_path, self.det_df.seq_info_dict['det_file_name'] + '.pkl')
        self.det_df.to_pickle(det_df_path)

    def _store_gt_df(self):
        # Save also the ground truth data frame
        processed_gt_path = osp.join(self.det_df.seq_info_dict['seq_path'], 'processed_data', 'gt')
        os.makedirs(processed_gt_path, exist_ok=True)
        gt_df_path = osp.join(processed_gt_path, 'gt_df' + '.pkl')
        self.gt_df.to_pickle(gt_df_path)

    def _store_gt_masks(self):
        """
        Converts ground truth masks into prediction format (Convert to binary mask + RoI Align) and stores masks
        and valid idxs (detections with a gt match)
        """
        # Mask directories
        gt_mask_path = osp.join(self.det_df.seq_info_dict['seq_path'], 'processed_data/gt', 'gt_mask', 'masks')

        if osp.exists(gt_mask_path):
            print("Found existing stored gt masks. Deleting them and replacing them for new ones")
            shutil.rmtree(gt_mask_path)

        os.makedirs(gt_mask_path)

        # Valid ixs directories
        gt_valid_ixs_path = osp.join(self.det_df.seq_info_dict['seq_path'], 'processed_data/gt', 'gt_mask', 'valid_ixs')
        if osp.exists(gt_valid_ixs_path):
            print("Found existing stored valid_ixs. Deleting them and replacing them for new ones")
            shutil.rmtree(gt_valid_ixs_path)

        os.makedirs(gt_valid_ixs_path)

        gt_masks_all, valid_ixs_all = [], []
        frame_nums_all, det_ids_all = [], []

        print("Processing the ground truth masks")

        # Loop over the dataframe
        for frame in tqdm(self.det_df['frame'].unique()):
            frame_detects = self.det_df[self.det_df.frame == frame]

            # Get the masks and the corresponding bounding boxes
            gt_masks = frame_detects['gt_rle'].values
            bboxes = frame_detects[['bb_left', 'bb_top', 'bb_right', 'bb_bot']].to_numpy()
            ids = frame_detects['id'].values

            # Get frame numbers and detection ids
            frame_nums = frame_detects['frame'].values
            det_ids = frame_detects['detection_id'].values

            gt_available_ixs = np.logical_and(ids != -1, ids != 10000)  # Find the entries for which a valid gt mask is available

            # Get valid masks and boxes
            gt_masks = gt_masks[gt_available_ixs]
            bboxes = bboxes[gt_available_ixs]

            gt_masks = gt_masks.tolist()
            gt_available_ixs = torch.from_numpy(gt_available_ixs)

            # Create a zero tensor with N = # masks including invalid ones
            final_gt_masks = torch.zeros((gt_available_ixs.shape[0], 1, self.dataset_params['gt_mask_spatial_size'][0],
                                          self.dataset_params['gt_mask_spatial_size'][1]))

            if gt_masks:
                # Decode the masks
                gt_masks = rletools.decode(gt_masks)
                gt_masks = np.transpose(gt_masks, (2, 0, 1))  # Reshape into N, C, H, W
                gt_masks = torch.from_numpy(gt_masks).unsqueeze(dim=1).float()
                bboxes = torch.from_numpy(bboxes).float()

                # Convert boxes into roi format
                bboxes = torch.cat((torch.arange(gt_masks.shape[0], dtype=torch.float32).view(-1, 1),
                                    bboxes), dim=1)

                # Apply roi align
                gt_masks = roi_align(gt_masks, bboxes, (self.dataset_params['gt_mask_spatial_size'][0],
                                                        self.dataset_params['gt_mask_spatial_size'][1]), 1.)

                final_gt_masks[gt_available_ixs] = gt_masks

            # Append the values
            gt_masks_all.append(final_gt_masks.cpu())
            valid_ixs_all.append(gt_available_ixs)
            frame_nums_all.append(torch.from_numpy(frame_nums))
            det_ids_all.append(torch.from_numpy(det_ids))

        # Reshape
        gt_masks_all = torch.cat(gt_masks_all, dim=0)
        valid_ixs_all = torch.cat(valid_ixs_all, dim=0)
        det_ids_all = torch.cat(det_ids_all, dim=0)
        frame_nums_all = torch.cat(frame_nums_all, dim=0)

        # Add detection ids as first column to ensure that embeddings are loaded correctly
        gt_masks_all = torch.cat((det_ids_all.view(-1, 1, 1, 1).
                                  float().expand(-1, -1, self.dataset_params['gt_mask_spatial_size'][0],
                                                 self.dataset_params['gt_mask_spatial_size'][1]), gt_masks_all), dim=1)

        valid_ixs_all = torch.cat((det_ids_all.float().view(-1, 1), valid_ixs_all.float().view(-1, 1)), dim=1)

        print('Saving the processed ground truth masks')

        # Store the gt masks and valid ixs
        for frame in self.det_df.frame.unique():
            mask = frame_nums_all == frame

            frame_gt_masks = gt_masks_all[mask]
            frame_valid_ixs = valid_ixs_all[mask]

            frame_gt_masks_path = osp.join(gt_mask_path, f"{frame}.pt")
            frame_valid_ixs_path = osp.join(gt_valid_ixs_path, f"{frame}.pt")

            torch.save(frame_gt_masks, frame_gt_masks_path)
            torch.save(frame_valid_ixs, frame_valid_ixs_path)

        print('Ground truth masks are saved')

    def _store_embeddings(self):
        """
        Stores node and reid embeddings corresponding for each detection in the given sequence.
        Embeddings are stored at:
        {seq_info_dict['seq_path']}/processed_data/embeddings/{seq_info_dict['det_file_name']}/dataset_params['node/reid_embeddings_dir'}/FRAME_NUM.pt
        Essentially, each set of processed detections (e.g. raw, prepr w. frcnn, prepr w. tracktor) has a storage path, corresponding
        to a detection file (det_file_name). Within this path, different CNNs, have different directories
        (specified in dataset_params['node_embeddings_dir'] and dataset_params['reid_embeddings_dir']), and within each
        directory, we store pytorch tensors corresponding to the embeddings in a given frame, with shape
        (N, EMBEDDING_SIZE), where N is the number of detections in the frame.
        """
        from time import time
        assert self.dataset_params['reid_embeddings_dir'] is not None and \
               self.dataset_params['node_core_embeddings_dir'] is not None and \
               self.dataset_params['node_ext_embeddings_dir'] is not None

        bbox_dataset = BoundingBoxDataset(self.det_df, seq_info_dict=self.det_df.seq_info_dict, return_det_ids_and_frame = True)
        bbox_loader = DataLoader(bbox_dataset, batch_size=self.dataset_params['img_batch_size'], pin_memory=True,
                                 num_workers=4)

        # Feed all bboxes to the CNN to obtain node and reid embeddings
        t = time()
        print(f"Computing reid embeddings for {len(bbox_dataset)} detections")

        cnn_arch = self.dataset_params['cnn_params']['arch']
        cnn_model = resnet50_fc256(10, loss='xent', pretrained=True).cuda()
        load_pretrained_weights(cnn_model, osp.join(OUTPUT_PATH, self.dataset_params['cnn_params']['model_weights_path'][cnn_arch]))
        cnn_model.return_embeddings = True
        cnn_model.eval()
        reid_embeds, node_core_embeds = [], []
        frame_nums, det_ids = [], []

        with torch.no_grad():
            for frame_num, det_id, bboxes in tqdm(bbox_loader):
                node_core_out, reid_out = cnn_model(bboxes.cuda())
                node_core_embeds.append(node_core_out.cpu())
                reid_embeds.append(reid_out.cpu())
                frame_nums.append(frame_num)
                det_ids.append(det_id)
        #print("IT TOOK ", time() - t)
        print(f"Finished computing reid embeddings")

        det_ids = torch.cat(det_ids, dim=0)
        frame_nums = torch.cat(frame_nums, dim=0)

        node_core_embeds = torch.cat(node_core_embeds, dim=0)
        reid_embeds = torch.cat(reid_embeds, dim=0)

        # Add detection ids as first column of embeddings, to ensure that embeddings are loaded correctly
        #node_core_embeds = torch.cat((det_ids.view(-1, 1).float(), node_core_embeds), dim=1)
        node_core_embeds = torch.cat((det_ids.view(-1, 1, 1, 1).float().expand(-1, -1, 8, 4), node_core_embeds), dim=1)
        reid_embeds = torch.cat((det_ids.view(-1, 1).float(), reid_embeds), dim=1)

        reid_embeds_path = osp.join(self.det_df.seq_info_dict['seq_path'], 'processed_data/embeddings',
                                   self.det_df.seq_info_dict['det_file_name'], self.dataset_params['reid_embeddings_dir'])
        node_core_embeds_path = osp.join(self.det_df.seq_info_dict['seq_path'], 'processed_data/embeddings',
                                   self.det_df.seq_info_dict['det_file_name'], self.dataset_params['node_core_embeddings_dir'])

        if osp.exists(reid_embeds_path):
            print("Found existing stored reid embeddings. Deleting them and replacing them for new ones")
            shutil.rmtree(reid_embeds_path)

        if osp.exists(node_core_embeds_path):
            print("Found existing stored node core embeddings. Deleting them and replacing them for new ones")
            shutil.rmtree(node_core_embeds_path)

        os.makedirs(reid_embeds_path)
        os.makedirs(node_core_embeds_path)

        print("Storing the reid embeddings...")
        # Save embeddings grouped by frame
        for frame in tqdm(self.det_df.frame.unique()):
            mask = frame_nums == frame
            frame_node_core_embeds = node_core_embeds[mask]
            frame_reid_embeds = reid_embeds[mask]

            frame_node_core_embeds_path = osp.join(node_core_embeds_path, f"{frame}.pt")
            frame_reid_embeds_path = osp.join(reid_embeds_path, f"{frame}.pt")

            torch.save(frame_node_core_embeds, frame_node_core_embeds_path)
            torch.save(frame_reid_embeds, frame_reid_embeds_path)

        del reid_embeds, node_core_embeds
        print("Finished storing reid embeddings")

        ###########################################################################################
        # node_ext_embeddings and mask_embeddings
        ###########################################################################################

        node_embedding_dataset = NodeEmbeddingDataset(self.det_df)

        node_embedding_model = MaskRCNN_FPN(num_classes=91)
        state_dict = load_state_dict_from_url(self.dataset_params['node_embedding_model_url'])
        node_embedding_model.load_state_dict(state_dict)
        node_embedding_model.cuda()
        node_embedding_model.eval()

        # ROIAligns to map the features and the mask to the required size
        feature_roi_align = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                               output_size=self.dataset_params['embedding_spatial_size'],
                                               sampling_ratio=2)

        # Create dirs to store embeddings
        node_embeds_path = osp.join(self.det_df.seq_info_dict['seq_path'], 'processed_data/embeddings',
                                   self.det_df.seq_info_dict['det_file_name'], self.dataset_params['node_ext_embeddings_dir'])


        if osp.exists(node_embeds_path):
            print("Found existing stored node embeddings. Deleting them and replacing them for new ones")
            shutil.rmtree(node_embeds_path)

        os.makedirs(node_embeds_path)

        print(f"Computing node embeddings for {len(node_embedding_dataset)} frames")

        # Batch it to save cpu memory - otherwise it might cause SIGKILL
        frame_batch_size = 25
        for first_frame_ix in tqdm(range(0, len(node_embedding_dataset), frame_batch_size)):
            last_frame_ix = min(first_frame_ix+frame_batch_size, len(self.det_df['frame'].unique()))
            frame_ixs = list(range(first_frame_ix, last_frame_ix))
            subset = Subset(node_embedding_dataset, frame_ixs)
            node_embedding_data_loader = DataLoader(subset, batch_size=1, shuffle=False, pin_memory=True,
                                                    num_workers=4)

            node_embeds = []
            frame_nums, det_ids = [], []

            # Calculate node embeddings
            with torch.no_grad():
                for frame_num, frame_img, det_id, roi_boxes in node_embedding_data_loader:
                    # Fix the dimensions
                    frame_num = frame_num.squeeze(0)
                    det_id = det_id.squeeze(0)
                    roi_boxes = roi_boxes.squeeze(0)

                    # Get node and mask embeddings
                    node_embedding_model.load_image(frame_img)
                    frame_node_embeds = node_embedding_model.get_node_embeddings(roi_boxes, feature_roi_align)

                    # Save
                    node_embeds.append(frame_node_embeds.cpu())
                    frame_nums.append(frame_num)
                    det_ids.append(det_id)


            det_ids = torch.cat(det_ids, dim=0)
            frame_nums = torch.cat(frame_nums, dim=0)
            node_embeds = torch.cat(node_embeds, dim=0)

            # Add detection ids along dim=1 as a (M, M) tensor so that we can keep track of them
            node_embeds = torch.cat((det_ids.view(-1, 1, 1, 1).float().expand(-1, -1,
                                                                              self.dataset_params['embedding_spatial_size'],
                                                                              self.dataset_params['embedding_spatial_size'])
                                     , node_embeds), dim=1)
            # Save embeddings grouped by frame
            for f_ix in frame_ixs:
                frame = self.det_df['frame'].unique()[f_ix]
                mask = frame_nums == frame
                frame_node_embeds = node_embeds[mask]

                frame_node_embeds_path = osp.join(node_embeds_path, f"{frame}.pt")

                torch.save(frame_node_embeds, frame_node_embeds_path)
        print(f"Finished computing node embeddings")

        del node_embeds
        del node_embedding_model
        print("Finished storing node and mask embeddings")

    def process_detections(self):
        # See class header
        self._get_det_df()
        self._assign_gt()
        self._store_df()

        if self.det_df.seq_info_dict['has_gt']:
            self._store_gt_df()
            self._store_gt_masks()

        if self.dataset_params['precomputed_embeddings']:
            self._store_embeddings()

        return self.det_df

    def load_or_process_detections(self):
        """
        Tries to load a set of processed detections if it's safe to do so. otherwise, it processes them and stores the
        result
        """
        # Check if the processed detections file already exists.
        seq_path = osp.join(self.dataset_path, self.seq_name)
        det_file_to_use = self.dataset_params['det_file_name'] if not self.seq_name.endswith('GT') else 'gt'
        seq_det_df_path = osp.join(seq_path, 'processed_data/det', det_file_to_use + '.pkl')

        # If loading precomputed embeddings, check if embeddings have already been stored (otherwise, we need to process dets again)
        node_embeds_path = osp.join(seq_path, 'processed_data/embeddings', det_file_to_use, self.dataset_params['node_ext_embeddings_dir'])
        reid_embeds_path = osp.join(seq_path, 'processed_data/embeddings', det_file_to_use, self.dataset_params['reid_embeddings_dir'])
        try:
            num_frames = len(pd.read_pickle(seq_det_df_path)['frame'].unique())
            processed_dets_exist = True
        except:
            num_frames = -1
            processed_dets_exist = False

        embeds_ok = osp.exists(node_embeds_path) and len(os.listdir(node_embeds_path)) ==num_frames
        embeds_ok = embeds_ok and osp.exists(reid_embeds_path) and len(os.listdir(reid_embeds_path)) == num_frames
        embeds_ok = embeds_ok or not self.dataset_params['precomputed_embeddings']

        if processed_dets_exist and embeds_ok and not self.dataset_params['overwrite_processed_data']:
            print(f"Loading processed dets for sequence {self.seq_name} from {seq_det_df_path}")
            seq_det_df = pd.read_pickle(seq_det_df_path).reset_index().sort_values(by=['frame', 'detection_id'])

        else:
            print(f'Detections for sequence {self.seq_name} need to be processed. Starting processing')
            seq_det_df = self.process_detections()

        return seq_det_df
