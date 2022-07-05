import numpy as np
import pandas as pd

import torch

from mot_neural_solver.data.mot_graph import Graph

from mot_neural_solver.tracker.projectors import GreedyProjector, ExactProjector
from mot_neural_solver.tracker.postprocessing import Postprocessor

from mot_neural_solver.utils.graph import get_knn_mask, to_undirected_graph, to_lightweight_graph
from mot_neural_solver.utils.misc import load_gt_df
import pycocotools.mask as rletools

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import time

from torchvision.models.detection.roi_heads import paste_masks_in_image
from mot_neural_solver.utils.mots import ensure_unique_masks
import matplotlib.pyplot as plt
from datetime import datetime


from skimage.io import imread
from torchvision.transforms import ToTensor

from tracktor_masked.maskrcnn_fpn import MaskRCNN_FPN
from torchvision.models.utils import load_state_dict_from_url

VIDEO_COLUMNS = ['frame_path', 'frame', 'ped_id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'bb_right', 'bb_bot',
                 'img_height', 'img_width', 'label', 'id', 'roi_masks']
TRACKING_OUT_COLS = ['frame', 'ped_id', 'label', 'img_height', 'img_width', 'ped_mask']



class MPNTracker:
    """
    Class used to track video sequences.

    See 'track'  method for an overview.
    """
    def __init__(self, dataset, graph_model, use_gt, eval_params = None,
                 dataset_params=None, logger=None):

        self.dataset = dataset
        self.use_gt = use_gt
        self.logger = logger

        self.eval_params = eval_params
        self.dataset_params = dataset_params

        self.graph_model = graph_model

        if self.graph_model is not None:
            self.graph_model.eval()


    def _estimate_frames_per_graph(self, seq_name):
        """
        Determines the number of frames to be included in each batch of frames evaluated within a sequence
        """
        num_frames = len(self.dataset.seq_det_dfs[seq_name].frame.unique())
        num_detects = self.dataset.seq_det_dfs[seq_name].shape[0]

        avg_detects_per_frame = num_detects / float(num_frames)
        expected_frames_per_graph = round(self.dataset.dataset_params['max_detects'] / avg_detects_per_frame)

        return min(expected_frames_per_graph, self.dataset.dataset_params['frames_per_graph'])

    def _load_full_seq_graph_object(self, seq_name):
        """
        Loads a MOTGraph (see data/mot_graph.py) object corresponding to the entire sequence.
        """
        step_size = self.dataset.seq_info_dicts[seq_name]['step_size']
        frames_per_graph = self._estimate_frames_per_graph(seq_name)
        start_frame = self.dataset.seq_det_dfs[seq_name].frame.min()
        end_frame = self.dataset.seq_det_dfs[seq_name].frame.max()

        if self.dataset.dataset_params['max_frame_dist'] == 'max':
            max_frame_dist = step_size * (frames_per_graph - 1)

        else:
            max_frame_dist = self.dataset.dataset_params['max_frame_dist']

        full_graph = self.dataset.get_from_frame_and_seq(seq_name=seq_name,
                                                         start_frame=start_frame,
                                                         end_frame=end_frame,
                                                         return_full_object=True,
                                                         ensure_end_is_in=True,
                                                         max_frame_dist = max_frame_dist,
                                                         inference_mode=True)
        full_graph.frames_per_graph = frames_per_graph
        return full_graph

    def _predict_edges_and_masks(self, subgraph, pred_oracle_mode=None):
        """
        Predicts edge and mask values for a subgraph (i.e. batch of frames) from the entire sequence.
        Args:
            subgraph: Graph Object corresponding to a subset of frames

        Returns:
            tuple containing a torch.Tensor with the predicted value for every edge in the subgraph, and a binary mask
            indicating which edges inside the subgraph where pruned with KNN
        """
        # Prune graph edges
        knn_mask = get_knn_mask(pwise_dist= subgraph.reid_emb_dists, edge_ixs = subgraph.edge_index,
                                num_nodes = subgraph.num_nodes, top_k_nns = self.dataset_params['top_k_nns'],
                                use_cuda = True, reciprocal_k_nns=self.dataset_params['reciprocal_k_nns'],
                                symmetric_edges=True)
        subgraph.edge_index = subgraph.edge_index.T[knn_mask].T
        subgraph.edge_attr = subgraph.edge_attr[knn_mask]
        if hasattr(subgraph, 'edge_labels'):
            subgraph.edge_labels = subgraph.edge_labels[knn_mask]

        # Predict active edges
        if self.use_gt: # For debugging purposes and obtaining oracle results
            pruned_edge_preds = subgraph.edge_labels
            node_preds = subgraph.mask_labels

        else:
            with torch.no_grad():
                output = self.graph_model(subgraph)
                if pred_oracle_mode=='gt_edge':
                    pruned_edge_preds = subgraph.edge_labels
                    node_preds = torch.sigmoid(output['mask_predictions'][-1])
                elif pred_oracle_mode=='gt_mask':
                    pruned_edge_preds = torch.sigmoid(output['classified_edges'][-1].view(-1))
                    node_preds = subgraph.mask_labels
                else:
                    pruned_edge_preds = torch.sigmoid(output['classified_edges'][-1].view(-1))
                    node_preds = torch.sigmoid(output['mask_predictions'][-1])

        edge_preds = torch.zeros(knn_mask.shape[0]).to(pruned_edge_preds.device)
        edge_preds[knn_mask] = pruned_edge_preds

        if self.eval_params['set_pruned_edges_to_inactive']:
            return edge_preds, torch.ones_like(knn_mask), node_preds

        else:
            return edge_preds, knn_mask, node_preds

    def _evaluate_graph_in_batches(self, pred_oracle_mode=None):
        """
        Feeds the entire sequence though the MPN in batches. It does so by applying a 'sliding window' over the sequence,
        where windows correspond consecutive pairs of start/end frame locations (e.g. frame 1 to 15, 5 to 20, 10 to 25,
        etc.).
        For every window, a subgraph is created by selecting all detections that fall inside it. Then this graph
        is fed to the message passing network, and predictions are stored.
        Since windows overlap, we end up with several predictions per edge. We simply average them overall all
        windows.
        """
        device = torch.device('cuda')
        all_frames = np.array(self.full_graph.frames)
        frame_num_per_node = torch.from_numpy(self.full_graph.graph_df.frame.values).to(device)
        node_names = torch.arange(self.full_graph.graph_obj.x.shape[0])

        # Iterate over overlapping windows of (starg_frame, end_frame)
        overall_edge_preds = torch.zeros(self.full_graph.graph_obj.num_edges).to(device)
        overall_num_preds = overall_edge_preds.clone()

        overall_node_preds = torch.zeros((self.full_graph.graph_obj.num_nodes, 1,
                                          self.dataset_params['gt_mask_spatial_size'][0],
                                          self.dataset_params['gt_mask_spatial_size'][1])).to(device)
        overall_num_node_preds = torch.zeros(self.full_graph.graph_obj.num_nodes).to(device)

        for eval_round, (start_frame, end_frame) in enumerate(zip(all_frames, all_frames[self.full_graph.frames_per_graph - 1:])):
            assert ((start_frame <= all_frames) & (all_frames <= end_frame)).sum() == self.full_graph.frames_per_graph

            # Create and evaluate a a subgraph corresponding to a batch of frames
            nodes_mask = (start_frame <= frame_num_per_node) & (frame_num_per_node <= end_frame)
            edges_mask = nodes_mask[self.full_graph.graph_obj.edge_index[0]] & nodes_mask[
                self.full_graph.graph_obj.edge_index[1]]

            subgraph = Graph(x=self.full_graph.graph_obj.x[nodes_mask],
                             x_ext=self.full_graph.graph_obj.x_ext[nodes_mask],
                             edge_attr=self.full_graph.graph_obj.edge_attr[edges_mask],
                             reid_emb_dists=self.full_graph.graph_obj.reid_emb_dists[edges_mask],
                             edge_index=self.full_graph.graph_obj.edge_index.T[edges_mask].T - node_names[nodes_mask][0])

            if hasattr(self.full_graph.graph_obj, 'edge_labels'):
                subgraph.edge_labels = self.full_graph.graph_obj.edge_labels[edges_mask]

            if hasattr(self.full_graph.graph_obj, 'mask_labels'):
                subgraph.mask_labels = self.full_graph.graph_obj.mask_labels[nodes_mask]

            if hasattr(self.full_graph.graph_obj, 'mask_gt_ixs'):
                subgraph.mask_gt_ixs = self.full_graph.graph_obj.mask_gt_ixs[nodes_mask]

            # Predict edge values for the current batch
            edge_preds, pred_mask, node_preds = self._predict_edges_and_masks(subgraph=subgraph,
                                                                              pred_oracle_mode=pred_oracle_mode)

            # Store predictions
            overall_edge_preds[edges_mask] += edge_preds
            assert (overall_num_preds[torch.where(edges_mask)[0][pred_mask]] == overall_num_preds[edges_mask][pred_mask]).all()
            overall_num_preds[torch.where(edges_mask)[0][pred_mask]] += 1

            overall_node_preds[nodes_mask] += node_preds
            overall_num_node_preds[nodes_mask] += 1

        # Average predictions over all batches
        final_edge_preds = overall_edge_preds / overall_num_preds
        final_edge_preds[torch.isnan(final_edge_preds)] = 0
        self.full_graph.graph_obj.edge_preds = final_edge_preds
        to_undirected_graph(self.full_graph, attrs_to_update=('edge_preds','edge_labels'))
        to_lightweight_graph(self.full_graph)

        final_node_preds = torch.div(overall_node_preds, overall_num_node_preds.view(-1, 1, 1, 1))
        self.full_graph.graph_obj.node_preds = final_node_preds

    def _project_graph_model_output(self):
        """
        Rounds MPN predictions either via Linear Programming or a greedy heuristic
        """

        if self.eval_params['rounding_method'] == 'greedy':
            projector = GreedyProjector(self.full_graph)

        elif self.eval_params['rounding_method'] == 'exact':
            projector = ExactProjector(self.full_graph, solver_backend=self.eval_params['solver_backend'])

        else:
            raise RuntimeError("Rounding type for projector not understood")

        projector.project()

        self.full_graph.graph_obj = self.full_graph.graph_obj.numpy()
        self.full_graph.constr_satisf_rate = projector.constr_satisf_rate

    def _assign_ped_ids(self):
        """
        Assigns pedestrian Ids to each detection in the sequence, by determining all connected components in the graph
        """
        # Only keep the non-zero edges and Express the result as a CSR matrix so that it can be fed to 'connected_components')
        nonzero_mask = self.full_graph.graph_obj.edge_preds == 1
        nonzero_edge_index = self.full_graph.graph_obj.edge_index.T[nonzero_mask].T
        nonzero_edges = self.full_graph.graph_obj.edge_preds[nonzero_mask].astype(int)
        graph_shape = (self.full_graph.graph_obj.num_nodes, self.full_graph.graph_obj.num_nodes)
        csr_graph = csr_matrix((nonzero_edges, (tuple(nonzero_edge_index))), shape=graph_shape)

        # Get the connected Components:
        n_components, labels = connected_components(csgraph=csr_graph, directed=False, return_labels=True)
        assert len(labels) == self.full_graph.graph_df.shape[0], "Ped Ids Label format is wrong"

        # Each Connected Component is a Ped Id. Assign those values to our DataFrame:
        self.final_projected_output = self.full_graph.graph_df.copy()
        self.final_projected_output['ped_id'] = labels

    def _assign_roi_masks(self):
        """
        Assign RoI masks predicted by the network into the corresponding rows of the df
        """
        assert (self.final_projected_output['detection_id'] == self.full_graph.graph_df['detection_id'].values).all(), \
            'Detection id mismatch between projected output and node predictions!'
        # Store the raw mask predictions in the data frame (conversion to list is required to store inside pd.df)
        self.final_projected_output['roi_masks'] = list(torch.squeeze(self.full_graph.graph_obj.node_preds,
                                                                      1).cpu().numpy())

    def _convert_df_to_output_format(self):
        """
        Get rid of the unnecessary columns of the dataframe and store it in a format which is more compatible with the
        MOTS output format
        """
        self.final_projected_output = self.final_projected_output[VIDEO_COLUMNS + ['conf', 'detection_id']].copy()

    def _to_full_masks(self):
        """
        Transform RoI masks into full masks and ensure that each mask in a frame covers unique pixels
        """
        print('Ensuring unique masks and converting the masks into the output format')
        self.tracking_out['ped_mask'] = None
        for frame in self.tracking_out['frame'].unique():
            frame_ixs = self.tracking_out.index[self.tracking_out.frame == frame]
            frame_roi_masks = self.tracking_out.loc[frame_ixs, 'roi_masks'].to_numpy()
            frame_roi_masks = np.array([np.array(x) for x in frame_roi_masks])  # Required as the masks are stored as a list
            frame_roi_masks = np.expand_dims(frame_roi_masks, axis=1)  # (N, C, H, W) format

            frame_boxes = self.tracking_out.loc[frame_ixs, ['bb_left', 'bb_top', 'bb_right', 'bb_bot']].to_numpy()

            frame_height = int(self.tracking_out.loc[frame_ixs[0], 'img_height'])
            frame_width = int(self.tracking_out.loc[frame_ixs[0], 'img_width'])

            # Get the full mask
            frame_masks = paste_masks_in_image(torch.from_numpy(frame_roi_masks),
                                               torch.from_numpy(frame_boxes), (frame_height, frame_width))
            frame_masks = np.squeeze(frame_masks, axis=1)

            frame_masks = ensure_unique_masks(frame_masks)

            # Make the mask binary
            frame_masks = np.where(frame_masks >= self.eval_params['mask_threshold'], 1, 0).astype(np.uint8)
            frame_masks = np.transpose(frame_masks, (1, 2, 0))  # rletools expects (h, w, n) format for the masks

            # Convert to rle format and save
            frame_rles = rletools.encode(np.asfortranarray(frame_masks))
            frame_rles = np.array([s['counts'].decode(encoding='UTF-8')for s in frame_rles])
            self.tracking_out.loc[frame_ixs, 'ped_mask'] = frame_rles

    def _predict_nan_masks(self):
        """
        Predicts masks for the freshly generated new instances via interpolation by using a segmentation network
        """
        print('Predicting NaN masks')
        self.tracking_out = self.tracking_out.sort_values(by=['ped_id', 'frame']).reset_index(drop=True)  # Sort and reindex
        ix = self.tracking_out.index[self.tracking_out.isnull().any(axis=1) == True]  # rows with NaN mask

        # Only use short ones
        nan_clusters = np.split(ix, np.where(np.diff(ix) != 1)[0]+1)
        valid_ix = pd.Index([item for sublist in nan_clusters if len(sublist) <= 3 for item in sublist])
        nan_ix = pd.Index([item for sublist in nan_clusters if len(sublist) > 3 for item in sublist])

        sub_df = self.tracking_out.loc[valid_ix, ['frame', 'ped_id', 'frame_path', 'bb_left', 'bb_top', 'bb_right', 'bb_bot']]
        to_tensor = ToTensor()

        # Loop over the frames with NaN masks
        with torch.no_grad():
            upsampler = torch.nn.Upsample(scale_factor=2, mode='bilinear')  # Map 28x28 output to 56x56
            for frame in sub_df['frame'].unique():
                frame_ixs = sub_df.index[sub_df.frame == frame]

                # Path is not present sub_df, get the frame path from graph_df
                path_ixs = self.full_graph.graph_df.index[self.full_graph.graph_df.frame == frame]
                frame_path = self.full_graph.graph_df.loc[path_ixs[0], ['frame_path']].to_numpy()

                frame_boxes = torch.from_numpy(sub_df.loc[frame_ixs, ['bb_left', 'bb_top', 'bb_right',
                                                                      'bb_bot']].to_numpy()).float()

                # Read the frame
                img = imread(frame_path[0])
                img = to_tensor(img).unsqueeze(0)

                # Predict masks
                self.mask_model.load_image(img)
                roi_masks = self.mask_model.predict_masks(boxes=frame_boxes, return_roi_masks=True)

                # TODO: deal with different mask shapes before putting them into the df
                roi_masks = upsampler(roi_masks)
                self.tracking_out.loc[frame_ixs, 'roi_masks'] = list(torch.squeeze(roi_masks, 1).cpu().numpy())

            self.tracking_out = self.tracking_out.drop(nan_ix)


    def _ensure_unique_masks(self):
        for frame in self.tracking_out['frame'].unique():
            frame_detects = self.tracking_out[self.tracking_out.frame == frame]
            frame_ixs = self.tracking_out.index[self.tracking_out.frame == frame]
            frame_rles = [{'size': [h, w], 'counts': r.encode(encoding='UTF-8')} for h, w, r in
                       frame_detects[['img_height', 'img_width', 'rle']].values]
            frame_masks = rletools.decode(frame_rles)

            # Make sure that no pixel is occupied twice
            i = np.argmax(frame_masks, axis=2)
            a1, a2 = np.indices((frame_masks.shape[0], frame_masks.shape[1]))
            ix = (a1, a2, i)
            zs = frame_masks == 0


            safe_masks = np.zeros_like(frame_masks)
            safe_masks[ix] = 1
            safe_masks[zs] = 0

            safe_rles = rletools.encode(safe_masks)
            safe_rles = np.array([s['counts'].decode(encoding='UTF-8')for s in safe_rles])
            self.tracking_out.loc[frame_ixs, 'rle'] = safe_rles

    def track(self, seq_name, pred_oracle_mode=None):
        """
        Main method. Given a sequence name, it tracks all detections and produces an output DataFrame, where each
        detection is assigned an ID.

        It starts loading a the graph corresponding to an entire video sequence and detections, then uses an MPN to
        sequentially evaluate batches of frames (i.e. subgraphs) and finally rounds predictions and applies
        postprocessing.

        """
        # Load the graph corresponding to the entire sequence
        self.full_graph = self._load_full_seq_graph_object(seq_name)

        # Feed graph through MPN in batches
        self._evaluate_graph_in_batches(pred_oracle_mode=pred_oracle_mode)

        # Round predictions and assign IDs to trajectories
        self._project_graph_model_output()
        self._assign_ped_ids()
        self._assign_roi_masks()
        self._convert_df_to_output_format()

        postprocess = Postprocessor(self.final_projected_output.copy(),
                                    seq_info_dict= self.dataset.seq_info_dicts[seq_name],
                                    eval_params=self.eval_params)

        self.tracking_out = postprocess.postprocess_trajectories()
        self._to_full_masks()

        return self.tracking_out

    def save_results_to_file(self, output_file_path):
        """
        Stores the tracking result to a txt file, in MOTChallenge format.
        """

        # MOTS id format
        self.tracking_out['ped_id'] += self.tracking_out['label'].values * 1000 + 1

        self.tracking_out['ped_id'] = self.tracking_out['ped_id'].astype('int64')
        self.tracking_out['label'] = self.tracking_out['label'].astype('int64')
        self.tracking_out['img_height'] = self.tracking_out['img_height'].astype('int64')
        self.tracking_out['img_width'] = self.tracking_out['img_width'].astype('int64')



        final_out = self.tracking_out[TRACKING_OUT_COLS].sort_values(by=['frame', 'ped_id'])
        final_out.to_csv(output_file_path, header=False, index=False, sep=' ')
        date = '{date:%m-%d_%H:%M}'.format(date=datetime.now())
        file_name_with_date = output_file_path[:-4] + '_' + date + '.txt'
        final_out.to_csv(file_name_with_date, header=False, index=False, sep=' ')



    def _add_tracktor_detects(self, seq_name):
        def ensure_detects_can_be_used(start_end_per_ped_id):
            """
            We make sure that there is no overlap between MPN trajectories. To do so, we make sure that the ending frame
            for every trajectory is smaller than the starting frame than the next one.
            """
            if start_end_per_ped_id.shape[0] == 1:  # If there is a single detection there is nothing to check
                return True

            start_end_per_ped_id_ = start_end_per_ped_id.sort_values(by='min')

            comparisons = start_end_per_ped_id_['min'].values.reshape(-1, 1) <= start_end_per_ped_id_[
                'max'].values.reshape(1, -1)
            triu_ixs, tril_ixs = np.triu_indices_from(comparisons), np.tril_indices_from(comparisons, k=-1)
            return (comparisons[triu_ixs]).all() & (~comparisons[tril_ixs]).all()


        # Retrieve the complete scene DataFrame
        big_dets_df = self.dataset.seq_det_dfs[seq_name].copy()

        complete_df = self.final_projected_output.merge(big_dets_df[
                                                            ['detection_id', 'tracktor_id', 'frame', 'bb_left',
                                                             'bb_top', 'bb_width', 'bb_height', 'bb_right',
                                                             'bb_bot', 'frame_path', 'img_height', 'img_width', 'rle',
                                                             'label', 'id']], how='outer')
        assert complete_df.shape[0] == big_dets_df.shape[0], "Merging to add tracktor detects did not work properly"
        unique_tracktor_ids = complete_df.tracktor_id.unique()
        complete_df.sort_values(by=['tracktor_id', 'frame'], inplace=True)
        complete_df.set_index('tracktor_id', inplace=True)
        for tracktor_id in unique_tracktor_ids:
            detects_per_tracktor_id = complete_df.loc[tracktor_id][['detection_id', 'ped_id', 'frame']]

            if not isinstance(detects_per_tracktor_id,
                              pd.Series):  # If there is a single detect, then there's nothing to do
                initial_num_of_dets = detects_per_tracktor_id['ped_id'].isnull().sum()
                # For each MPN id, determine which detections under this 'tracktor id
                start_end_per_ped_id = \
                    detects_per_tracktor_id[detects_per_tracktor_id.ped_id.notnull()].groupby(['ped_id'])[
                        'frame'].agg(
                        ['min', 'max'])

                if ensure_detects_can_be_used(start_end_per_ped_id):
                    # We will build an empty assignment array, to give tracktor detects their id
                    ped_ids = np.empty(detects_per_tracktor_id.shape[0])
                    ped_ids[...] = np.nan
                    for ped_id, (start_frame, end_frame) in start_end_per_ped_id.iterrows():
                        ixs = np.where(detects_per_tracktor_id['frame'].between(start_frame, end_frame))[0]
                        ped_ids[ixs] = ped_id

                    if self.eval_params['use_tracktor_start_ends']:
                        assigned_ped_ids_ixs = np.where(~np.isnan(ped_ids))[0]
                        if len(assigned_ped_ids_ixs) > 0:
                            first_ped_id_ix, last_ped_id_ix = assigned_ped_ids_ixs.min(), assigned_ped_ids_ixs.max()
                            ped_ids[:first_ped_id_ix + 1] = ped_ids[first_ped_id_ix]
                            ped_ids[last_ped_id_ix + 1:] = ped_ids[last_ped_id_ix]


                    # Finally, assign the ped_ids to the given
                    complete_df.loc[tracktor_id, 'ped_id'] = ped_ids.reshape(-1, 1)
                else:

                    assign_ped_ids_ixs = sorted(np.where(detects_per_tracktor_id.ped_id.notnull())[0])
                    assign_ped_ids = detects_per_tracktor_id.iloc[assign_ped_ids_ixs]['ped_id']
                    changes = np.where((assign_ped_ids[:-1] - assign_ped_ids[1:]) != 0)[0]
                    # build_intervals

                    # Iterate over id switches among them in order to determines which intervals can be safely interpolated
                    start_ix = assign_ped_ids_ixs[0]
                    # curr_ped_id = assign_ped_ids.iloc[start_ix]
                    curr_ped_id = assign_ped_ids.iloc[0]
                    # curr_ped_id = assign_ped_ids.iloc[0]
                    interv_dict = {ped_id: [] for ped_id in assign_ped_ids}
                    for change in changes:
                        interv_dict[curr_ped_id].append(np.arange(start_ix, assign_ped_ids_ixs[change] + 1))

                        start_ix = assign_ped_ids_ixs[change + 1]  # Next ped id appearance
                        curr_ped_id = assign_ped_ids.iloc[change + 1]

                    # Append the last interval
                    end_ix = assign_ped_ids_ixs[-1]
                    interv_dict[curr_ped_id].append(np.arange(start_ix, end_ix + 1))

                    # Create the id assignment array
                    ped_ids = np.empty(detects_per_tracktor_id.shape[0])
                    ped_ids[...] = np.nan
                    for ped_id, ixs_list in interv_dict.items():
                        if len(ixs_list) > 0:
                            all_ixs = np.concatenate(ixs_list)
                            ped_ids[all_ixs] = ped_id

                    # TODO: Repeated code.
                    if self.eval_params['use_tracktor_start_ends']:
                        if len(assign_ped_ids_ixs) > 0:
                            first_ped_id_ix, last_ped_id_ix = assign_ped_ids_ixs[0], assign_ped_ids_ixs[-1]
                            ped_ids[:first_ped_id_ix + 1] = ped_ids[first_ped_id_ix]
                            ped_ids[last_ped_id_ix + 1:] = ped_ids[last_ped_id_ix]

                    complete_df.loc[tracktor_id, 'ped_id'] = ped_ids.reshape(-1, 1)
                    # print_or_log(f"Recovered {(~np.isnan(ped_ids)).sum()} detects", self.logger)

        # Our final DataFrame is this one!!!!!!!!!!!!!!!!
        final_out = complete_df[complete_df.ped_id.notnull()].reset_index()
        final_out['conf'] = final_out['conf'].fillna(1)

        # If some rare cases two dets in the same frame may get mapped to the same id, just average coordinates:
        #final_out = final_out.groupby(['frame', 'frame_path', 'ped_id']).mean().reset_index()

        # If some rare cases two dets in the same frame may get mapped to the same id, just choose one:
        final_out = final_out.drop_duplicates(subset=['frame', 'frame_path', 'ped_id']).reset_index()

        assert final_out[['frame', 'ped_id']].drop_duplicates().shape[0] == final_out.shape[0]

        return final_out