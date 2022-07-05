import torch
import  torch.nn.functional as F

import numpy as np

from mot_neural_solver.data.augmentation import MOTGraphAugmentor

from mot_neural_solver.utils.graph import get_time_valid_conn_ixs, get_knn_mask, \
    compute_edge_feats_dict, construct_net_flow_id_matrix
from mot_neural_solver.utils.rgb import load_embeddings_from_imgs, load_precomputed_embeddings

from torch_geometric.data import Data
import pycocotools.mask as rletools
from torchvision.ops import roi_align
import time

import matplotlib.pyplot as plt
import os.path as osp
from torch_scatter import scatter_min

class Graph(Data):
    """
    This is the class we use to instantiate our graph objects. We inherit from torch_geometric's Data class and add a
    few convenient methods to it, mostly related to changing data types in a single call.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _change_attrs_types(self, attr_change_fn):
        """
        Base method for all methods related to changing attribute types. Iterates over the attributes names in
        _data_attr_names, and changes its type via attr_change_fun

        Args:
            attr_change_fn: callable function to change a variable's type
        """
        # These are our standard 'data-related' attribute names.
        _data_attr_names = ['x', # Node feature vecs
                            'x_ext', # 3D Node features
                           'edge_attr', # Edge Feature vecs
                           'edge_index', # Sparse Adjacency matrix
                           'mask_attr', # Mask prediction from the detections
                           'node_names', # Node names (integer values)
                           'edge_labels', # Edge labels according to Network Flow MOT formulation
                           'edge_preds', # Predicted approximation to edge labels
                           'reid_emb_dists'] # Reid distance for each edge

        for attr_name in _data_attr_names:
            if hasattr(self, attr_name):
                if getattr(self, attr_name ) is not None:
                    old_attr_val = getattr(self, attr_name)
                    setattr(self, attr_name, attr_change_fn(old_attr_val))

    def tensor(self):
        self._change_attrs_types(attr_change_fn= torch.tensor)
        return self

    def float(self):
        self._change_attrs_types(attr_change_fn= lambda x: x.float())
        return self

    def numpy(self):
        self._change_attrs_types(attr_change_fn= lambda x: x if isinstance(x, np.ndarray) else x.detach().cpu().numpy())
        return self

    def cpu(self):
        #self.tensor()
        self._change_attrs_types(attr_change_fn= lambda x: x.cpu())
        return self

    def cuda(self):
        #self.tensor()
        self._change_attrs_types(attr_change_fn=lambda x: x.cuda())
        return self

    def to(self, device):
        self._change_attrs_types(attr_change_fn=lambda x: x.to(device))

    def device(self):
        if isinstance(self.edge_index, torch.Tensor):
            return self.edge_index.device

        return torch.device('cpu')

class MOTGraph(object):
    """
    This the main class we use to create MOT graphs from detection (and possibly ground truth) files. Its main attribute
    is 'graph_obj', which is an instance of the class 'Graph' and serves as input to the tracking model.

    Moreover, each 'MOTGraph' has several additional attributes that provide further information about the detections in
    the subset of frames from which the graph is constructed.

    """
    def __init__(self, seq_det_df = None, start_frame = None, end_frame = None, ensure_end_is_in = False, step_size = None,
                 seq_info_dict = None, dataset_params = None, inference_mode = False, max_frame_dist = None):
        self.dataset_params = dataset_params
        self.step_size = step_size
        self.seq_info_dict = seq_info_dict
        self.inference_mode = inference_mode
        self.max_frame_dist = max_frame_dist

        if seq_det_df is not None:
            self.graph_df, self.frames = self._construct_graph_df(seq_det_df= seq_det_df.copy(),
                                                                  start_frame = start_frame,
                                                                  end_frame = end_frame,
                                                                  ensure_end_is_in=ensure_end_is_in)

    def _construct_graph_df(self, seq_det_df, start_frame, end_frame = None, ensure_end_is_in = False):
        """
        Determines which frames will be in the graph, and creates a DataFrame with its detection's information.

        Args:
            seq_det_df: DataFrame with scene detections information
            start_frame: frame at which the graph starts
            end_frame: (optional) frame at which the graph ends
            ensure_end_is_in: (only if end_frame is given). Bool indicating whether end_frame must be in the graph.

        Returns:
            graph_df: DataFrame with rows of scene_df between the selected frames
            valid_frames: list of selected frames

        """
        if end_frame is not None:
            # Just load all frames between start_frame and end_frame at the desired step size
            valid_frames = np.arange(start_frame, end_frame + 1, self.step_size)

            if ensure_end_is_in and (end_frame not in valid_frames):
                valid_frames = valid_frames.tolist() + [end_frame]

        else:
            # Consider all posible future frames (at distance step_size)
            valid_frames = np.arange(start_frame, seq_det_df.frame.max(), self.step_size)

            # We cannot have more than dataset_params['frames_per_graph'] frames
            if self.dataset_params['frames_per_graph'] != 'max':
                valid_frames = valid_frames[:self.dataset_params['frames_per_graph']]

            # We cannot have more than dataset_params['max_detects'] detections
            if self.dataset_params['max_detects'] is not None:
                scene_df_ = seq_det_df[seq_det_df.frame.isin(valid_frames)].copy()
                frames_cumsum = scene_df_.groupby('frame')['bb_left'].count().cumsum()
                valid_frames = frames_cumsum[frames_cumsum <= self.dataset_params['max_detects']].index

        graph_df = seq_det_df[seq_det_df.frame.isin(valid_frames)].copy()
        graph_df = graph_df.sort_values(by=['frame', 'detection_id']).reset_index(drop=True)

        return graph_df, sorted(graph_df.frame.unique())

    def augment(self):
        augmentor = MOTGraphAugmentor(graph_df=self.graph_df, dataset_params=self.dataset_params)
        self.graph_df = augmentor.augment()

    def _load_appearance_data(self):
        """
        Loads embeddings for node features and reid.
        Returns:
            tuple with (reid embeddings, node_feats), both are torch.tensors with shape (num_nodes, embed_dim)
        """
        # if self.inference_mode and not self.dataset_params['precomputed_embeddings']:
        #     assert self.cnn_model is not None
        #     print("USING CNN FOR APPEARANCE")
        #     _, node_feats, reid_embeds = load_embeddings_from_imgs(det_df = self.graph_df,
        #                                                             dataset_params = self.dataset_params,
        #                                                             seq_info_dict = self.seq_info_dict,
        #                                                             cnn_model = self.cnn_model,
        #                                                             return_imgs = False,
        #                                                             use_cuda = self.inference_mode)
        #
        # else:

        emb_dir = osp.join('embeddings', self.seq_info_dict['det_file_name'])
        reid_embeds = load_precomputed_embeddings(det_df=self.graph_df,
                                                  seq_info_dict=self.seq_info_dict,
                                                  embeddings_dir=osp.join(emb_dir, self.dataset_params['reid_embeddings_dir']),
                                                  use_cuda=self.inference_mode)
        if self.dataset_params['reid_embeddings_dir'] == self.dataset_params['node_core_embeddings_dir']:
            node_core_feats = reid_embeds.clone()
        else:
            node_core_feats =  load_precomputed_embeddings(det_df=self.graph_df,
                                                     seq_info_dict=self.seq_info_dict,
                                                     embeddings_dir=osp.join(emb_dir, self.dataset_params['node_core_embeddings_dir']),
                                                     use_cuda=self.inference_mode,
                                                     embedding_dim='3D')

        #t = time.time()
        node_ext_feats = load_precomputed_embeddings(det_df=self.graph_df,
                                                 seq_info_dict=self.seq_info_dict,
                                                 embeddings_dir=osp.join(emb_dir, self.dataset_params['node_ext_embeddings_dir']),
                                                 use_cuda=self.inference_mode,
                                                 embedding_dim='3D')
        #print(time.time() - t)

        return reid_embeds, node_core_feats, node_ext_feats

    def _get_edge_ixs(self, reid_embeddings):
        """
        Constructs graph edges by taking pairs of nodes with valid time connections (not in same frame, not too far
        apart in time) and perhaps taking KNNs according to reid embeddings.
        Args:
            reid_embeddings: torch.tensor with shape (num_nodes, reid_embeds_dim)

        Returns:
            torch.tensor withs shape (2, num_edges)
        """

        edge_ixs = get_time_valid_conn_ixs(frame_num = torch.from_numpy(self.graph_df.frame.values),
                                           max_frame_dist = self.max_frame_dist, use_cuda=self.inference_mode and self.graph_df['frame_path'].iloc[0].find('MOT17-03') == -1)

        # During inference, top k nns must not be done here, as it is computed independently for sequence chunks
        if not self.inference_mode and self.dataset_params['top_k_nns'] is not None:
            reid_pwise_dist = F.pairwise_distance(reid_embeddings[edge_ixs[0]], reid_embeddings[edge_ixs[1]])
            k_nns_mask = get_knn_mask(pwise_dist = reid_pwise_dist,
                                      edge_ixs = edge_ixs,
                                      num_nodes = self.graph_df.shape[0],
                                      top_k_nns = self.dataset_params['top_k_nns'],
                                      reciprocal_k_nns = self.dataset_params['reciprocal_k_nns'],
                                      symmetric_edges = False,
                                      use_cuda=self.inference_mode)
            edge_ixs = edge_ixs.T[k_nns_mask].T

        return edge_ixs

    def assign_edge_labels(self):
        """
        Assigns self.graph_obj edge labels (tensor with shape (num_edges,)), with labels defined according to the
        network flow MOT formulation
        """
        mode = self.dataset_params['true_edge_labels']
        ids = torch.as_tensor(self.graph_df.id.values, device=self.graph_obj.edge_index.device)
        per_edge_ids = torch.stack([ids[self.graph_obj.edge_index[0]], ids[self.graph_obj.edge_index[1]]])
        same_id = (per_edge_ids[0] == per_edge_ids[1]) & (per_edge_ids[0] != -1)
        self.graph_obj.edge_labels = torch.zeros_like(same_id, dtype=torch.float)

        if mode == 'all':
            self.graph_obj.edge_labels[same_id] = 1
        elif mode == 'closest':
            same_ids_ixs = torch.where(same_id)
            same_id_edges = self.graph_obj.edge_index.T[same_id].T

            time_dists = torch.abs(same_id_edges[0] - same_id_edges[1])

            # For every node, we get the index of the node in the future (resp. past) with the same id that is closest in time
            future_mask = same_id_edges[0] < same_id_edges[1]
            active_fut_edges = \
            scatter_min(time_dists[future_mask], same_id_edges[0][future_mask], dim=0, dim_size=self.graph_obj.num_nodes)[1]
            original_node_ixs = torch.cat((same_id_edges[1][future_mask], torch.as_tensor([-1],
                                                                                          device=same_id.device)))  # -1 at the end for nodes that were not present
            active_fut_edges = original_node_ixs[active_fut_edges]  # Recover the node id of the corresponding
            fut_edge_is_active = active_fut_edges[same_id_edges[0]] == same_id_edges[1]

            # Analogous for past edges
            past_mask = same_id_edges[0] > same_id_edges[1]
            active_past_edges = \
            scatter_min(time_dists[past_mask], same_id_edges[0][past_mask], dim=0, dim_size=self.graph_obj.num_nodes)[1]
            original_node_ixs = torch.cat((same_id_edges[1][past_mask], torch.as_tensor([-1],
                                                                                        device=same_id.device)))  # -1 at the end for nodes that were not present
            active_past_edges = original_node_ixs[active_past_edges]
            past_edge_is_active = active_past_edges[same_id_edges[0]] == same_id_edges[1]

            # Recover the ixs of active edges in the original edge_index tensor o
            active_edge_ixs = same_ids_ixs[0][past_edge_is_active | fut_edge_is_active]
            self.graph_obj.edge_labels[active_edge_ixs] = 1

    def assign_mask_labels(self):
        """
        Converts ground truth masks into prediction format (Convert to binary mask + RoI Align)
        """
        gt_dir = osp.join('gt', 'gt_mask')

        mask_labels = load_precomputed_embeddings(det_df=self.graph_df,
                                                  seq_info_dict=self.seq_info_dict,
                                                  embeddings_dir=osp.join(gt_dir, 'masks'),
                                                  use_cuda=self.inference_mode,
                                                  embedding_dim='3D')
        self.graph_obj.mask_labels = mask_labels.to(self.graph_obj.device())

        mask_gt_ixs = load_precomputed_embeddings(det_df=self.graph_df,
                                                  seq_info_dict=self.seq_info_dict,
                                                  embeddings_dir=osp.join(gt_dir, 'valid_ixs'),
                                                  use_cuda=self.inference_mode)
        self.graph_obj.mask_gt_ixs = mask_gt_ixs.view(-1).bool().to(self.graph_obj.device())

    def construct_graph_object(self):
        """
        Constructs the entire Graph object to serve as input to the MPN, and stores it in self.graph_obj,
        """
        # Load Appearance Data
        reid_embeddings, node_core_feats, node_ext_feats = self._load_appearance_data()

        # Determine graph connectivity (i.e. edges) and compute edge features
        edge_ixs = self._get_edge_ixs(reid_embeddings)
        edge_feats_dict = compute_edge_feats_dict(edge_ixs = edge_ixs, det_df = self.graph_df,
                                                  fps = self.seq_info_dict['fps'],
                                                  use_cuda = self.inference_mode)
        edge_feats = [edge_feats_dict[feat_names] for feat_names in self.dataset_params['edge_feats_to_use'] if feat_names in edge_feats_dict]
        edge_feats = torch.stack(edge_feats).T

        # Compute embeddings distances. Pairwise distance computation might create out of memmory errors, hence we batch it
        emb_dists = []
        for i in range(0, edge_ixs[0].shape[0], 50000):
            emb_dists.append(F.pairwise_distance(reid_embeddings[edge_ixs[0][i:i + 50000]],
                                                 reid_embeddings[edge_ixs[1][i:i + 50000]]).view(-1, 1))
        emb_dists = torch.cat(emb_dists, dim=0)

        # Add embedding distances to edge features if needed
        if 'emb_dist' in self.dataset_params['edge_feats_to_use']:
            edge_feats = torch.cat((edge_feats, emb_dists), dim = 1)

        self.graph_obj = Graph(x = node_core_feats,
                               x_ext = node_ext_feats,
                               edge_attr = torch.cat((edge_feats, edge_feats), dim = 0),
                               edge_index = torch.cat((edge_ixs, torch.stack((edge_ixs[1], edge_ixs[0]))), dim=1))

        if self.inference_mode:
            self.graph_obj.reid_emb_dists = torch.cat((emb_dists, emb_dists))

        self.graph_obj.to(torch.device("cuda" if torch.cuda.is_available() and self.inference_mode else "cpu"))