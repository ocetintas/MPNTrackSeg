import torch
from torch import nn

from torch_scatter import scatter_mean, scatter_max, scatter_add
from torch_scatter.composite import scatter_softmax

from mot_neural_solver.models.mlp import MLP
from mot_neural_solver.models.cnn import CNN, MaskRCNNPredictor


class MetaLayer(torch.nn.Module):
    """
    Core Message Passing Network Class. Extracted from torch_geometric, with minor modifications.
    (https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html)
    """
    def __init__(self, edge_model=None, node_model=None):
        """
        Args:
            edge_model: Callable Edge Update Model
            node_model: Callable Node Update Model
        """
        super(MetaLayer, self).__init__()

        self.edge_model = edge_model
        self.node_model = node_model
        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        """
        Does a single node and edge feature vectors update.
        Args:
            x: node features matrix
            edge_index: tensor with shape [2, M], with M being the number of edges, indicating nonzero entries in the
            graph adjacency (i.e. edges)
            edge_attr: edge features matrix (ordered by edge_index)

        Returns: Updated Node and Edge Feature matrices

        """

        # Edge Update
        if self.edge_model is not None:
            edge_attr = self.edge_model(x, edge_index, edge_attr)

        # Node Update
        if self.node_model is not None:
            x = self.node_model(x, edge_index, edge_attr)

        return x, edge_attr

    def __repr__(self):
        return '{}(edge_model={}, node_model={})'.format(self.__class__.__name__, self.edge_model, self.node_model)

class EdgeModel(nn.Module):
    """
    Class used to peform the edge update during Neural message passing
    """
    def __init__(self, edge_model):
        super(EdgeModel, self).__init__()
        self.edge_model = edge_model

    def forward(self, node_feats, edge_index, edge_attr):
        row, col = edge_index
        return self.edge_model(torch.cat([node_feats[row], node_feats[col], edge_attr], dim=1))

class TimeAwareNodeModel(nn.Module):
    """
    Class used to peform the node update during Neural mwssage passing
    """
    def __init__(self, flow_in_model, flow_out_model, node_model, node_agg_fn):
        super(TimeAwareNodeModel, self).__init__()

        self.flow_in_model = flow_in_model
        self.flow_out_model = flow_out_model
        self.node_model = node_model
        self.node_agg_fn = node_agg_fn

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        flow_out_mask = row < col
        flow_out_row, flow_out_col = row[flow_out_mask], col[flow_out_mask]
        flow_out_input = torch.cat([x[flow_out_col], edge_attr[flow_out_mask]], dim=1)
        flow_out = self.flow_out_model(flow_out_input)
        flow_out = self.node_agg_fn(flow_out, flow_out_row, x.size(0))

        flow_in_mask = row > col
        flow_in_row, flow_in_col = row[flow_in_mask], col[flow_in_mask]
        flow_in_input = torch.cat([x[flow_in_col], edge_attr[flow_in_mask]], dim=1)
        flow_in = self.flow_in_model(flow_in_input)

        flow_in = self.node_agg_fn(flow_in, flow_in_row, x.size(0))
        flow = torch.cat((flow_in, flow_out), dim=1)

        return self.node_model(flow)


class TimeAwareAttentionModel(nn.Module):
    """
    Class used to peform the node update during Neural mwssage passing
    """
    def __init__(self, node_model, flow_in_attention_model, flow_out_attention_model):
        super(TimeAwareAttentionModel, self).__init__()

        self.node_model = node_model

    def forward(self, x, edge_index, edge_attr, cls_net):

        row, col = edge_index
        dec_edge_feats, _ = cls_net(edge_attr)


        flow_out_mask = row < col
        flow_out_row, flow_out_col = row[flow_out_mask], col[flow_out_mask]
        flow_out_weights = dec_edge_feats[flow_out_mask]


        flow_out_weights = scatter_softmax(flow_out_weights, flow_out_row, dim=0)
        flow_out = x[flow_out_col]*flow_out_weights[:, :, None, None]  # Element-wise multiplication with neighbors
        flow_out = scatter_add(flow_out, flow_out_row, dim=0, dim_size=x.size(0))

        flow_in_mask = row > col
        flow_in_row, flow_in_col = row[flow_in_mask], col[flow_in_mask]



        flow_in_weights = dec_edge_feats[flow_in_mask]
        flow_in_weights = scatter_softmax(flow_in_weights, flow_in_row, dim=0)
        flow_in = x[flow_in_col]*flow_in_weights[:, :, None, None]  # Element-wise multiplication with neighbors
        flow_in = scatter_add(flow_in, flow_in_row, dim=0, dim_size=x.size(0))

        flow = torch.cat((x, flow_in, flow_out), dim=1)
        return self.node_model(flow), dec_edge_feats

class MLPGraphIndependent(nn.Module):
    """
    Class used to to encode (resp. classify) features before (resp. after) neural message passing.
    It consists of two networks, one for nodes and one for edges, and they are applied independently to node and edge
    features, respectively.

    This class is based on: https://github.com/deepmind/graph_nets tensorflow implementation.
    """

    def __init__(self, edge_in_dim = None, node_in_dim = None, edge_out_dim = None, node_out_dim = None,
                 node_dims = None, edge_dims = None, dropout_p = None, use_batchnorm = None):
        super(MLPGraphIndependent, self).__init__()

        if node_in_dim is not None :
            self.node_model = MLP(input_dim=node_in_dim, fc_dims=list(node_dims) + [node_out_dim],
                                  dropout_p=dropout_p, use_batchnorm=use_batchnorm)
        else:
            self.node_model = None

        if edge_in_dim is not None :
            self.edge_model = MLP(input_dim=edge_in_dim, fc_dims=list(edge_dims) + [edge_out_dim],
                                  dropout_p=dropout_p, use_batchnorm=use_batchnorm)
        else:
            self.edge_model = None

    def forward(self, edge_feats = None, nodes_feats = None):

        if self.node_model is not None:
            out_node_feats = self.node_model(nodes_feats)

        else:
            out_node_feats = nodes_feats

        if self.edge_model is not None:
            out_edge_feats = self.edge_model(edge_feats)

        else:
            out_edge_feats = edge_feats

        return out_edge_feats, out_node_feats

class MaskModel(nn.Module):
    """
    Class used to perform mask predictions
    """
    def __init__(self, mask_model_params):
        super(MaskModel, self).__init__()

        feature_encoder_feats_dict = mask_model_params['feature_encoder_feats_dict']
        mask_head_feats_dict = mask_model_params['mask_head_feats_dict']
        mask_predictor_feats_dict = mask_model_params['mask_predictor_feats_dict']

        # Simple feature encoder network to reduce the number of channels obtained from the backbone
        self.feature_encoder = CNN(**feature_encoder_feats_dict)
        self.layer_norm = nn.LayerNorm([64, 14, 14])

        # Mask head and mask predictor inspired from the MaskRCNN
        self.mask_head = CNN(**mask_head_feats_dict)

        self.mask_predictor = MaskRCNNPredictor(**mask_predictor_feats_dict)

    def forward(self, feature_embeds, node_embeds):
        feature_embeds = self.feature_encoder(feature_embeds)
        x = torch.cat((feature_embeds, node_embeds), dim=1)
        x = self.layer_norm(x)
        x = self.mask_head(x)
        x = self.mask_predictor(x)
        return x


class MOTMPNet(nn.Module):
    """
    Main Model Class. Contains all the components of the model. It consists of of several networks:
    - 2 encoder networks (1 for nodes, 1 for edges) that provide the initial node and edge embeddings, respectively,
    - 4 update networks (3 for nodes, 1 per edges used in the 'core' Message Passing Network
    - 1 edge classifier MLP that performs binary classification over the Message Passing Network's output.
    - 1 mask refinement network that performs mask prediction over the Message Passing Network's output.

    This class was initially based on: https://github.com/deepmind/graph_nets tensorflow implementation.
    """

    def __init__(self, model_params, bb_encoder = None):
        """
        Defines all components of the model
        Args:
            bb_encoder: (might be 'None') CNN used to encode bounding box apperance information.
            model_params: dictionary contaning all model hyperparameters
        """
        super(MOTMPNet, self).__init__()

        self.node_cnn = bb_encoder
        self.model_params = model_params

        # Define Encoder and Classifier Networks
        encoder_feats_dict = model_params['encoder_feats_dict']
        classifier_feats_dict = model_params['classifier_feats_dict']
        node_ext_encoder_feats_dict = model_params['node_ext_encoder_feats_dict']

        self.encoder = MLPGraphIndependent(**encoder_feats_dict)
        self.classifier = MLPGraphIndependent(**classifier_feats_dict)
        self.node_ext_encoder = CNN(**node_ext_encoder_feats_dict)

        self.mask_predictor = MaskModel(model_params['mask_model_feats_dict'])

        # Define the 'Core' message passing network (i.e. node and edge update models)
        self.MPNet = self._build_core_MPNet(model_params=model_params, encoder_feats_dict=encoder_feats_dict)

        # Define 2nd MPN
        self.MPAttentionNet = self._build_attention_MPNet(model_params=model_params)

        self.num_enc_steps = model_params['num_enc_steps']
        self.num_class_steps = model_params['num_class_steps']

        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _build_core_MPNet(self, model_params, encoder_feats_dict):
        """
        Builds the core part of the Message Passing Network: Node Update and Edge Update models.
        Args:
            model_params: dictionary contaning all model hyperparameters
            encoder_feats_dict: dictionary containing the hyperparameters for the initial node/edge encoder
        """

        # Define an aggregation operator for nodes to 'gather' messages from incident edges
        node_agg_fn = model_params['node_agg_fn']
        assert node_agg_fn.lower() in ('mean', 'max', 'sum'), "node_agg_fn can only be 'max', 'mean' or 'sum'."

        if node_agg_fn == 'mean':
            node_agg_fn = lambda out, row, x_size: scatter_mean(out, row, dim=0, dim_size=x_size)

        elif node_agg_fn == 'max':
            node_agg_fn = lambda out, row, x_size: scatter_max(out, row, dim=0, dim_size=x_size)[0]

        elif node_agg_fn == 'sum':
            node_agg_fn = lambda out, row, x_size: scatter_add(out, row, dim=0, dim_size=x_size)

        # Define all MLPs involved in the graph network
        self.reattach_initial_nodes = model_params['reattach_initial_nodes']
        self.reattach_initial_edges = model_params['reattach_initial_edges']

        self.edge_factor = 2 if self.reattach_initial_edges else 1
        self.node_factor = 2 if self.reattach_initial_nodes else 1

        edge_model_in_dim = self.node_factor * 2 * encoder_feats_dict['node_out_dim'] + \
                            self.edge_factor * encoder_feats_dict['edge_out_dim']

        node_model_in_dim = self.node_factor * encoder_feats_dict['node_out_dim'] + encoder_feats_dict['edge_out_dim']

        attention_model_in_dim = encoder_feats_dict['edge_out_dim']

        # Define all MLPs used within the MPN
        edge_model_feats_dict = model_params['edge_model_feats_dict']
        node_model_feats_dict = model_params['node_model_feats_dict']
        # attention_model_feats_dict = model_params['attention_model_feats_dict']

        edge_model = MLP(input_dim=edge_model_in_dim,
                         fc_dims=edge_model_feats_dict['dims'],
                         dropout_p=edge_model_feats_dict['dropout_p'],
                         use_batchnorm=edge_model_feats_dict['use_batchnorm'])

        flow_in_model = MLP(input_dim=node_model_in_dim,
                          fc_dims=node_model_feats_dict['dims'],
                          dropout_p=node_model_feats_dict['dropout_p'],
                          use_batchnorm=node_model_feats_dict['use_batchnorm'])

        flow_out_model = MLP(input_dim=node_model_in_dim,
                           fc_dims=node_model_feats_dict['dims'],
                           dropout_p=node_model_feats_dict['dropout_p'],
                           use_batchnorm=node_model_feats_dict['use_batchnorm'])

        node_model = nn.Sequential(*[nn.Linear(2 * encoder_feats_dict['node_out_dim'], encoder_feats_dict['node_out_dim']),
                                   nn.ReLU(inplace=True)])

        # Define all MLPs used within the MPN
        return MetaLayer(edge_model=EdgeModel(edge_model = edge_model),
                         node_model=TimeAwareNodeModel(flow_in_model = flow_in_model,
                                                       flow_out_model = flow_out_model,
                                                       node_model = node_model,
                                                       node_agg_fn = node_agg_fn))

    def _build_attention_MPNet(self, model_params):
        attention_model_feats_dict = model_params['attention_model_feats_dict']
        node_ext_model_feats_dict = model_params['node_ext_model_feats_dict']

        attention_model_in_dim = model_params['encoder_feats_dict']['edge_out_dim'] * self.edge_factor
        node_ext_model_in_dim = 3 * model_params['node_ext_encoder_feats_dict']['dims'][-1] * self.node_factor

        flow_in_attention_model = MLP(input_dim=attention_model_in_dim, **attention_model_feats_dict)
        flow_out_attention_model = MLP(input_dim=attention_model_in_dim, **attention_model_feats_dict)
        node_ext_model = CNN(input_dim=node_ext_model_in_dim, **node_ext_model_feats_dict)

        return TimeAwareAttentionModel(node_model=node_ext_model, flow_in_attention_model=flow_in_attention_model,
                                       flow_out_attention_model=flow_out_attention_model)

    def forward(self, data):
        """
        Provides a fractional solution to the data association problem.
        First, node and edge features are independently encoded by the encoder network. Then, they are iteratively
        'combined' for a fixed number of steps via the Message Passing Network (self.MPNet). Finally, they are
        classified independently by the classifiernetwork.
        Args:
            data: object containing attribues
              - x: node features matrix
              - edge_index: tensor with shape [2, M], with M being the number of edges, indicating nonzero entries in the
                graph adjacency (i.e. edges) (i.e. sparse adjacency)
              - edge_attr: edge features matrix (sorted by edge apperance in edge_index)

        Returns:
            classified_edges: list of unnormalized node probabilites after each MP step
        """
        x, x_ext, edge_index, edge_attr = data.x, data.x_ext, data.edge_index, data.edge_attr

        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)

        # Encoding features step
        latent_edge_feats, latent_node_feats = self.encoder(edge_attr, x)
        latent_node_ext_feats = self.node_ext_encoder(x_ext)

        initial_edge_feats = latent_edge_feats
        initial_node_feats = latent_node_feats
        initial_node_ext_feats = latent_node_ext_feats

        # During training, the feature vectors that the MPNetwork outputs for the  last self.num_class_steps message
        # passing steps are classified in order to compute the loss.
        first_class_step = self.num_enc_steps - self.num_class_steps + 1
        outputs_dict = {'classified_edges': [], 'mask_predictions': []}
        for step in range(1, self.num_enc_steps + 1):

            # Reattach the initially encoded embeddings before the update
            if self.reattach_initial_edges:
                latent_edge_feats = torch.cat((initial_edge_feats, latent_edge_feats), dim=1)
            if self.reattach_initial_nodes:
                latent_node_feats = torch.cat((initial_node_feats, latent_node_feats), dim=1)
                latent_node_ext_feats = torch.cat((initial_node_ext_feats, latent_node_ext_feats), dim=1)

            # Message Passing Step
            latent_node_feats, latent_edge_feats = self.MPNet(latent_node_feats, edge_index, latent_edge_feats)
            latent_node_ext_feats, dec_edge_feats = self.MPAttentionNet(latent_node_ext_feats, edge_index, latent_edge_feats, self.classifier)

            if step >= first_class_step:
                # Classification Step
                outputs_dict['classified_edges'].append(dec_edge_feats)

                # Mask Prediction Step
                mask_preds = self.mask_predictor(x_ext, latent_node_ext_feats)
                outputs_dict['mask_predictions'].append(mask_preds)

        if self.num_enc_steps == 0:
            dec_edge_feats, _ = self.classifier(latent_edge_feats)
            outputs_dict['classified_edges'].append(dec_edge_feats)

            mask_preds = self.mask_predictor(x_ext, latent_node_ext_feats)
            outputs_dict['mask_predictions'].append(mask_preds)

        return outputs_dict