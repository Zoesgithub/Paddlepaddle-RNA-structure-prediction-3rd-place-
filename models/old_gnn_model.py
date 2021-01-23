from .supermodel import supermodel
from .egsage import egsage
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
class GNNStack(supermodel):
    def __init__(self, 
                node_input_dim, edge_input_dim,
                node_dim, edge_dim, edge_mode,
                model_types, dropout, activation,
                concat_states, node_post_mlp_hiddens,
                normalize_embs, aggr
                ):
        super(GNNStack, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.concat_states = concat_states
        self.model_types = model_types
        self.gnn_layer_num = len(model_types)

        # convs
        node_input_dim=100
        self.convs = self.build_convs(node_input_dim, edge_input_dim,
                                    node_dim, edge_dim, edge_mode,
                                    model_types, normalize_embs, activation, aggr)
        self.embedding=nn.Embedding(13459+1, 100)

        # post node update
        if concat_states:
            self.node_post_mlp = self.build_node_post_mlp(int(node_dim*len(model_types)), int(node_dim*len(model_types)), node_post_mlp_hiddens, dropout, activation)
        else:
            self.node_post_mlp = self.build_node_post_mlp(node_dim, node_dim, node_post_mlp_hiddens, dropout, activation)

        self.edge_update_mlps = self.build_edge_update_mlps(node_dim, edge_input_dim, edge_dim, self.gnn_layer_num, activation)
        self.out_edge_matrix=nn.parameter.Parameter(data=torch.rand(20, 80))
        self.out_linear=nn.Linear(node_dim, 100)

        #self.out_edge_linear=nn.Sequential(nn.Linear(node_dim*2, 128), nn.ReLU(), nn.Linear(128, 1), nn.ReLU())
        #self.out_node_linear=nn.Sequential(nn.Linear(node_dim, 128), nn.ReLU(), nn.Linear(128,20))
        self.optimizer=torch.optim.Adam(self.parameters(), lr=1e-3)

    def build_node_post_mlp(self, input_dim, output_dim, hidden_dims, dropout, activation):
        if 0 in hidden_dims:
            return activation()#get_activation('none')
        else:
            layers = []
            for hidden_dim in hidden_dims:
                layer = nn.Sequential(
                            nn.Linear(input_dim, hidden_dim),
                            activation(),
                            nn.Dropout(dropout),
                            )
                layers.append(layer)
                input_dim = hidden_dim
            layer = nn.Linear(input_dim, output_dim)
            layers.append(layer)
            return nn.Sequential(*layers)

    def build_convs(self, node_input_dim, edge_input_dim,
                     node_dim, edge_dim, edge_mode,
                     model_types, normalize_embs, activation, aggr):
        convs = nn.ModuleList()
        conv = self.build_conv_model(model_types[0],node_input_dim,node_dim,
                                    edge_input_dim, edge_mode, normalize_embs[0], activation, aggr)
        convs.append(conv)
        for l in range(1,len(model_types)):
            conv = self.build_conv_model(model_types[l],node_dim, node_dim,
                                    edge_dim, edge_mode, normalize_embs[l], activation, aggr)
            convs.append(conv)
        return convs

    def build_conv_model(self, model_type, node_in_dim, node_out_dim, edge_dim, edge_mode, normalize_emb, activation, aggr):
        #print(model_type)
        if model_type == 'GCN':
            return pyg_nn.GCNConv(node_in_dim,node_out_dim)
        elif model_type == 'GraphSage':
            return pyg_nn.SAGEConv(node_in_dim,node_out_dim)
        elif model_type == 'GAT':
            return pyg_nn.GATConv(node_in_dim,node_out_dim)
        elif model_type == 'EGCN':
            return EGCNConv(node_in_dim,node_out_dim,edge_dim,edge_mode)
        elif model_type == 'EGSAGE':
            return egsage(node_in_dim,node_out_dim,edge_dim,activation,edge_mode,normalize_emb, aggr)

    def build_edge_update_mlps(self, node_dim, edge_input_dim, edge_dim, gnn_layer_num, activation):
        edge_update_mlps = nn.ModuleList()
        edge_update_mlp = nn.Sequential(
                nn.Linear(node_dim+node_dim+edge_input_dim,edge_dim),
                activation(),
                )
        edge_update_mlps.append(edge_update_mlp)
        for l in range(1,gnn_layer_num):
            edge_update_mlp = nn.Sequential(
                nn.Linear(node_dim+node_dim+edge_dim,edge_dim),
                activation(),
                )
            edge_update_mlps.append(edge_update_mlp)
        return edge_update_mlps

    def update_edge_attr(self, x, edge_attr, edge_index, mlp):
        x_i = x[edge_index[0],:]
        x_j = x[edge_index[1],:]
        edge_attr = mlp(torch.cat((x_i,x_j,edge_attr),dim=-1))
        return edge_attr

    def forward(self, x, edge_attr, edge_index):
        if self.concat_states:
            concat_x = []
        for l,(conv_name,conv) in enumerate(zip(self.model_types,self.convs)):
            # self.check_input(x,edge_attr,edge_index)
            if conv_name == 'EGCN' or conv_name == 'EGSAGE':
                #print(x.shape, edge_attr.shape, edge_index.shape)
                x = conv(x, edge_attr, edge_index)
            else:
                x = conv(x, edge_index)
            if self.concat_states:
                concat_x.append(x)
            edge_attr = self.update_edge_attr(x, edge_attr, edge_index, self.edge_update_mlps[l])
            #print(edge_attr.shape)
        if self.concat_states:
            x = torch.cat(concat_x, 1)
        x = self.node_post_mlp(x)
        # self.check_input(x,edge_attr,edge_index)
        return x

    def check_input(self, xs, edge_attr, edge_index):
        Os = {}
        for indx in range(128):
            i=edge_index[0,indx].detach().numpy()
            j=edge_index[1,indx].detach().numpy()
            xi=xs[i].detach().numpy()
            xj=list(xs[j].detach().numpy())
            eij=list(edge_attr[indx].detach().numpy())
            if str(i) not in Os.keys():
                Os[str(i)] = {'x_j':[],'e_ij':[]}
            Os[str(i)]['x_i'] = xi
            Os[str(i)]['x_j'] += xj
            Os[str(i)]['e_ij'] += eij

        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(1,3,1)
        for i in Os.keys():
            plt.plot(Os[str(i)]['x_i'],label=str(i))
            plt.title('x_i')
        plt.legend()
        plt.subplot(1,3,2)
        for i in Os.keys():
            plt.plot(Os[str(i)]['e_ij'],label=str(i))
            plt.title('e_ij')
        plt.legend()
        plt.subplot(1,3,3)
        for i in Os.keys():
            plt.plot(Os[str(i)]['x_j'],label=str(i))
            plt.title('x_j')
        plt.legend()
        plt.show()
