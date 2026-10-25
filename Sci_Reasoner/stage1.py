import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.utils import to_dense_batch

from Sci_Reasoner.transformer import Decoder, Embeddings, MultiHeadedAttention, PositionwiseFeedForward, DecoderLayer


class GAT(nn.Module):
    def __init__(self,heads=4,neurons=64,dropout=0.5):
        super(GAT, self).__init__()

        n_h, n_hX2          = neurons, neurons*2
        self.neurons        = neurons
        self.neg_slope      = 0.2

        self.embed_n        = Linear(92,n_h)
        self.embed_e        = Linear(41,n_h)
        self.embed_comp     = Linear(103,n_h)

        self.conv1 = TransformerConv(n_h, 16, heads=heads, concat=True, edge_dim=n_h)
        self.conv2 = TransformerConv(64, 16, 4, concat=True, edge_dim=n_h)
        self.conv4 = TransformerConv(64, n_h, 4, concat=False, edge_dim=n_h)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(n_h)

        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self,data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        batch, global_feat, cluster = data.batch, data.global_feature, data.cluster

        x = F.leaky_relu(self.embed_n(x)) 

        edge_attr = F.leaky_relu(self.embed_e(edge_attr), self.neg_slope)
        global_feat = F.leaky_relu(self.embed_comp(global_feat), self.neg_slope)


        # run GNN to update node embedding
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.conv4(x, edge_index, edge_attr)
        x = self.bn4(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        y = global_mean_pool(x, batch)
        z = torch.cat([y, global_feat], dim=-1)
        return z, x, batch


class Structure_Aware_Spectrum_Decoding(nn.Module):
    def __init__(self, args):
        super(Structure_Aware_Spectrum_Decoding, self).__init__()
        n_heads = args.num_heads
        number_neurons = args.d_model
        number_layers = args.num_layers
        self.graph_encoder = GAT(n_heads, neurons=number_neurons, dropout=args.graph_dropout)

        self.label_dim = args.label_dim # 51

        self.dec_layers = args.dec_layers
        self.dec_dropout = args.dec_dropout
        self.bs = args.batch_size
        self.d_model = args.d_model
        self.h = args.h
        self.d_ff = args.d_ff
        self.args = args

        c = copy.deepcopy
        attn = MultiHeadedAttention(self.h, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dec_dropout)
        self.tgt_embed = Embeddings(self.d_model, self.label_dim)
        self.fc = Linear(number_neurons*2+self.d_model, self.d_model)
        self.decoder1 = Decoder(DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dec_dropout), self.dec_layers)
        
        self.decoder2 = Decoder(DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dec_dropout), self.dec_layers)
        
        self.generator = Linear(self.d_model, 1)

        self.final_activate = nn.Softplus()

        # XANES is element-specific: condition the crystal context on the
        # absorbing element. Only instantiate the embedding for XANES targets
        # so non-XANES checkpoints stay parameter-compatible with the original.
        self.use_absorber_emb = str(getattr(args, "data_src", "")).startswith("xanes")
        if self.use_absorber_emb:
            # crystal_ctx is concat(graph mean pool, composition feat) = 2*d_model.
            self.absorber_emb = nn.Embedding(120, 2 * self.d_model)
            nn.init.zeros_(self.absorber_emb.weight)

    def encode(self, data):
        crystal_ctx, nodes, batch = self.graph_encoder(data)
        if self.use_absorber_emb and getattr(data, "absorber_z", None) is not None:
            z = data.absorber_z.view(-1).long().clamp(min=0, max=119)
            crystal_ctx = crystal_ctx + self.absorber_emb(z)
        return crystal_ctx, nodes, batch

    def forward(self, data):
        label = data.y
        bs = label.shape[0]
        seq_len = self.label_dim
        crystal_ctx, nodes, batch = self.encode(data)
        
        nodes, nd_mask = to_dense_batch(nodes, batch)
        pos = torch.arange(self.label_dim, device=crystal_ctx.device).repeat(bs, 1)
        tgt_embs = self.tgt_embed(pos)
        dec_input = torch.cat((tgt_embs, crystal_ctx.unsqueeze(1).repeat(1, seq_len, 1)), dim=-1)
        dec_input = F.leaky_relu(self.fc(dec_input))

        out = self.decoder1(dec_input, nodes, nd_mask.unsqueeze(1), None) 
        out = self.decoder2(out, nodes, nd_mask.unsqueeze(1), None)
        out = self.generator(out).squeeze(-1) 
        out = self.final_activate(out)

        return out


def compute_loss(input_label, input_label_sum, fx_out, args):
    row_sum = torch.sum(fx_out, dim=1, keepdim=True)
    pred = fx_out / row_sum
    input_label_normalize = input_label / torch.sum(input_label, dim=1, keepdim=True)


    log_target = torch.log(input_label_normalize+1e-10)
    log_pred = torch.log(pred+1e-10)
    label_weights = F.softmax(log_target*args.temp)
    loss = torch.sum(label_weights*(log_target-log_pred),dim=1)
    
    loss_lst = loss
    total_loss = torch.mean(loss_lst)

    return total_loss, pred
            
