import torch
import torch.nn as nn
from torch.nn import Conv2d, Parameter
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import weight_norm
from torch_geometric.nn import SAGEConv, GATConv, JumpingKnowledge
import math


# class Spatial_Attention_layer(nn.Module):
#     def __int__(self):
#         super(Spatial_Attention_layer,self).__int__()
#         self.dropout = nn.Dropout(p=0)
#
#     def forward(self, x1, x2):
#         batch_size, in_channels, num_of_vertices, num_of_timesteps = x1.shape
#
#         x1 = x1.permute(0,3,2,1).reshape((-1, num_of_vertices,in_channels))
#         x2 = x2.permute(0, 3, 2, 1).reshape((-1, num_of_vertices, in_channels))
#         score = torch.matmul(x1, x2.transpose(1,2))/math.sqrt(in_channels)
#         score = self.dropout(F.softmax(score,dim=-1))
#         return score.reshape((batch_size, num_of_timesteps, num_of_vertices, num_of_vertices))


class s(nn.Module):
    def __init__(self, dropout=.0):
        super(s,self).__init__()

        self.alpha = nn.Parameter(torch.Tensor([0.0]),requires_grad=True)
        self.dropout = nn.Dropout(p=0)


    def forward(self, x1, x2):

        batch_size, in_channels, num_of_vertices, num_of_timesteps = x1.shape
        x1 = x1.permute(0, 3, 2, 1).reshape((-1, num_of_vertices, in_channels))
        x2 = x2.permute(0, 3, 2, 1).reshape((-1, num_of_vertices, in_channels))
        score = torch.matmul(x1, x2.transpose(1, 2)) / math.sqrt(in_channels)
        score = F.softmax(score, dim=-1).reshape((batch_size, num_of_timesteps, num_of_vertices, num_of_vertices))
        spatial_attention = score.reshape((-1, num_of_vertices, num_of_vertices))
        x_gcn = torch.matmul(spatial_attention, x1)
        return F.relu(x_gcn.reshape((batch_size, in_channels, num_of_vertices, num_of_timesteps)))


class Diffusion_GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=2, order=1):
        super().__init__()
        c_in = (order * support_len + 1) * c_in
        self.conv = Conv2d(c_in, c_out, (1, 1), padding=(
            0, 0), stride=(1, 1), bias=True)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support: list):
        out = [x]
        for a in support:
            x1 = torch.einsum('bcnt,nm->bcmt', (x, a)).contiguous()
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = torch.einsum('bcnt,nm->bcmt', (x1, a)).contiguous()
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.conv(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

def nconv(input, graph, power):
    list = []
    for i in range(power):
        output = torch.einsum('bcnt,nm->bcmt', (input, graph)).contiguous()
        list.append(output)
        input = output
    output1 = torch.cat(list, dim=1)
    return output1


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class learned_GCN(nn.Module):
    def __init__(self, node_num, in_feature, out_feature):
        super(learned_GCN, self).__init__()
        self.node_num = node_num
        self.in_feature = in_feature
        self.out_feature = out_feature

        self.source_embed = nn.Parameter(torch.Tensor(self.node_num, 10))
        self.target_embed = nn.Parameter(torch.Tensor(10, self.node_num))
        self.linear = nn.Linear(self.in_feature, self.out_feature)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.source_embed.size(0))
        self.source_embed.data.uniform_(-stdv, stdv)
        self.target_embed.data.uniform_(-stdv, stdv)

    def forward(self, input):
        learned_matrix = F.softmax(F.relu(torch.mm(self.source_embed, self.target_embed)), dim=1)
        output = learned_matrix.matmul(input)
        output = self.linear(output)
        return output


class STCell(nn.Module):
    def __init__(self, num_of_x, node_num=170, seq_len=12, graph_dim=16,
                 tcn_dim=[10], choice=[1,1,1], atten_head=2,pred_len=6):
        super(STCell, self).__init__()
        self.node_num = node_num
        self.seq = seq_len
        self.num_of_x = num_of_x
        self.seq_len = seq_len
        self.graph_dim = graph_dim
        self.tcn_dim = tcn_dim
        self.output_dim = (np.sum(choice)-1) * graph_dim  # TODO： 要改
        self.choice = choice
        self.pred_len = pred_len

        # self.jklayer = JumpingKnowledge("max")
        # self.jklayer = JumpingKnowledge("lstm", self.graph_dim, 1)
        # self.seq_linear = nn.Linear(in_features=self.seq_len*self.num_of_x, out_features=self.seq_len*self.num_of_x)
        self.seq_linear = nn.Linear(in_features=self.seq_len * self.num_of_x, out_features=self.seq_len * self.num_of_x)
        self.output_linear = nn.Linear(in_features=32, out_features=self.seq_len*self.num_of_x)


        # if choice[0] == 1:
        #     print(f"[TCN]")
        #     self.self_atten = nn.MultiheadAttention(embed_dim=node_num+1 if node_num%2==1 else node_num, num_heads=atten_head,bias=False)
        #     self.extra_linear1 = nn.Linear(self.node_num, self.node_num+1)
        #     self.extra_linear2 = nn.Linear(self.node_num+1, self.node_num)
        #     self.tcn = TemporalConvNet(num_inputs=1, num_channels=self.tcn_dim)
        #     self.tlinear = nn.Linear(in_features=self.tcn_dim[-1] * self.seq_len*self.num_of_x, out_features=self.graph_dim)

        if choice[1] == 1:
            print(f"[SP]")
            self.sp_origin = nn.Linear(in_features=seq_len*num_of_x, out_features=graph_dim)
            self.sp_gconv1 = GATConv(seq_len*num_of_x, graph_dim, heads=3, concat=False)
            self.sp_gconv2 = GATConv(graph_dim, graph_dim, heads=3, concat=False)
            self.sp_gconv3 = GATConv(graph_dim, graph_dim, heads=3, concat=False)
            self.sp_gconv4 = GATConv(graph_dim, graph_dim, heads=1, concat=False)
            self.sp_gconv5 = GATConv(graph_dim, graph_dim, heads = 1, concat = False)
            self.sp_source_embed = nn.Parameter(torch.Tensor(self.node_num, 12))
            self.sp_target_embed = nn.Parameter(torch.Tensor(12, self.node_num))
            self.sp_linear_1 = nn.Linear(self.seq_len*num_of_x, self.graph_dim)
            self.sp_linear_2 = nn.Linear(self.graph_dim, self.graph_dim)
            self.sp_linear_3 = nn.Linear(self.graph_dim, self.graph_dim)
            self.sp_linear_4 = nn.Linear(self.graph_dim, self.graph_dim)
            # self.w = nn.Parameter(torch.rand((1),requires_grad=True)).to('cuda')
            self.final_sp_linear = nn.Linear(self.graph_dim * 4, self.graph_dim)
            # self.sp_linear_5 = nn.Linear(self.graph_dim, self.graph_dim)
            # self.sp_jklayer = JumpingKnowledge("max")

            nn.init.xavier_uniform_(self.sp_source_embed)
            nn.init.xavier_uniform_(self.sp_target_embed)

            if choice[2] == 1:
                print(f"[DTW]")
                self.dtw_origin = nn.Linear(in_features=seq_len*num_of_x, out_features=graph_dim)
                self.dtw_gconv1 = GATConv(seq_len*num_of_x, graph_dim, heads=3, concat=False)
                self.dtw_gconv2 = GATConv(graph_dim, graph_dim, heads=3, concat=False)
                self.dtw_gconv3 = GATConv(graph_dim, graph_dim, heads=3, concat=False)
                self.dtw_gconv4 = GATConv(graph_dim, graph_dim, heads=3, concat=False)
                self.dtw_gconv5 = GATConv(graph_dim, graph_dim, heads = 1, concat = False)
                self.dtw_source_embed = nn.Parameter(torch.Tensor(self.node_num, 12))
                self.dtw_target_embed = nn.Parameter(torch.Tensor(12, self.node_num))
                self.dtw_linear_1 = nn.Linear(self.seq_len*num_of_x, self.graph_dim)
                self.dtw_linear_2 = nn.Linear(self.graph_dim, self.graph_dim)
                self.dtw_linear_3 = nn.Linear(self.graph_dim, self.graph_dim)
                self.dtw_linear_4 = nn.Linear(self.graph_dim, self.graph_dim)
                # self.v = nn.Parameter(torch.rand((1), requires_grad=True)).to('cuda')
                # self.dtw_linear_5 = nn.Linear(self.graph_dim, self.graph_dim)
                self.final_dtw_linear = nn.Linear(self.graph_dim*4, self.graph_dim)


                nn.init.xavier_uniform_(self.dtw_source_embed)
                nn.init.xavier_uniform_(self.dtw_target_embed)

    def forward(self, x, edge_index, dtw_edge_index,adj):
        # x shape is [batch*node_num, seq_len]
        # tcn/dtw/sp/adaptive output shape is [batch, node_num, graph_dim]
        output_list = [0, 0, 0]

        # if self.choice[0] == 1:
        #     atten_input = torch.reshape(x, (-1, self.node_num, self.seq_len*self.num_of_x)).permute(2, 0, 1)  # 6,32,170
        #     if self.node_num % 2 == 1:
        #         atten_input1 = self.extra_linear1(atten_input)
        #         atten_output, _ = self.self_atten(atten_input1, atten_input1, atten_input1)  # 12,64,170
        #         atten_output = self.extra_linear2(atten_output)
        #     else:
        #         atten_output, _ = self.self_atten(atten_input, atten_input, atten_input)  # 6,32,170
        #     atten_output = torch.tanh(atten_output + atten_input)  # 6,32,170
        #     atten_output = torch.reshape(atten_output.permute(1, 2, 0), (-1, self.seq_len*self.num_of_x))  # 32*170, 6
        #
        #     tcn_input = atten_output.unsqueeze(1)  # 32*170,1,6
        #     tcn_output = self.tcn(tcn_input)  # 32*170,10,6
        #     tcn_output = torch.reshape(tcn_output, (tcn_output.shape[0], self.tcn_dim[-1] * self.seq_len*self.num_of_x))  # 32*170,60
        #     tcn_output = self.tlinear(tcn_output)  # 32*170,16
        #     tcn_output = torch.reshape(tcn_output, (-1, self.node_num, self.graph_dim))  # 32,170,16
        #     output_list[0] = tcn_output

        if self.choice[1] == 1:
            x = self.seq_linear(x) + x

            sp_learned_matrix =F.softmax(F.relu(torch.mm(self.sp_source_embed, self.sp_target_embed)+torch.eye(self.node_num).to('cuda')), dim=1)  # 170,170
            # sp_learned_matrix = self.w*sp_learned_matrix+(1-self.w)*adj

            sp_gout_1 = self.sp_gconv1(x, edge_index)  # GAT 10880,16
            adp_input_1 = torch.reshape(x, (-1, self.node_num, self.seq_len*self.num_of_x))  # 64,170,12
            sp_adp_1 = self.sp_linear_1(sp_learned_matrix.matmul(F.dropout(adp_input_1, p=0.1)))  # 门控值 64,170,16
            sp_adp_1 = torch.reshape(sp_adp_1, (-1, self.graph_dim))  # 10880,16
            sp_origin = self.sp_origin(x)  # 10880,16
            sp_output_1 = torch.tanh(sp_gout_1) * torch.sigmoid(sp_adp_1) + sp_origin * (1 - torch.sigmoid(sp_adp_1))  # 10880,16


            sp_gout_2 = self.sp_gconv2(torch.tanh(sp_output_1), edge_index)
            adp_input_2 = torch.reshape(torch.tanh(sp_output_1), (-1, self.node_num, self.graph_dim))
            sp_adp_2 = self.sp_linear_2(sp_learned_matrix.matmul(F.dropout(adp_input_2, p=0.1)))
            sp_adp_2 = torch.reshape(sp_adp_2, (-1, self.graph_dim))
            sp_output_2 = F.leaky_relu(sp_gout_2) * torch.sigmoid(sp_adp_2) + sp_output_1 * (
                        1 - torch.sigmoid(sp_adp_2))

            sp_gout_3 = self.sp_gconv3(F.relu(sp_output_2), edge_index)
            adp_input_3 = torch.reshape(F.relu(sp_output_2), (-1, self.node_num, self.graph_dim))
            sp_adp_3 = self.sp_linear_3(sp_learned_matrix.matmul(F.dropout(adp_input_3, p=0.1)))
            sp_adp_3 = torch.reshape(sp_adp_3, (-1, self.graph_dim))
            sp_output_3 = F.relu(sp_gout_3) * torch.sigmoid(sp_adp_3) + sp_output_2 * (1 - torch.sigmoid(sp_adp_3))

            sp_gout_4 = self.sp_gconv4(F.relu(sp_output_3), edge_index)
            adp_input_4 = torch.reshape(F.relu(sp_output_3), (-1, self.node_num, self.graph_dim))
            sp_adp_4 = self.sp_linear_4(sp_learned_matrix.matmul(F.dropout(adp_input_4, p=0.1)))
            sp_adp_4 = torch.reshape(sp_adp_4, (-1, self.graph_dim))
            sp_output_4 = F.relu(sp_gout_4) * torch.sigmoid(sp_adp_4) + sp_output_3 * (1 - torch.sigmoid(sp_adp_4))  # 10800,16


            # sp_gout_5 = self.sp_gconv5(F.relu(sp_output_4), edge_index)
            # adp_input_5 = torch.reshape(F.relu(sp_output_4), (-1, self.node_num, self.graph_dim))
            # sp_adp_5 = self.sp_linear_5(sp_learned_matrix.matmul(F.dropout(adp_input_5,p=0.1)))
            # sp_adp_5 = torch.reshape(sp_adp_5, (-1, self.graph_dim))
            # sp_output_5 = F.relu(sp_gout_5) * torch.sigmoid(sp_adp_5) + sp_output_4 * (1 - torch.sigmoid(sp_adp_5))
            out = torch.cat((sp_output_1,sp_output_2,sp_output_3,sp_output_4),dim=1)
            sp_output_4 = self.final_sp_linear(out)
            sp_output = torch.reshape(sp_output_4, (-1, self.node_num, self.graph_dim))  # 64,170,16

            output_list[1] = sp_output

            if self.choice[2] == 1:
                x = self.seq_linear(x) + x  # 10800,12

                dtw_learned_matrix = F.softmax(F.relu(torch.mm(self.dtw_source_embed, self.dtw_target_embed)+torch.eye(self.node_num).to('cuda')), dim=1)  # 170,170

                dtw_gout_1 = self.dtw_gconv1(x, dtw_edge_index)  # GAT 10800,16
                adp_input_1 = torch.reshape(x, (-1, self.node_num, self.seq_len*self.num_of_x))  # 64,170,12
                dtw_adp_1 = self.dtw_linear_1(dtw_learned_matrix.matmul(F.dropout(adp_input_1, p=0.1)))  # 64,170,16
                dtw_adp_1 = torch.reshape(dtw_adp_1, (-1, self.graph_dim))  # 10800,16
                dtw_origin = self.dtw_origin(x)  # 10800,16
                dtw_output_1 = torch.tanh(dtw_gout_1) * torch.sigmoid(dtw_adp_1) + dtw_origin * (
                            1 - torch.sigmoid(dtw_adp_1))  # 10800,16

                dtw_gout_2 = self.dtw_gconv2(torch.tanh(dtw_output_1), dtw_edge_index)
                adp_input_2 = torch.reshape(torch.tanh(dtw_output_1), (-1, self.node_num, self.graph_dim))
                dtw_adp_2 = self.dtw_linear_2(dtw_learned_matrix.matmul(F.dropout(adp_input_2, p=0.1)))
                dtw_adp_2 = torch.reshape(dtw_adp_2, (-1, self.graph_dim))
                dtw_output_2 = F.leaky_relu(dtw_gout_2) * torch.sigmoid(dtw_adp_2) + dtw_output_1 * (
                            1 - torch.sigmoid(dtw_adp_2))

                dtw_gout_3 = self.dtw_gconv3(F.relu(dtw_output_2), dtw_edge_index)
                adp_input_3 = torch.reshape(F.relu(dtw_output_2), (-1, self.node_num, self.graph_dim))
                dtw_adp_3 = self.dtw_linear_3(dtw_learned_matrix.matmul(F.dropout(adp_input_3, p=0.1)))
                dtw_adp_3 = torch.reshape(dtw_adp_3, (-1, self.graph_dim))
                dtw_output_3 = F.relu(dtw_gout_3) * torch.sigmoid(dtw_adp_3) + dtw_output_2 * (
                           1 - torch.sigmoid(dtw_adp_3))

                dtw_gout_4 = self.dtw_gconv4(F.relu(dtw_output_3), dtw_edge_index)
                adp_input_4 = torch.reshape(F.relu(dtw_output_3), (-1, self.node_num, self.graph_dim))
                dtw_adp_4 = self.dtw_linear_4(dtw_learned_matrix.matmul(F.dropout(adp_input_4, p=0.1)))
                dtw_adp_4 = torch.reshape(dtw_adp_4, (-1, self.graph_dim))
                dtw_output_4 = F.relu(dtw_gout_4) * torch.sigmoid(dtw_adp_4) + dtw_output_3 * (
                             1 - torch.sigmoid(dtw_adp_4))

                # dtw_gout_5 = self.dtw_gconv5(F.relu(dtw_output_4), dtw_edge_index)
                # adp_input_5 = torch.reshape(F.relu(dtw_output_4), (-1, self.node_num, self.graph_dim))
                # dtw_adp_5 = self.dtw_linear_5(dtw_learned_matrix.matmul(F.dropout(adp_input_5,p=0.1)))
                # dtw_adp_5 = torch.reshape(dtw_adp_5, (-1, self.graph_dim))
                # dtw_output_5 = F.relu(dtw_gout_5) * torch.sigmoid(dtw_adp_5) + dtw_output_4 * (1 - torch.sigmoid(dtw_adp_5))

                out1 = torch.cat((dtw_output_1, dtw_output_2, dtw_output_3, dtw_output_4), dim=1)
                dtw_output_4 = self.final_sp_linear(out1)
                dtw_output = torch.reshape(dtw_output_4, (-1, self.node_num, self.graph_dim))  # 64,170,16
                # dtw_output = dtw_output_4
                output_list[2] = dtw_output

            step = 0
            for i in range(1, len(self.choice)):
                if self.choice[i] == 1 and step == 0:
                    cell_output = output_list[i]
                    step += 1
                elif self.choice[i] == 1:
                    cell_output = torch.cat((cell_output, output_list[i]), dim=2)  # 32, 170, 48

            # cell_output = self.jklayer([output_list[0], output_list[1], output_list[2]])
            # cell_output = self.out(cell_output)

            cell_output = torch.reshape(cell_output, (-1, self.output_dim))  # (32*170,48)
            output = self.output_linear(cell_output).reshape(-1,self.node_num,self.seq_len*self.num_of_x).unsqueeze(1)  # 10800,12

            return output


class D_GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, power, support_len=2, order=1):
        super().__init__()
        c_in = c_in * power
        self.conv = Conv2d(c_in, c_out, (1, 1), padding=(
            0, 0), stride=(1, 1), bias=True)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support: list, power):  # (32, 64, 170, 6)
        for a in support:
            x1 = nconv(x, a, power)
        h = self.conv(x1)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


# class Diffusion_GCN(nn.Module):
#     def __init__(self, c_in, c_out, dropout, support_len=2, order=1):
#         super().__init__()
#         c_in = (order * support_len + 1) * c_in
#         self.conv = Conv2d(c_in, c_out, (1, 1), padding=(
#             0, 0), stride=(1, 1), bias=True)
#         self.dropout = dropout
#         self.order = order
#
#     def forward(self, x, support: list):  # (32, 64, 170, 6)
#         out = [x]
#         for a in support:  # 对于列表每个图
#             x1 = nconv(x, a)  # 输入和图进行爱因斯坦求和 （32， 64， 170， 6）
#             out.append(x1)
#             for k in range(2, self.order + 1):
#                 x2 = nconv(x1, a)
#                 out.append(x2)
#                 x1 = x2
#         h = torch.cat(out, dim=1)  # （64， 64*4， 170， 12）
#         h = self.conv(h)  # 64,64,170,12
#         h = F.dropout(h, self.dropout, training=self.training)
#         return h
    
    
def sample_gumbel(device, shape, eps=1e-20):
    U = torch.rand(shape).to(device)
    return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))


def gumbel_softmax_sample(device, logits, temperature,  eps=1e-10):
    sample = sample_gumbel(device, logits.size(), eps=eps)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(device, logits, temperature, hard=False, eps=1e-10):
    y_soft = gumbel_softmax_sample(
        device, logits, temperature=temperature, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        y_hard = torch.zeros(*shape).to(device)
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


class Graph_Generator(nn.Module):
    def __init__(self, device, channels, num_nodes, seq_len, num_of_x, adj, dtw,adj_mx, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.node = num_nodes
        self.device = device
        self.adj = adj
        self.dtw = dtw
        self.fc0 = nn.Linear(channels, num_nodes)
        self.fc1 = nn.Linear(num_nodes, 2*num_nodes)
        self.fc2 = nn.Linear(2*num_nodes, num_nodes)
        self.STCell = STCell(num_of_x=num_of_x, node_num=num_nodes,seq_len=int(seq_len/2), graph_dim=16, choice=[1, 1, 1],
                             atten_head=2, pred_len=12)
        self.start_conv = nn.Conv2d(in_channels=1,
                                    out_channels=channels,
                                    kernel_size=(1, 1))
        self.adj_mx = torch.Tensor(adj_mx[0])

    # def forward(self, x, adj):  # x: (32, 64, 170, 6)   x1: (32,1,170,6)   adj: (170, 170)
    def forward(self, x1, adj):  # x: (32, 64, 170, 6)   x1: (32,1,170,6)   adj: (170, 170)
        # x = self.diffusion_conv(x, [adj])  # (32, 64, 170, 6)
        input = x1
        input = input.reshape(input.size(0) * input.size(2), -1)
        output = self.STCell(input, edge_index=self.adj.to('cuda'), dtw_edge_index=self.dtw.to('cuda'), adj = self.adj_mx.to('cuda'))  # 32,1,170,6
        output1 = self.start_conv(output)
        x = output1[-1,:,:,:]
        x = x.sum(2)
        x = x.permute(1, 0)  # (170, 64)
        x = self.fc0(x)  # (170, 170)
        x = torch.tanh(x)
        x = self.fc1(x)  # (170, 340)
        x = torch.tanh(x)
        x = self.fc2(x)  # (170, 170)
        x = torch.tanh(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.log(F.softmax(x, dim=-1))
        x = gumbel_softmax(self.device, x, temperature=0.5, hard=True)  # (170, 170)
        mask = torch.eye(x.shape[0], x.shape[0]).bool().to(device=self.device)
        x.masked_fill_(mask, 0)
        return x, output






class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()

    def even(self, x):
        return x[:, :, :, ::2]

    def odd(self, x):
        return x[:, :, :, 1::2]

    def forward(self, x):
        return (self.even(x), self.odd(x))


class IDGCN(nn.Module):
    def __init__(self, device, channels, seq_len, num_of_x, adj, dtw, adj_mx,splitting=True,
                 num_nodes=170, dropout=0.25, pre_adj=None, pre_adj_len=1
                 ):
        super(IDGCN, self).__init__()

        device = device
        self.dropout = dropout
        self.pre_adj_len = pre_adj_len
        self.num_nodes = num_nodes
        self.splitting = splitting
        self.pre_graph = pre_adj or []
        self.split = Splitting()
        self.seq_len = seq_len
        self.num_of_x = num_of_x


        Conv1 = []
        Conv2 = []
        Conv3 = []
        Conv4 = []
        pad_l = 3
        pad_r = 3

        apt_size = 10
        aptinit = pre_adj[0]
        self.pre_adj_len = 1
        nodevecs = self.svd_init(apt_size, aptinit)
        self.nodevec1, self.nodevec2 = [
            Parameter(n.to(device), requires_grad=True) for n in nodevecs]

        Conv1 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 5), dilation=1, stride=1, groups=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 3), dilation=1, stride=1, groups=1),
            nn.Tanh(),
        ]
        Conv2 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 5), dilation=1, stride=1, groups=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 3), dilation=1, stride=1, groups=1),
            nn.Tanh()
        ]

        Conv4 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 5), dilation=1, stride=1, groups=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 3), dilation=1, stride=1, groups=1),
            nn.Tanh()
        ]
        Conv3 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 5), dilation=1, stride=1, groups=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 3), dilation=1, stride=1, groups=1),
            nn.Tanh()
        ]

        self.conv1 = nn.Sequential(*Conv1)
        self.conv2 = nn.Sequential(*Conv2)
        self.conv3 = nn.Sequential(*Conv3)
        self.conv4 = nn.Sequential(*Conv4)

        self.a = nn.Parameter(torch.rand(1).to(
            device=device), requires_grad=True).to(device)

        self.graph_generator = Graph_Generator(
            device, channels, num_nodes, seq_len, num_of_x, adj, dtw,adj_mx)

        self.diffusion_conv1 = D_GCN(
            channels, channels, dropout, support_len=self.pre_adj_len, power=3)

        self.diffusion_conv2 = D_GCN(
            channels, channels, dropout, support_len=self.pre_adj_len, power=3)

        self.diffusion_conv3 = D_GCN(
            channels, channels, dropout, support_len=self.pre_adj_len, power=3)

        self.diffusion_conv4 = D_GCN(
            channels, channels, dropout, support_len=self.pre_adj_len, power=3)
        self.sat = s()


    @staticmethod
    def svd_init(apt_size, aptinit):  # 求E1和E2
        m, p, n = torch.svd(aptinit)
        nodevec1 = torch.mm(m[:, :apt_size], torch.diag(p[:apt_size] ** 0.5))
        nodevec2 = torch.mm(torch.diag(
            p[:apt_size] ** 0.5), n[:, :apt_size].t())
        return nodevec1, nodevec2

    def gaussin_kernel(self,a,b,sigma):
        a = a.view(a.shape[0], -1)
        b = b.view(b.shape[0], -1)
        distance = torch.norm(a-b,dim=1)** 2/(a.shape[1])
        similarity = torch.exp(-distance/(2*sigma**2))
        return similarity.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


    def forward(self, x, x_1):  # (32, 64, 170, 12)
        if self.splitting:
            (x_even, x_odd) = self.split(x)  # (32, 64, 170, 6)
            (x_1even, x_1odd) = self.split(x_1)  # (32,1,170,6)
        else:
            (x_even, x_odd) = x
            (x_1even, x_1odd) = x_1

        adaptive_adj = F.softmax(  # (170, 170)
            F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        #             x
        # xeven               xodd
        #         x1      x2
        #     c      dgcn     d
        #         x3      x4
        # xevenup             xoddup

        x1 = self.conv1(x_even)  # (32, 64, 170, 6)
        learn_adj, output1 = self.graph_generator(x_1even, adaptive_adj)
        x1 = x1+self.diffusion_conv1(x1, [learn_adj], 3)  # (32, 64, 170, 6)

        # dot = torch.matmul(x1, x_odd.transpose(-2,-1))/math.sqrt(32)
        # softmax = nn.Softmax(dim=-1)
        # attention = softmax(dot)
        # d= torch.matmul(attention, x1)

        # x1=self.sat(x1,x_odd)

        # x1 = self.gaussin_kernel(x1,x_odd,1)

        d = x_odd.mul(torch.tanh(x1))  # (32, 64, 170, 6)



        x2 = self.conv2(x_odd)
        learn_adj, output2 = self.graph_generator(x_1odd, adaptive_adj)
        x2 = x2+self.diffusion_conv2(x2, [learn_adj], 3)

        # dot = torch.matmul(x2, x_even.transpose(-2, -1))/math.sqrt(32)
        # softmax = nn.Softmax(dim=-1)
        # attention = softmax(dot)
        # c= torch.matmul(attention, x2)
        # x2 = self.gaussin_kernel(x2, x_even, 1)

        # x2= self.sat(x2, x_even)

        c = x_even.mul(torch.tanh(x2))  # (32, 64, 170, 6)



        x3 = self.conv3(d)
        learn_adj, output3 = self.graph_generator(output1, adaptive_adj)
        x3 = x3+self.diffusion_conv3(x3, [learn_adj], 3)

        # dot = torch.matmul(x3, c.transpose(-2, -1))/math.sqrt(32)
        # softmax = nn.Softmax(dim=-1)
        # attention = softmax(dot)
        # x_odd_update= torch.matmul(attention,x3)

        # x3 = self.gaussin_kernel(x3, d, 1)

        # x3= self.sat(x3,d)

        x_odd_update = c + x3 # Either "+" or "-" here does not have much effect on the results. (32, 64, 170, 6)



        x4 = self.conv4(c)
        learn_adj, output4 = self.graph_generator(output2, adaptive_adj)
        x4 = x4+self.diffusion_conv4(x4, [learn_adj], 3)

        # dot = torch.matmul(x4, d.transpose(-2, -1))/math.sqrt(32)
        # softmax = nn.Softmax(dim=-1)
        # attention = softmax(dot)
        # x_even_update= torch.matmul(attention, x4)

        # x4 = self.gaussin_kernel(x4, c, 1)

        # x4= self.sat(x4, c)

        x_even_update = d + x4 # Either "+" or "-" here does not have much effect on the results.  (32, 64, 170, 6)
        output = torch.cat((output3, output4),dim=-1)

        return (x_even_update, x_odd_update, learn_adj, output)


class IDGCN_Tree(nn.Module):
    def __init__(self, device, num_nodes, channels, adj, dtw, num_levels, dropout, seq_len, num_of_x,adj_mx, pre_adj=None, pre_adj_len=1):
        super().__init__()
        self.levels = num_levels
        self.pre_graph = pre_adj or []

        self.IDGCN1 = IDGCN(splitting=True, channels=channels, seq_len=seq_len, num_of_x=num_of_x,device=device,
                            pre_adj=pre_adj, num_nodes=num_nodes, dropout=dropout, pre_adj_len=pre_adj_len,
                            adj=adj,dtw=dtw,adj_mx=adj_mx)
        self.IDGCN2 = IDGCN(splitting=True, channels=channels, seq_len=seq_len, num_of_x=num_of_x,device=device,
                            pre_adj=pre_adj, num_nodes=num_nodes, dropout=dropout, pre_adj_len=pre_adj_len,
                            adj=adj,dtw=dtw,adj_mx=adj_mx)
        self.IDGCN3 = IDGCN(splitting=True, channels=channels, seq_len=seq_len, num_of_x=num_of_x,device=device,
                            pre_adj=pre_adj, num_nodes=num_nodes, dropout=dropout, pre_adj_len=pre_adj_len,
                            adj=adj,dtw=dtw,adj_mx=adj_mx)

        self.a = nn.Parameter(torch.rand(1).to(
            device=device), requires_grad=True).to(device)
        self.b = nn.Parameter(torch.rand(1).to(
            device=device), requires_grad=True).to(device)
        self.c = nn.Parameter(torch.rand(1).to(
            device=device), requires_grad=True).to(device)

    def concat(self, even, odd):
        even = even.permute(3, 1, 2, 0)
        odd = odd.permute(3, 1, 2, 0)
        len = even.shape[0]
        _ = []
        for i in range(len):
            _.append(even[i].unsqueeze(0))
            _.append(odd[i].unsqueeze(0))
        return torch.cat(_, 0).permute(3, 1, 2, 0)

    def forward(self, x, x1):
        x_even_update1, x_odd_update1, dadj1, x2 = self.IDGCN1(x, x1)
        x_even_update2, x_odd_update2, dadj2, x3 = self.IDGCN2(x_even_update1, x2)
        x_even_update3, x_odd_update3, dadj3, x4 = self.IDGCN3(x_odd_update1, x3)
        concat1 = self.concat(x_even_update2, x_odd_update2)
        concat2 = self.concat(x_even_update3, x_odd_update3)
        concat0 = self.concat(concat1, concat2)  # （32，64，170 12）
        adj = dadj1*self.a+dadj2*self.b+dadj3*self.c
        return concat0, adj


class STIDGCN1(nn.Module):
    def __init__(self, device, num_nodes, seq_len, num_of_weeks, channels, adj, dtw,adj_mx,dropout=0.25, pre_adj=None):
        super().__init__()

        self.dropout = dropout
        self.num_nodes = num_nodes
        self.input_len = 12
        self.output_len = 12
        self.num_levels = 2
        self.groups = 1
        input_channel = 1
        apt_size = 10
        self.seq_len = seq_len
        self.dtw = dtw
        self.adj = adj
        self.pre_graph = pre_adj or []
        self.pre_adj_len = len(self.pre_graph)+1
        self.num_of_weeks = num_of_weeks

        aptinit = pre_adj[0]
        nodevecs = self.svd_init(apt_size, aptinit)
        self.nodevec1, self.nodevec2 = [
            Parameter(n.to(device), requires_grad=True) for n in nodevecs]

        self.a = nn.Parameter(torch.rand(1).to(
            device=device), requires_grad=True).to(device)

        self.start_conv = nn.Conv2d(in_channels=input_channel,
                                    out_channels=channels,
                                    kernel_size=(1, 1))

        self.tree = IDGCN_Tree(
            device=device,
            channels=channels,
            num_nodes=self.num_nodes,
            num_levels=self.num_levels,
            dropout=self.dropout,
            pre_adj_len=self.pre_adj_len,
            pre_adj=pre_adj,
            seq_len=seq_len,
            num_of_x=num_of_weeks,
            adj=adj,
            dtw=dtw,
            adj_mx=adj_mx
        )



        self.diffusion_conv = Diffusion_GCN(
            channels, channels, dropout, support_len=self.pre_adj_len)

        self.Conv1 = nn.Conv2d(in_channels=channels,
                               out_channels=256,
                               kernel_size=(1, 1), stride=(1, 1))
        self.Conv2 = nn.Conv2d(in_channels=256,
                               out_channels=512,
                               kernel_size=(1, 12*num_of_weeks), stride=(1, 1))
        self.Conv3 = nn.Conv2d(in_channels=512,
                               out_channels=12,
                               kernel_size=(1, 1), stride=(1, 1))
        self.W = nn.Parameter(torch.empty(12, self.num_nodes))
        nn.init.xavier_uniform(self.W)

    @staticmethod
    def svd_init(apt_size, aptinit):
        m, p, n = torch.svd(aptinit)
        nodevec1 = torch.mm(m[:, :apt_size], torch.diag(p[:apt_size] ** 0.5))
        nodevec2 = torch.mm(torch.diag(
            p[:apt_size] ** 0.5), n[:, :apt_size].t())
        return nodevec1, nodevec2

    def forward(self, input):
        x1 = input

        x = self.start_conv(x1)  # (64,64,170,36)

        # a = torch.mm(self.nodevec1, self.nodevec2)
        # adaptive_adj = F.softmax(F.relu(a), dim=1)

        skip = x
        x, adj = self.tree(x, x1)  # 生成交互学习图Alearn （170 170）和输出（64, 64, 170, 36）
        x = skip + x  # （64, 64, 170, 12）

        # adj = self.a*adaptive_adj+(1-self.a)*dadj  #生成Adyn
        adj = self.pre_graph + [adj]

        gcn = self.diffusion_conv(x, adj)  # (64, 64, 170, 12)

        x = gcn + x  # (64, 64, 170, 12)

        # x = F.relu(self.Conv1(x))  # （64,256,170,12）
        # x = F.relu(self.Conv2(x))  # (64,512,170,1)
        # x = self.Conv3(x).squeeze()  # （64,12,170,1）
        #x = x*self.W
        # return x.unsqueeze(-1)
        return x


class STIDGCN2(nn.Module):
    def __init__(self, device, num_nodes, seq_len, num_of_days, channels, adj,dtw,adj_mx, dropout=0.25, pre_adj=None):
        super().__init__()

        self.dropout = dropout
        self.num_nodes = num_nodes
        self.input_len = 12
        self.output_len = 12
        self.num_levels = 2
        self.groups = 1
        input_channel = 1
        apt_size = 10
        self.seq_len = seq_len
        self.dtw = dtw
        self.adj = adj
        self.pre_graph = pre_adj or []
        self.pre_adj_len = len(self.pre_graph)+1
        self.num_of_days = num_of_days

        aptinit = pre_adj[0]
        nodevecs = self.svd_init(apt_size, aptinit)
        self.nodevec1, self.nodevec2 = [
            Parameter(n.to(device), requires_grad=True) for n in nodevecs]

        self.a = nn.Parameter(torch.rand(1).to(
            device=device), requires_grad=True).to(device)

        self.start_conv = nn.Conv2d(in_channels=input_channel,
                                    out_channels=channels,
                                    kernel_size=(1, 1))

        self.tree = IDGCN_Tree(
            device=device,
            channels=channels,
            num_nodes=self.num_nodes,
            num_levels=self.num_levels,
            dropout=self.dropout,
            pre_adj_len=self.pre_adj_len,
            pre_adj=pre_adj,
            seq_len=seq_len,
            num_of_x=num_of_days,
            adj=adj,
            dtw=dtw,
            adj_mx=adj_mx
        )

        self.diffusion_conv = Diffusion_GCN(
            channels, channels, dropout, support_len=self.pre_adj_len)

        self.Conv1 = nn.Conv2d(in_channels=channels,
                               out_channels=256,
                               kernel_size=(1, 1), stride=(1, 1))
        self.Conv2 = nn.Conv2d(in_channels=256,
                               out_channels=512,
                               kernel_size=(1, 12*self.num_of_days), stride=(1, 1))
        self.Conv3 = nn.Conv2d(in_channels=512,
                               out_channels=12,
                               kernel_size=(1, 1), stride=(1, 1))
        self.W = nn.Parameter(torch.empty(12, self.num_nodes))
        nn.init.xavier_uniform(self.W)

    @staticmethod
    def svd_init(apt_size, aptinit):
        m, p, n = torch.svd(aptinit)
        nodevec1 = torch.mm(m[:, :apt_size], torch.diag(p[:apt_size] ** 0.5))
        nodevec2 = torch.mm(torch.diag(
            p[:apt_size] ** 0.5), n[:, :apt_size].t())
        return nodevec1, nodevec2

    def forward(self, input):
        x1 = input

        x = self.start_conv(x1)  # (64,64,170,36)

        # a = torch.mm(self.nodevec1, self.nodevec2)
        # adaptive_adj = F.softmax(F.relu(a), dim=1)
        # adaptive_adj = F.softmax(  # 生成自适应图Aapt
        #     F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)

        skip = x
        x, adj = self.tree(x, x1)  # 生成交互学习图Alearn （170 170）和输出（64, 64, 170, 36）
        x = skip + x  # （32, 64, 170, 12）

        # adj = self.a*adaptive_adj+(1-self.a)*dadj  #生成Adyn
        adj = self.pre_graph + [adj]

        gcn = self.diffusion_conv(x, adj)  # (32, 64, 170, 12)

        x = gcn + x

        # x = F.relu(self.Conv1(x))
        # x = F.relu(self.Conv2(x))
        # x = self.Conv3(x).squeeze()
        #x = x*self.W
        # return x.unsqueeze(-1)
        return x

class STIDGCN3(nn.Module):
    def __init__(self, device, num_nodes, seq_len, num_of_hours, channels, adj,dtw,adj_mx, dropout=0.25, pre_adj=None):
        super().__init__()

        self.dropout = dropout
        self.num_nodes = num_nodes
        self.input_len = 12
        self.output_len = 12
        self.num_levels = 2
        self.groups = 1
        input_channel = 1
        apt_size = 10
        self.seq_len = seq_len*num_of_hours
        self.dtw = dtw
        self.adj = adj
        self.pre_graph = pre_adj or []
        self.pre_adj_len = len(self.pre_graph)+1
        self.adj_mx = adj_mx

        aptinit = pre_adj[0]
        nodevecs = self.svd_init(apt_size, aptinit)
        self.nodevec1, self.nodevec2 = [
            Parameter(n.to(device), requires_grad=True) for n in nodevecs]

        self.a = nn.Parameter(torch.rand(1).to(
            device=device), requires_grad=True).to(device)

        self.start_conv = nn.Conv2d(in_channels=input_channel,
                                    out_channels=channels,
                                    kernel_size=(1, 1))

        self.tree = IDGCN_Tree(
            device=device,
            channels=channels,
            num_nodes=self.num_nodes,
            num_levels=self.num_levels,
            dropout=self.dropout,
            pre_adj_len=self.pre_adj_len,
            pre_adj=pre_adj,
            seq_len=seq_len,
            num_of_x=num_of_hours,
            adj=adj,
            dtw=dtw,
            adj_mx=adj_mx
        )

        self.diffusion_conv = Diffusion_GCN(
            channels, channels, dropout, support_len=self.pre_adj_len)

        self.Conv1 = nn.Conv2d(in_channels=channels,
                               out_channels=256,
                               kernel_size=(1, 1), stride=(1, 1))
        self.Conv2 = nn.Conv2d(in_channels=256,
                               out_channels=512,
                               kernel_size=(1, 12*num_of_hours), stride=(1, 1))
        self.Conv3 = nn.Conv2d(in_channels=512,
                               out_channels=12,
                               kernel_size=(1, 1), stride=(1, 1))
        self.W = nn.Parameter(torch.empty(12, self.num_nodes))
        nn.init.xavier_uniform(self.W)

    @staticmethod
    def svd_init(apt_size, aptinit):
        m, p, n = torch.svd(aptinit)
        nodevec1 = torch.mm(m[:, :apt_size], torch.diag(p[:apt_size] ** 0.5))
        nodevec2 = torch.mm(torch.diag(
            p[:apt_size] ** 0.5), n[:, :apt_size].t())
        return nodevec1, nodevec2

    def forward(self, input):
        x1 = input

        x = self.start_conv(x1)  # (64,64,170,36)

        # a = torch.mm(self.nodevec1, self.nodevec2)
        # adaptive_adj = F.softmax(F.relu(a), dim=1)
        # adaptive_adj = F.softmax(  # 生成自适应图Aapt
        #     F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)

        skip = x
        x, adj = self.tree(x, x1)  # 生成交互学习图Alearn （170 170）和输出（64, 64, 170, 36）
        x = skip + x  # （32, 64, 170, 12）

        # adj1 = self.a*adaptive_adj+(1-self.a)*dadj  #生成Adyn
        adj = self.pre_graph + [adj]

        gcn = self.diffusion_conv(x, adj)  # (64, 64, 170, 36)

        x = gcn + x

        # x = F.relu(self.Conv1(x))
        # x = F.relu(self.Conv2(x))
        #
        # x = self.Conv3(x).squeeze()
        #x = x*self.W
        # return x.unsqueeze(-1), adj1
        return x,adj

class ISGCN(nn.Module):
    def __init__(self, device, num_nodes, seq_len,channels, dropout, adj,dtw, num_of_weeks, num_of_days,num_of_hours, adj_mx,pre_adj=None):
        super().__init__()

        self.dropout = dropout
        self.num_nodes = num_nodes
        self.input_len = 12
        self.output_len = 12
        self.num_levels = 2
        self.groups = 1
        self.channels = channels
        self.adj = adj_mx


        self.pre_graph = pre_adj or []
        self.pre_adj_len = len(self.pre_graph)+1
        # self.linear1 = nn.Linear(36,64)
        # self.linear2 = nn.Linear(64,12)
        # self.relu1 = nn.ReLU()
        # self.relu2 = nn.ReLU()
        self.Conv1 = nn.Conv2d(in_channels=channels*(num_of_hours),
                               out_channels=256,
                               kernel_size=(1, 1), stride=(1, 1))
        self.Conv2 = nn.Conv2d(in_channels=256,
                               out_channels=512,
                               kernel_size=(1, 12 ), stride=(1, 1))
        self.Conv3 = nn.Conv2d(in_channels=512,
                               out_channels=12,
                               kernel_size=(1, 1), stride=(1, 1))

        # self.STIDGCN1 = STIDGCN1(device=device, num_nodes=num_nodes, seq_len=seq_len, num_of_weeks=num_of_weeks,
        #                        channels=channels, dropout=dropout, pre_adj=pre_adj, adj=adj,dtw=dtw,adj_mx=self.adj)
        # self.STIDGCN2 = STIDGCN2(device=device, num_nodes=num_nodes, seq_len=seq_len, num_of_days=num_of_days,
        #                  channels=channels, dropout=dropout, pre_adj=pre_adj,adj=adj,dtw=dtw,adj_mx=self.adj)
        self.STIDGCN3 = STIDGCN3(device=device, num_nodes=num_nodes, seq_len=seq_len, num_of_hours=num_of_hours,
                                 channels=channels, dropout=dropout, pre_adj=pre_adj,adj=adj,dtw=dtw,adj_mx=self.adj)

    def forward(self, input):
        # x1 = self.STIDGCN1(input['input1'])
        # x2 = self.STIDGCN2(input['input2'])
        x3,dadj = self.STIDGCN3(input['input3'])
        # x3_1 = x3[:,:,:,:12]
        # x3_2 = x3[:,:, :,12:24]  # x3_2 = x3[:,:, :,12:]
        # x3_3 = x3[:,:,:,24:36]
       # x3_4=x3[:,:,:,36:]
       #  x = torch.cat((x1,x2,x3),dim=1) # x = torch.cat((x1,x2,x3_1,x3_2),dim=1)
       #  x = torch.cat(( x2,x3_1, x3_2,x3_3), dim=1)
        x = F.relu(self.Conv1(x3))
        x = F.relu(self.Conv2(x))

        x = self.Conv3(x).squeeze()
        # x = torch.cat((x1,x2,x3), dim=2).transpose(2,1)
        # x = self.relu1(self.linear1(x.reshape(-1,36)))
        # x= self.relu2(self.linear2(x)).reshape(x2.shape[0], x2.shape[1], x2.shape[2])
        #
        # return x.unsqueeze(-1)
        return x.unsqueeze(-1),dadj
        # x = (x3+x1+x2)/3
        # return x,dadj