import torch
from model1 import *
import util
from ranger21 import Ranger



class trainer():
    def __init__(self, num_nodes, seq_len, channels, dropout, lrate, wdecay, device, pre_adj,mean, std, adj_mx, dtw_mx,
                 num_of_weeks, num_of_days, num_of_hours,adj):
        self.model =ISGCN(device, num_nodes, seq_len, channels,
                             dropout, adj_mx, dtw_mx, num_of_weeks, num_of_days, num_of_hours,adj, pre_adj=pre_adj)
        self.model.to(device)
        self.optimizer = Ranger(self.model.parameters(),
                                lr=lrate, weight_decay=wdecay)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.clip = 5
        self.mean = mean
        self.std = std
        self.dtw_mx = dtw_mx
        self.adj_mx = adj_mx
        self.seq_len = seq_len
        #self.criterion = torch.nn.SmoothL1Loss()

    #def train(self, input1, real_val):
    def train(self,  input3, real_val):

        dict = { 'input3': input3}
        self.model.train()
        self.optimizer.zero_grad()

        output,dadj = self.model(dict)
        output=output.transpose(1, 3)# 64,1,170,12
        predict = output * self.std + self.mean

        real = torch.unsqueeze(real_val, dim=1)  # （64, 170，12）---（64 1 170 12）

        loss = self.loss(predict, real, 0.0)
       # loss = self.criterion(predict, real)

        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mape, rmse,dadj

    def eval(self,  input3, real_val):
        self.model.eval()
        dict = { 'input3': input3}
        output,dadj = self.model(dict)
        output=output.transpose(1, 3)
        predict = output * self.std + self.mean
        real = torch.unsqueeze(real_val, dim=1)
        loss = self.loss(predict, real, 0.0)
        #loss = self.criterion(predict, real)
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mape, rmse
