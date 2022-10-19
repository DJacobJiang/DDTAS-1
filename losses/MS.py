from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

def sigmoid(x):
    # 直接返回sigmoid函数
    return 1. / (1. + np.exp(-x))
class MSLoss(nn.Module):
    def __init__(self, alpha=10, beta=2, margin=0.5, hard_mining=1, **kwargs):
        super(MSLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.hard_mining = hard_mining

    def forward(self, inputs, targets, margin):
        n = inputs.size(0)
        # print('input_shape:', np.shape(inputs))

        sim_mat = torch.matmul(inputs, inputs.t())
        # print('smt', sim_mat)
        #targets = targets

        base = 0.7
        loss = list()
        # loss_list = list()
        c = 0
        length_ap = 0
        length_an = 0
        length_AP = 0
        length_ap_t = 0
        length_an_t = 0




        for i in range(n):

            pos_pair_ = torch.masked_select(sim_mat[i], targets==targets[i])
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < (1-(1e-6)))
            neg_pair_ = torch.masked_select(sim_mat[i], targets!=targets[i])

            pos_pair_ = torch.sort(pos_pair_)[0]
            # print('maxpos',pos_pair_[-1])
            neg_pair_ = torch.sort(neg_pair_)[0]
            length_AP += len(pos_pair_)
            # print(1,len(neg_pair_), len(pos_pair_))


            if self.hard_mining is not None:

                neg_pair = torch.relu(neg_pair_ + (margin/10) - pos_pair_[0])
                pos_pair = torch.relu(neg_pair_[-1] - pos_pair_ + margin)

                neg_pair = torch.masked_select(neg_pair, neg_pair > 0) -  (margin//10) + pos_pair_[0]
                pos_pair = -torch.masked_select(pos_pair, pos_pair > 0) +  margin+ neg_pair_[-1]

                if len(neg_pair) < 1 or len(pos_pair) < 1:
                    c += 1
                    continue

                length_ap += len(pos_pair)
                length_an += len(neg_pair)

                #Bio
                pos_loss = 2.0/self.beta * torch.mean(torch.log(1 + torch.exp(-self.beta*(pos_pair - base))))
                neg_loss = 2.0/self.alpha * torch.mean(torch.log(1 + torch.exp(self.alpha*(neg_pair - base))))
                # pos_loss = torch.mean(torch.log(1 + torch.exp(-self.beta * (pos_pair - base))))
                # neg_loss = torch.mean(torch.log(1 + torch.exp(self.alpha * (neg_pair - base))))
                loss.append(neg_loss + pos_loss)


            else:
                # print('hello world')
                neg_pair = neg_pair_

                pos_pair = pos_pair_

                length_ap += len(pos_pair)
                length_an += len(neg_pair)

                # pos_loss = 2.0/self.beta * torch.log(1 + torch.sum(torch.exp(-self.beta * (pos_pair - base ))))
                # neg_loss = 2.0/self.alpha * torch.log(1 + torch.sum(torch.exp(self.alpha * (neg_pair - base ))))

                # Bio
                pos_loss = 2.0/self.beta * torch.mean(torch.log(1 + torch.exp(-self.beta*(pos_pair - base))))
                neg_loss = 2.0/self.alpha * torch.mean(torch.log(1 + torch.exp(self.alpha*(neg_pair - base))))

                loss.append(neg_loss + pos_loss)
        loss = sum(loss)/n
        # loss = sum(loss)/2
        print('lap', length_ap, '||', 'lan', length_an)
        prec = float(c)/n
        mean_neg_sim = torch.mean(neg_pair_).item()
        mean_pos_sim = torch.mean(pos_pair_).item()
        print([(length_an/2)/(length_ap/2), sigmoid((length_an/2)/(length_ap/2))], [(length_an/2)/(length_AP/2),sigmoid((length_an/2)/(length_AP/2))])
        return loss, prec, mean_pos_sim, mean_neg_sim, length_ap/2, length_an/2, [(length_an/2)/(length_ap/2), sigmoid((length_an/2)/(length_ap/2))], [(length_an/2)/(length_AP/2),sigmoid((length_an/2)/(length_AP/2))]

def main():
    data_size = 32
    input_dim = 3
    output_dim = 2
    num_class = 4
    # margin = 0.5
    x = Variable(torch.rand(data_size, input_dim), requires_grad=False)
    # print(x)
    w = Variable(torch.rand(input_dim, output_dim), requires_grad=True)
    inputs = x.mm(w)
    y_ = 8*list(range(num_class))
    targets = Variable(torch.IntTensor(y_))

    print(WeightLoss()(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')


