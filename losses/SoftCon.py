from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

def sigmoid(x):
    return 1. / (1. + np.exp(-x))
class SoftContrastiveLoss(nn.Module):
    def __init__(self, alpha=10, beta=2, margin=0.5, hard_mining=1, **kwargs):
        super(SoftContrastiveLoss, self).__init__()
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
        length_AP_t = 0
        length_ap_t = 0
        length_an_t = 0
        # global length_ap_t
        # global length_an_t
        #
        for i in range(n):
            pos_pair_ = torch.masked_select(sim_mat[i], targets == targets[i])

            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < (1 - (1e-6)))

            neg_pair_ = torch.masked_select(sim_mat[i], targets != targets[i])

            pos_pair_ = torch.sort(pos_pair_)[0]
            # print('maxpos',pos_pair_[-1])
            neg_pair_ = torch.sort(neg_pair_)[0]
            length_AP_t += len(pos_pair_)
            # print(1,len(neg_pair_), len(pos_pair_))
            if self.hard_mining is not None:

                neg_pair = torch.relu(neg_pair_ + (margin / 1) - pos_pair_[0])
                pos_pair = torch.relu(neg_pair_[-1] - pos_pair_ + margin)

                neg_pair = torch.masked_select(neg_pair, neg_pair > 0) - (margin / 1) + pos_pair_[0]
                pos_pair = -torch.masked_select(pos_pair, pos_pair > 0) + margin + neg_pair_[-1]

                # print('AAAA', pos_pair_[0], neg_pair_[-1])

                # print(2,len(neg_pair_),len(pos_pair_))
                # sigmoid((length_an / 2) / (length_ap / 2))
                # sigmoid((length_an / 2) / (length_AP / 2))
                # if len(neg_pair) < 1 or len(pos_pair) < 1:
                #     # print(len(neg_pair),len(pos_pair))
                #     # print('aaaaaaaaaaaaaaa')
                #     c += 1
                #     continue
                length_ap_t += len(pos_pair)
                length_an_t += len(neg_pair)
        #Alg 3-5
        if (length_an_t / length_AP_t) > 1:
            print("Alg 3-5")
            tol_tre = (sigmoid(length_an_t / length_AP_t))/2
            # print(tol_tre,'ap_n',margin+(2*tol_tre*margin),'an_n',(margin/10)-(tol_tre*(margin/10)))
            margin_ap_t = margin + (tol_tre * margin)
            margin_an_t = (margin / 10) - (tol_tre * (margin / 10))
        else:
            print("Alg 6-7")
            margin_ap_t = margin
            margin_an_t = margin/10




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
                # print(neg_pair_)
                ################################ New Margin (Alg.9)################################
                neg_pair = torch.relu(neg_pair_ + (margin_an_t) - pos_pair_[0])
                pos_pair = torch.relu(neg_pair_[-1] - pos_pair_ + margin_ap_t)

                neg_pair = torch.masked_select(neg_pair, neg_pair > 0) -  (margin_an_t) + pos_pair_[0]
                pos_pair = -torch.masked_select(pos_pair, pos_pair > 0) +  margin_ap_t+ neg_pair_[-1]

                neg_pair_j = torch.relu(neg_pair_ + (margin/10) - pos_pair_[0])
                pos_pair_j = torch.relu(neg_pair_[-1] - pos_pair_ + margin)

                # neg_pair_j = torch.masked_select(neg_pair_j, neg_pair_j > 0) - (margin/10) + pos_pair_[0]
                # pos_pair_j = -torch.masked_select(pos_pair_j, pos_pair_j > 0) + margin + neg_pair_[-1]
                ######################################################################################
                ############################Original mining strategy##################################
                # neg_pair = torch.relu(neg_pair_ + (margin/10) - pos_pair_[0])
                # pos_pair = torch.relu(neg_pair_[-1] - pos_pair_ + margin)
                #
                # neg_pair = torch.masked_select(neg_pair, neg_pair > 0) - (margin/10) + pos_pair_[0]
                # pos_pair = -torch.masked_select(pos_pair, pos_pair > 0) + margin + neg_pair_[-1]
                #####################################################################################
                # print('AAAA', pos_pair_[0], neg_pair_[-1])
                # if len(neg_pair) < 1:
                #     if (len(neg_pair) - len(neg_pair)) > 0:
                #         neg_pair = neg_pair

                # pos_pair = pos_pair_

                # print(2,len(neg_pair_),len(pos_pair_))
                # sigmoid((length_an / 2) / (length_ap / 2))
                # sigmoid((length_an / 2) / (length_AP / 2))


                # print('AAAA',length_ap)

                if len(neg_pair) < 1 or len(pos_pair) < 1:
                    # print(len(neg_pair_j)-len(neg_pair),len(neg_pair),len(pos_pair))
                    # print('aaaaaaaaaaaaaaa')
                    c += 1
                    continue

                length_ap += len(pos_pair)
                length_an += len(neg_pair)
                
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
        # print('lap', length_ap, '||', 'lan', length_an,'||','lapt','||','laptt', length_ap_t, '||', 'lann', length_an_t,
        #       '||','lapt','||',length_ap-length_ap_t, '||', 'lant',length_an- length_an_t)
        print('lap', length_ap, '||', 'lan', length_an)
        prec = float(c)/n
        mean_neg_sim = torch.mean(neg_pair_).item()
        mean_pos_sim = torch.mean(pos_pair_).item()
        ln = length_an/2
        lp=length_ap/2
        Lp = length_AP/2
        # lnt=length_an_t/2
        # lpt=length_ap_t/2
        # Lpt=length_AP_t/2
        lnt = length_an / 2
        lpt = length_ap / 2
        Lpt = length_AP / 2
        print([(length_an / 2) / (length_ap / 2), sigmoid((length_an / 2) / (length_ap / 2))],
              [(length_an / 2) / (length_AP / 2), sigmoid((length_an / 2) / (length_AP / 2))])
        return (loss, prec, mean_pos_sim, mean_neg_sim,lp, ln,[ln/lp,sigmoid(ln/lp),ln/Lp,sigmoid(ln/Lp),lp/Lp],
               [lnt/lpt,sigmoid(lnt/lpt),lnt/Lpt,sigmoid(lnt/Lpt),lpt/Lpt])

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


