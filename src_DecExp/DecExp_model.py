import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import pdb

QUERY_LENGTH=13 #Number of words in the query, without counting [CLS] and [SEP]
SUBTASKS_ANSWER_LENGTH=1 #Number of output classes for subtasks
DEC_ANSWER_LENGTH=4 #Number of output classes for the Decision task

from param import args
from utils import fill_tensor_1D
from lxrt.entry import LXRTEncoder, convert_sents_to_features
from lxrt.modeling import BertLayerNorm, GeLU, BertEmbeddings


class DecExpModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mode='x'
        
        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(args, max_seq_length=QUERY_LENGTH+2, mode=self.mode)
        self.hid_dim = self.lxrt_encoder.dim
        
        # VQA Answer heads
        if self.mode=='x':
            self.logit_fc = nn.Sequential(
                nn.Linear(self.hid_dim, self.hid_dim * 2),
                GeLU(),
                BertLayerNorm(self.hid_dim * 2, eps=1e-12),
                #nn.Sigmoid() Not included there for numerical stability in the computation of the loss with BCEWithLogitsLoss
                                #To include it, change the loss to a standard BCELoss
                nn.Linear(self.hid_dim * 2, SUBTASKS_ANSWER_LENGTH)
            )
            #self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)
            
            
            self.Dec_fc = nn.Sequential(
                nn.Linear(self.hid_dim, self.hid_dim * 2),
                GeLU(),
                BertLayerNorm(self.hid_dim * 2, eps=1e-12),
                #nn.Sigmoid() Not included there for numerical stability in the computation of the loss with BCEWithLogitsLoss
                                #To include it, change the loss to a standard BCELoss
                nn.Linear(self.hid_dim * 2, DEC_ANSWER_LENGTH)
            )


        # elif 'x' in self.mode and ('l' in self.mode or 'r' in self.mode):
        #     self.logit_fc = nn.Sequential(
        #         nn.Linear((100+QUERY_LENGTH+2)*self.hid_dim, self.hid_dim * 2),
        #         GeLU(),
        #         BertLayerNorm(self.hid_dim * 2, eps=1e-12),
        #         #nn.Sigmoid() Not included there for numerical stability in the computation of the loss with BCEWithLogitsLoss
        #                         #To include it, change the loss to a standard BCELoss
        #         nn.Linear(self.hid_dim * 2, SUBTASKS_ANSWER_LENGTH)
        #     )
            #self.actionpredict_head=nn.Linear(self.hid_dim * 2, 4)
            #self.exppredict_head=nn.Linear(self.hid_dim * 2, SUBTASKS_ANSWER_LENGTH-4)
        #self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)
        
    def forward(self, imgid, feat, pos, sent, labels=None):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        
        if self.mode=='x':
            x = self.lxrt_encoder(imgid, sent, (feat, pos))
            out=(self.Dec_fc(x), self.logit_fc(x))



        # elif 'x' in self.mode and ('l' in self.mode or 'r' in self.mode):
        #     featseq, x = self.lxrt_encoder(imgid, sent, (feat, pos))
        #     logit=self.logit_fc(torch.cat((featseq[0].reshape([args.batch_size, (QUERY_LENGTH+2)*self.hid_dim]), \
        #         featseq[1].reshape([args.batch_size, 100*self.hid_dim])), axis=1))
        #actionlogit = self.actionpredict_head(prelogit)
        #explogit= self.exppredict_head(prelogit)
        #logit=torch.cat((actionlogit, explogit), 1)

        return out