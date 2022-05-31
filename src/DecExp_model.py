import torch.nn as nn
import sys

#QUERY_LENGTH=105 #number of characters
QUERY_LENGTH=16 #number of words in the input query
ANSWER_LENGTH=25 #Number of output classes

from param import args
from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU


class DecExpModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(args, max_seq_length=QUERY_LENGTH+2)
        hid_dim = self.lxrt_encoder.dim
        
        # VQA Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, ANSWER_LENGTH),
            #nn.Sigmoid() Not included there for numerical stability in the computation of the loss with BCEWithLogitsLoss
                            #To include it, change the loss to a standard BCELoss
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)
        
    def forward(self, imgid, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        x = self.lxrt_encoder(imgid, sent, (feat, pos))
        logit = self.logit_fc(x) #Does this return logits or label(logits followed by sigmoid) ? If latter -> Change loss in DecExp.py to BCELoss

        return logit