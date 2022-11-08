import torch.nn as nn
import sys
import pdb
from param import args

QUERY_LENGTH=16 #Number of words in the query, without counting [CLS] abd [SEP]
ANSWER_LENGTH=4 #Number of output classes

from param import args
from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU
from DecExp_data import WORDSET_ROOT
import torch


class DecExpModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mode='x'
        with open(WORDSET_ROOT, 'r') as f:
            loaded=f.read()
            self.len_wordset=len(loaded.split(" "))
            
        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(args, max_seq_length=QUERY_LENGTH+2, mode=self.mode)
        self.hid_dim = self.lxrt_encoder.dim
        
        #Build NLP decoder
        self.decoderlayer=nn.TransformerDecoderLayer(768, 12, 2048, 0.1)
        self.decoder=nn.TransformerDecoder(self.decoderlayer, args.num_decoderlayers, torch.nn.LayerNorm)
        self.decoder=self.decoder.cuda()
        self.postDecLin=nn.Linear(self.hid_dim, self.len_wordset)
        
        # VQA Answer heads
        if self.mode=='x':
            self.logit_fc = nn.Sequential(
                nn.Linear(self.hid_dim, self.hid_dim * 2),
                GeLU(),
                BertLayerNorm(self.hid_dim * 2, eps=1e-12),
                #nn.Sigmoid() Not included there for numerical stability in the computation of the loss with BCEWithLogitsLoss
                                #To include it, change the loss to a standard BCELoss
                nn.Linear(self.hid_dim * 2, ANSWER_LENGTH)
            )
            #self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)


        # elif 'x' in self.mode and ('l' in self.mode or 'r' in self.mode):
        #     self.logit_fc = nn.Sequential(
        #         nn.Linear((100+QUERY_LENGTH+2)*self.hid_dim, self.hid_dim * 2),
        #         GeLU(),
        #         BertLayerNorm(self.hid_dim * 2, eps=1e-12),
        #         #nn.Sigmoid() Not included there for numerical stability in the computation of the loss with BCEWithLogitsLoss
        #                         #To include it, change the loss to a standard BCELoss
        #         nn.Linear(self.hid_dim * 2, ANSWER_LENGTH)
        #     )
            #self.actionpredict_head=nn.Linear(self.hid_dim * 2, 4)
            #self.exppredict_head=nn.Linear(self.hid_dim * 2, ANSWER_LENGTH-4)
        #self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)
        
    def forward(self, imgid, feat, pos, sent, tgt=None):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        if tgt is not None:
            
            
            
            assert len(tgt.size())==2 or len(tgt.size())==3, 'size of tgt not handled'
        
        
        if self.mode=='x':
            x = self.lxrt_encoder(imgid, sent, (feat, pos))
            if tgt is not None: #If the target sentence is sent to the model
                if len(tgt.size())==3:
                    attn_mask=torch.triu(torch.ones([tgt[1], tgt[1]]), 1)
                elif len(tgt.size())==2:
                    attn_mask=torch.triu(torch.ones([tgt[0], tgt[0]]), 1)
                
                decoder_out = self.decoder(tgt=tgt, memory=x, tgt_mask=attn_mask)
                out = nn.Softmax(-1)(self.postDecLin(decoder_out))
            
            
            else: #Let the model iteratively generate the sentence
                out = self.logit_fc(x) #Does this return logits or label(logits followed by sigmoid) ? If latter -> Change loss in DecExp.py to BCELoss

                

        
        
        # elif 'x' in self.mode and ('l' in self.mode or 'r' in self.mode):
        #     featseq, x = self.lxrt_encoder(imgid, sent, (feat, pos))
        #     logit=self.logit_fc(torch.cat((featseq[0].reshape([args.batch_size, (QUERY_LENGTH+2)*self.hid_dim]), \
        #         featseq[1].reshape([args.batch_size, 100*self.hid_dim])), axis=1))
        #actionlogit = self.actionpredict_head(prelogit)
        #explogit= self.exppredict_head(prelogit)
        #logit=torch.cat((actionlogit, explogit), 1)

        return out