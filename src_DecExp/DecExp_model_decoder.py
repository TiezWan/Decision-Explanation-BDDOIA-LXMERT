import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import pdb

MAX_TEXT_ANSWER_LENGTH=50
QUERY_LENGTH=16 #Number of words in the query, without counting [CLS] and [SEP]
LOGIT_ANSWER_LENGTH=4 #Number of output classes

from param import args
from utils import fill_tensor_1D
from lxrt.entry import LXRTEncoder, convert_sents_to_features
from lxrt.modeling import BertLayerNorm, GeLU, BertEmbeddings
from DecExp_data import WORDSET_ROOT


class DecExpModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mode='x'
        with open(WORDSET_ROOT, 'r') as f:
            loaded=f.read()
            wordlist=loaded.split(" ")
            self.vocab_size=len(wordlist)
        self.word2index={wordlist[i]:i for i in range(len(wordlist))}
        self.index2word = {v: k for k, v in self.word2index.items()}
        
        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(args, max_seq_length=QUERY_LENGTH+2, mode=self.mode)
        self.hid_dim = self.lxrt_encoder.dim
        
        # Build NLP decoder
        self.decoderlayer=nn.TransformerDecoderLayer(768, 12, 2048, 0.1)
        self.decoder=nn.TransformerDecoder(self.decoderlayer, args.num_decoderlayers)
        self.decoder=self.decoder.cuda()
        
        self.MLPHead = nn.Linear(self.hid_dim, self.vocab_size, bias=False)
        self.position_embeddings = nn.Linear(MAX_TEXT_ANSWER_LENGTH, self.hid_dim, bias=False)
        #self.token_type_embeddings = nn.Embedding(2, self.hid_dim, padding_idx=0)        
        #self.MLPHead=nn.Linear(self.hid_dim, self.vocab_size, bias=False)
        
        # if getattr(self.MLPHead, "bias", None) is not None:
        #     self.MLPHead.bias.data = nn.functional.pad(
        #         self.MLPHead.bias.data,
        #         (0, self.MLPHead.weight.shape[0] - self.MLPHead.bias.shape[0],),
        #         "constant",
        #         0,
        #     )
        # if hasattr(self.MLPHead, "out_features") and hasattr(self.word_embeddings, "num_embeddings"):
        #     self.MLPHead.out_features = self.word_embeddings.num_embeddings
        
        #self.x=torch.rand([1, 2, 768], dtype=torch.float32).cuda()
        
        # VQA Answer heads
        if self.mode=='x':
            self.logit_fc = nn.Sequential(
                nn.Linear(self.hid_dim, self.hid_dim * 2),
                GeLU(),
                BertLayerNorm(self.hid_dim * 2, eps=1e-12),
                #nn.Sigmoid() Not included there for numerical stability in the computation of the loss with BCEWithLogitsLoss
                                #To include it, change the loss to a standard BCELoss
                nn.Linear(self.hid_dim * 2, LOGIT_ANSWER_LENGTH)
            )
            #self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)


        # elif 'x' in self.mode and ('l' in self.mode or 'r' in self.mode):
        #     self.logit_fc = nn.Sequential(
        #         nn.Linear((100+QUERY_LENGTH+2)*self.hid_dim, self.hid_dim * 2),
        #         GeLU(),
        #         BertLayerNorm(self.hid_dim * 2, eps=1e-12),
        #         #nn.Sigmoid() Not included there for numerical stability in the computation of the loss with BCEWithLogitsLoss
        #                         #To include it, change the loss to a standard BCELoss
        #         nn.Linear(self.hid_dim * 2, LOGIT_ANSWER_LENGTH)
        #     )
            #self.actionpredict_head=nn.Linear(self.hid_dim * 2, 4)
            #self.exppredict_head=nn.Linear(self.hid_dim * 2, LOGIT_ANSWER_LENGTH-4)
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
        tgt=None #Initialization
        
        #print(f"MLPHead weights: {self.MLPHead.weight}")
        
        if labels is not None:
            self.ans_len=max([len(labels[i].split(" ")) for i in range(args.batch_size)])
            
        ## Tokenization and embeddings
            #Tokenization
            try:
                input_ids_list=[[fill_tensor_1D(torch.zeros(self.vocab_size, dtype=torch.float), self.word2index[word], 1) for word in labels[i].split(" ")] for i in range(args.batch_size)]
                input_ids_integers=[[self.word2index[word] for word in labels[i].split(" ")] for i in range(args.batch_size)]
            except:
                print(labels)
            
            #Padding to the same length
            for ids in input_ids_list:
                ids.extend([fill_tensor_1D(torch.zeros(self.vocab_size, dtype=torch.float), self.word2index["[PAD]"], 1)]*(self.ans_len-len(ids)))
            
            input_ids=torch.stack([torch.stack(input_ids_list[i]) for i in range(args.batch_size)]).cuda()
            
            # Setting Decisions and Explanations to groups A and B
            # features_type_ids = []
            # for i in range(len(input_ids_list)):
            #     sep_index = input_ids_list[i].index(self.word2index["#"])
            #     temp = [0]*(sep_index)#0 for Decision, 1 for Explanation
            #     temp.extend([1]*(len(input_ids_list[i])-(sep_index)))
            #     features_type_ids.append(temp)
                
            # Embeddings
            #position_ids = torch.arange(self.ans_len, dtype=torch.float, device=input_ids.device)
            position_ids = torch.triu(torch.tril(torch.ones([args.batch_size, self.ans_len, MAX_TEXT_ANSWER_LENGTH]))).cuda()
            
            words_embeddings = F.linear(input_ids, self.MLPHead.weight.T)
            position_embeddings = self.position_embeddings(position_ids)
            #token_type_embeddings = self.token_type_embeddings(token_type_ids)

            tgt = words_embeddings + position_embeddings# + token_type_embeddings
            #features_type_ids=torch.tensor(features_type_ids, dtype=torch.int).cuda()
            #tgt = self.lxrt_encoder.model.bert.embeddings(input_ids)
            
            tgt = tgt.transpose(0,1)
            
        
        if self.mode=='x':
            x = self.lxrt_encoder(imgid, sent, (feat, pos))
            x = x.unsqueeze(0)
            if tgt is not None: #If the target sentence is sent to the model
                attn_mask=torch.triu(torch.ones([self.ans_len, self.ans_len], dtype=torch.bool), 1).cuda()
                decoder_out = self.decoder(tgt=tgt, memory=x, tgt_mask=attn_mask)
                decoder_out=decoder_out.transpose(0,1)
                out = self.MLPHead(decoder_out)
                #logits = self.logit_fc(x) #Does this return logits or label(logits followed by sigmoid) ? If latter -> Change loss in DecExp.py to BCELoss
                #print(f"out.size= {out.size()}")
                
            else: #Let the model iteratively generate the sentence
                sent_decoder=['<SOS>' for i in range(args.batch_size)]
                flagEOS=[False for i in range(args.batch_size)]
                while sum(flagEOS)<args.batch_size and max([len(sentence.split(" ")) for sentence in sent_decoder])<MAX_TEXT_ANSWER_LENGTH:
                    seq_len=len(sent_decoder[0].split(" "))
                    #Tokenization
                    input_ids_list=[[fill_tensor_1D(torch.zeros(self.vocab_size, dtype=torch.float), self.word2index[word], 1) for word in sent_decoder[i].split(" ")] for i in range(args.batch_size)]
                    input_ids_integers=[[self.word2index[word] for word in sent_decoder[i].split(" ")] for i in range(args.batch_size)]
                    
                    input_ids=torch.stack([torch.stack(input_ids_list[i]) for i in range(args.batch_size)]).cuda()
                    #input_ids=torch.tensor(input_ids_list, dtype=torch.long).cuda()
                    position_ids = torch.triu(torch.tril(torch.ones([args.batch_size, len(sent_decoder[0].split(" ")), MAX_TEXT_ANSWER_LENGTH]))).cuda()
                    #Embedding
                    words_embeddings = F.linear(input_ids, self.MLPHead.weight.T)
                    position_embeddings = self.position_embeddings(position_ids)
                    #token_type_embeddings = self.token_type_embeddings(token_type_ids)

                    tgt = words_embeddings + position_embeddings
                    tgt = tgt.transpose(0,1)
                    
                    #Forward-pass
                    #attn_mask=torch.triu(torch.ones([seq_len, seq_len], dtype=torch.bool), 1).cuda()
                    decoder_out = self.decoder(tgt=tgt, memory=x)
                    decoder_out = decoder_out.transpose(0,1)
                    #Choice to pick last self-attention pair, could also sum them
                    decoder_out = decoder_out[:,-1, :]
                    #Computing probabilities for each word
                    out_probs = nn.Softmax(-1)(self.MLPHead(decoder_out))
                    #Figure out the word with maximum-likelihood with the mapping
                    #that is, find argmax(decoder_out)
                    out= [out_probs.tolist()[i].index(max(out_probs.tolist()[i])) for i in range(args.batch_size)]
                    chosen_words=[self.index2word[idx] for idx in out]
                    #Add to tgt if EOS not previously chosen
                    for sentence in range(args.batch_size):
                        sent_decoder[sentence]=sent_decoder[sentence] + \
                                    f" {chosen_words[sentence]}"*(1-flagEOS[sentence]) + \
                                    " [PAD]"*(flagEOS[sentence])
                        #Register if <EOS> chosen
                        if chosen_words[sentence]=="<EOS>":
                            flagEOS[sentence]=True
                
                out=[sent_decoder[i].replace("<SOS> ", "").replace(" [PAD]", "") for i in range(len(sent_decoder))]
        
        # elif 'x' in self.mode and ('l' in self.mode or 'r' in self.mode):
        #     featseq, x = self.lxrt_encoder(imgid, sent, (feat, pos))
        #     logit=self.logit_fc(torch.cat((featseq[0].reshape([args.batch_size, (QUERY_LENGTH+2)*self.hid_dim]), \
        #         featseq[1].reshape([args.batch_size, 100*self.hid_dim])), axis=1))
        #actionlogit = self.actionpredict_head(prelogit)
        #explogit= self.exppredict_head(prelogit)
        #logit=torch.cat((actionlogit, explogit), 1)

        return out