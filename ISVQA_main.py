# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections

import torch
import torch.nn as nn
# import glob
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from src.param import args
from src.pretrain.qa_answer_table import load_lxmert_qa
from src.vqa_model import VQAModel
from src.vqa_data_preprocessing import VQADataset, FeatureLoader, VQATorchDataset, VQAEvaluator

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')

os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3, 4, 5, 6"
# device = torch.device('cuda:1')

def get_data_tuple(jsonfile: str, feature_loader: FeatureLoader, bs: int, shuffle=False, drop_last=False) -> DataTuple:
    dset = VQADataset(jsonfile)
    # features = FeatureLoader(feature_path)
    tset = VQATorchDataset(dset, feature_loader.img_data)
    evaluator = VQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=False
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)

# self_train_tuple_dataset_num_answers = 650

class VQA:
    def __init__(self):
        feature_load = FeatureLoader(args.feature_path)
        # Datasets
        self.train_tuple = get_data_tuple(
            args.train, feature_loader=feature_load, bs=args.batch_size, shuffle=True, drop_last=True
        )
        # features = FeatureLoader(args.feature_path)
        if args.valid != "":
            self.valid_tuple = get_data_tuple(
                args.valid, feature_loader=feature_load, bs=args.batch_size,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None
        
        # Model
        self.model = VQAModel(self.train_tuple.dataset.num_answers)
        # self.model = VQAModel(self_train_tuple_dataset_num_answers)
        # self.model.to(device)
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        # self.model = nn.DataParallel(self.model, device_ids=[0, 1])
        
        
        

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)
        
        # GPU options
        self.model = self.model.cuda()
        # self.model = self.model.to(device)
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu(args)
            # self.model = nn.DataParallel(self.model.lxrt_encoder, device_ids=devi)

        # Loss and Optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("BertAdam Total Iters: %d" % t_total)
            from src.lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(self.model.parameters(), args.lr)
        
        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    # def iter_wrapper(self, x):
    #     return tqdm(x, total=len(loader))

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        # if args.tqdm:
        #     iter_wrapper = (lambda x: tqdm(x, total=len(loader)))
        # else:
        #     iter_wrapper = (lambda x: x)
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)
        # def f(x):
        #     return 2*x+1
        # y = f(2)
        best_valid = 0.
        for epoch in range(args.epochs):
            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, target) in iter_wrapper(enumerate(loader)):
                print('\r{}/{}'.format(i+1, len(loader)), end='')
                # if i > 50:
                #     break
                self.model.train()
                self.optim.zero_grad()

                # feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                target = target.cuda()
                logit = self.model(feats, boxes, sent)  
                assert logit.dim() == target.dim() == 2  # assert: 一种报错机制，若后面的语句不成立则报错
                loss = self.bce_loss(logit, target)
                a = logit.size(1)
                loss = loss * logit.size(1)  # logit.size(1) logit有两个dim，.size in Tensor (pytorch) = .shape in numpy

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)  # 裁剪梯度范数，若梯度范数大于5则归一化至5
                self.optim.step()  # 更新权重

                score, label = logit.max(1) 
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

            self.save(f"EPOCH_{epoch}")

    def predict(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]
            # print(feats.shape, boxes.shape)
            print('\r{}/{}'.format(i+1, len(loader)), end='')
               # Avoid seeing ground truth
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        quesid2ans = self.predict(eval_tuple, dump)
        return eval_tuple.evaluator.evaluate(quesid2ans)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target) in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid.item()] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s" % path)
        # state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)


if __name__ == "__main__":
    # Build Class
    vqa = VQA()

    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        vqa.load(args.load)

    # Test or Train
    if args.test is not None:
        # args.fast = args.tiny = False       # Always loading all data in test
        if 'test' in args.test:
            vqa.predict(
                get_data_tuple(args.test, bs=args.batch_size,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'test_predict.json')
            )
        elif 'val' in args.test:    
            # Since part of valididation data are used in pre-training/fine-tuning,
            # only validate on the minival set.
            # eval_tuple = get_data_tuple(args.valid, bs=args.batch_size, 
            #                             shuffle=False, drop_last=False)
            # test_paths = glob.glob('./snap/test_2/*.pth')
            # test_paths = ['./snap/test/EPOCH_10.pth', './snap/test/EPOCH_15.pth']
            # for i, tp in enumerate(test_paths):
            #     print(tp)
            #     vqa.load(tp)
            # result = vqa.evaluate(vqa.valid_tuple, dump=os.path.join(args.output, 'epoch{}.json'.format(i)))
                # result = vqa.evaluate(vqa.valid_tuple, dump=os.path.join(args.output, 'epoch20.json'))
                # print(result)
            # result = vqa.evaluate(
            #     get_data_tuple(args.valid, bs=args.batch_size,
            #                    shuffle=False, drop_last=False),
            #     dump=os.path.join(args.output, 'test_predict.json')
            # )
             result = vqa.evaluate(
                vqa.valid_tuple,
                dump=os.path.join(args.output, 'val_predict.json')
            )
        else:
            assert False, "No such test option for %s" % args.test
    else:
        # print('Splits in Train data:')
        if vqa.valid_tuple is not None:
            print('Use Valid data:', vqa.valid_tuple)
            # print("Valid Oracle: %0.2f" % (vqa.oracle_score(vqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        vqa.train(vqa.train_tuple, vqa.valid_tuple)