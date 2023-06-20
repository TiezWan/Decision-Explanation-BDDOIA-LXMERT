import os, logging, pdb
from tqdm import tqdm
import numpy as np
import torch
from torch import optim, nn, Tensor
from torch.utils import data
from typing import List, Tuple, Union

import src.lxrt_modified.optimization as mod_optimizer
from src.utils.param import args
from src.utils.utils import Document
from src.model import ModLXRT


logger = logging.getLogger(__name__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DataLoader():
    def __init__(self, dataset_list: Union[tuple, List], dataloader_setup: dict) -> None:
        
        self.dataset_list = dataset_list
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        
        self._init_dataloader(dataloader_setup)

    def _init_dataloader(self, dataloader_setup: dict) -> None:
        """Create dataloader"""

        if len(self.dataset_list) == 1:
            # Testing
            self.tr_dataloader = None
            self.val_dataloader = None
            self.te_dataloader = data.DataLoader(self.dataset_list[0],
                                                 batch_size=self.batch_size,
                                                 num_workers=self.num_workers,
                                                 **dataloader_setup['test'])
            
        elif len(self.dataset_list) == 2:
            # Training and validation
            self.tr_dataloader = data.DataLoader(self.dataset_list[0],
                                                 batch_size=self.batch_size,
                                                 num_workers=self.num_workers,
                                                 **dataloader_setup['train'])
            self.val_dataloader = data.DataLoader(self.dataset_list[1],
                                                  batch_size=self.batch_size,
                                                  num_workers=self.num_workers,
                                                  **dataloader_setup['val'])
            self.te_dataloader = None
            
        else:
            raise ValueError(f"Detected wrong size of dataset_list: {len(self.dataset_list)}")


class AOptimizer(DataLoader):
    def __init__(self, dataset_list: Union[tuple, List], dataloader_setup: dict, max_norm: float, tolerance: float, model: ModLXRT, optimizer: optim, criterion: nn) -> None:
        
        self.max_norm = max_norm
        self.tolerance = tolerance
        self.model = model
        self.optim = optimizer
        self.criterion = criterion
        self.epochs = args.epochs   

        super().__init__(dataset_list, dataloader_setup)

    def _loss_related_batch_loop(self, input: Tensor, target: Tensor, get_eval_type: str) -> float:
        """Compute loss"""

        loss = self.criterion(input, target)
        loss = torch.mean(loss, 0)
        if get_eval_type == 'train':
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            self.optim.step()

        elif get_eval_type == 'eval_train' or get_eval_type == 'val':
            pass

        return loss.item()

    def _batch_loop(self, get_eval_type: str, dataloader: DataLoader) -> Tuple:
        """Optimize the parameters in each batch"""

        batch_loss = []
        pre_answ = []
        true_answ = []
        img_sents_pair = []
        progress_bar = tqdm(dataloader, total=dataloader.__len__())

        for batch_idx, batch in enumerate(progress_bar):
            # Bring the data to GUP
            idx = batch[0]
            img_id = batch[1]
            boxes = batch[2].to(device)
            features = batch[3].to(device)
            sents = batch[4]
            labels = batch[5].unsqueeze(1).to(device)

            if get_eval_type == 'train':
                self.optim.zero_grad()
                prediction = self.model(img_id, features, boxes, sents)

                loss = self._loss_related_batch_loop(prediction, labels, get_eval_type)

            elif get_eval_type == 'eval_train' or get_eval_type == 'val':
                torch.cuda.empty_cache()
                with torch.no_grad():
                    prediction = self.model(img_id, features, boxes, sents)

                    loss = self._loss_related_batch_loop(prediction, labels, get_eval_type)
                    pre_answ += nn.Sigmoid()(prediction).cpu().numpy().tolist()
                    true_answ += labels.cpu().numpy().tolist()

            elif get_eval_type == 'test':
                pass

            else:
                raise ValueError(f"Unknown evaluation type: {get_eval_type}")
            
            progress_bar.set_description(f"Loss: {round(loss, 3)}")
            batch_loss.append(loss)
            img_sents_pair.append((f'Batch {batch_idx}:', '\n'))
            if pre_answ and args.save_predictions:
                img_sents_pair += list(zip(img_id, '\t' * len(img_id), sents, '\t' * len(img_id), pre_answ, '\t' * len(img_id), true_answ, '\n' * len(img_id)))
            else:
                img_sents_pair += list(zip(img_id, '\t' * len(img_id), sents, '\n' * len(img_id)))
            img_sents_pair = sum(img_sents_pair, ())

        return batch_loss, pre_answ, true_answ, img_sents_pair
            
    def _epoch_checkpoint_save(self, epoch: int, file_name: str) -> None:
        """Save the weights for each epoch for later (re-)evaluation"""
        
        if not os.path.exists(f"{args.output}/checkpoints"):
            os.makedirs(f"{args.output}/checkpoints")
        torch.save({'epoch': epoch, 'model': self.model.state_dict()},
                   os.path.join(args.output, f"checkpoints/{file_name}_checkpoint_epoch_{epoch}.pth"))

    def train(self) -> None:
        """Train the model and do validation"""

        best_acc = 0.
        best_bitwise_acc = 0.
        best_f1_score = 0.

        for epoch in range(self.epochs):
            logger.info(f"Epoch: \t{epoch}/{self.epochs}")

            # Start training
            logger.info(f"Start training")
            self.model.train()
            self.model.to(device)
            if args.multi_GPU:
                self.model.lxrt_encoder.multi_gpu()
            batch_loss, _, _, img_sents_pair = self._batch_loop('train', self.tr_dataloader)
            
            # Document loss per epoch
            content = {'epoch': epoch, 'loss': batch_loss}
            Document.docu_training_loss_hist(content=content)
            # Saves the weights for each epoch for later (re-)evaluation
            self._epoch_checkpoint_save(epoch, 'train')
            
            # Evaluate training
            logger.info(f"Start evaluating training")
            avg_loss, conf_mat, f1_score, label_acc, bitwise_acc, trace, img_sents_pair = self.eval('eval_train')
            # Document the evaluation result per epoch
            content = {'epoch': epoch,
                       'training_accuracy': label_acc * 100,
                       'training_bitwise_accuracy': bitwise_acc * 100,
                       'training_loss': avg_loss,
                       'training_f1_score': f1_score,
                       'training_confusion_matrix': conf_mat,
                       'training_trace': trace,
                       'training_image_question_pair': img_sents_pair}
            Document.docu_eval_hist(content=content, file_name="training_hist")

            # Start validation
            logger.info(f"Start validation")
            avg_loss, conf_mat, f1_score, label_acc, bitwise_acc, trace, img_sents_pair = self.eval('val')
            # Document the validation result per epoch
            content = {'epoch': epoch,
                       'training_accuracy': label_acc * 100,
                       'training_bitwise_accuracy': bitwise_acc * 100,
                       'training_loss': avg_loss,
                       'training_f1_score': f1_score,
                       'training_confusion_matrix': conf_mat,
                       'training_trace': trace,
                       'training_image_question_pair': img_sents_pair}
            Document.docu_eval_hist(content=content, file_name="validation_hist")
            if label_acc > best_acc:
                best_acc = label_acc
                self._epoch_checkpoint_save(epoch, 'validation')
            if bitwise_acc > best_bitwise_acc:
                best_bitwise_acc = bitwise_acc
            if f1_score > best_f1_score:
                best_f1_score = f1_score

        content = {'epoch': None,
                   'best_accuracy': best_acc * 100,
                   'best_bitwise_accuracy': best_bitwise_acc * 100,
                   'best_f1_score': best_f1_score}
        Document.docu_eval_hist(content=content, file_name="best_result")

    def eval(self, get_eval_type: str) -> Tuple:
        """Evaluate the training, validation and test sets"""

        self.model.eval()
        if get_eval_type == 'eval_train':
            batch_loss, pre_answ, true_answ, img_sents_pair = self._batch_loop('eval_train', self.tr_dataloader)
            avg_loss = sum(batch_loss) / len(batch_loss)
            conf_mat = self.compute_conf_mat(pre_answ, true_answ)
            f1_score = self.compute_f1_score(conf_mat)
            label_acc, bitwise_acc, trace = self.compute_acc(pre_answ, true_answ)

        elif get_eval_type == 'val':
            batch_loss, pre_answ, true_answ, img_sents_pair = self._batch_loop('val', self.val_dataloader)
            avg_loss = sum(batch_loss) / len(batch_loss)
            conf_mat = self.compute_conf_mat(pre_answ, true_answ)
            f1_score = self.compute_f1_score(conf_mat)
            label_acc, bitwise_acc, trace = self.compute_acc(pre_answ, true_answ)

        elif get_eval_type == 'test':
            pass
        else:
            raise ValueError(f"Unknown evaluation type: {get_eval_type}")
        
        return avg_loss, conf_mat, f1_score, label_acc, bitwise_acc, trace, img_sents_pair
        
    def compute_conf_mat(self, pre_answ, true_answ) -> Tuple:
        """Compute the confusion matrix"""

        true_neg, false_pos, false_neg, true_pos = 0, 0, 0, 0
        for i in range(len(pre_answ)):
            for ques_idx in range(len(pre_answ[i])):
                true_neg += (1-int(pre_answ[i][ques_idx] + 0.5)) * (1 - int(true_answ[i][ques_idx]))
                false_pos += int(pre_answ[i][ques_idx] + 0.5) * (1 - int(true_answ[i][ques_idx]))
                false_neg += (1-int(pre_answ[i][ques_idx] + 0.5)) * int(true_answ[i][ques_idx])
                true_pos += int(pre_answ[i][ques_idx] + 0.5) * int(true_answ[i][ques_idx])

        return (true_neg, false_pos, false_neg, true_pos)
    
    def compute_acc(self, pre_answ, true_answ) -> Tuple:
        """Compute the accuracy"""

        correctly_answered_image = 0 # Number of images that all the associated questions are correctly answered
        correctly_answered_ques = 0 # Number of questions that are correctly answered
        trace = np.zeros(len(pre_answ[0])) # Trace whether the questions in different types can be correctly answered (for testing)

        for i in range(len(pre_answ)):
            correct_answ = True
            for ques_idx in range(len(pre_answ[i])):
                # For test dataset, one image will be asked several questions
                # In other cases, one image will only be asked one question
                if pre_answ[i][ques_idx] >= true_answ[i][ques_idx] - self.tolerance and pre_answ[i][ques_idx] < true_answ[i][ques_idx] + self.tolerance:
                    correctly_answered_ques += 1
                else:
                    correct_answ = False
                    trace[ques_idx] += 1
            
            if correct_answ:
                correctly_answered_image += 1

        label_acc = correctly_answered_image / len(pre_answ)
        bitwise_acc = correctly_answered_ques / (len(pre_answ) * len(pre_answ[0]))

        return label_acc, bitwise_acc, trace.tolist()
    
    def compute_f1_score(self, conf_mat) -> float:
        """Compute the F1-score"""

        try:
            return conf_mat[3] / (conf_mat[3] + 0.5 * (conf_mat[1] + conf_mat[2]))
        except:
            pdb.set_trace()
            return 999.


class Optimizer(AOptimizer):
    def __init__(self, dataset_list: Union[tuple, List], model: ModLXRT) -> None:

        self.lr = args.lr
        self.t_total = args.epochs * dataset_list[0].__len__()
        optimizer = self._init_optimizer(model)
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        dataloader_setup = {'test': {'shuffle': True, 'drop_last': True, 'pin_memory': False},
                            'train': {'shuffle': True, 'drop_last': True, 'pin_memory': False},
                            'val': {'shuffle': True, 'drop_last': True, 'pin_memory': False}}
        
        super().__init__(dataset_list, dataloader_setup, 5, 0.5, model, optimizer, criterion)
        
        text = f"Initializing an optimizer:\n"
        text += f"Optimizer: \t{args.optim}\n"
        text += f"Learning Rate: \t{self.lr}\n"
        text += f"Epochs: \t{self.epochs}\n"
        text += f"Batch Size: \t{self.batch_size}\n\n"
        logger.info(text)

    def _init_optimizer(self, model: ModLXRT) -> optim:
        """Initialize the optimizer"""

        if args.optimizer == 'bert':
            # The same as the baseline
            return mod_optimizer.BertAdam(model.parameters(), lr=self.lr, warmup=0.1, t_total=self.t_total)
        else:
            return args.optimizer(model.parameters(), self.lr)