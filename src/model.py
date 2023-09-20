import torch, logging
import torch.nn as nn
import numpy as np
from src.utils.param import args
import src.lxrt.modeling_base as modeling
from src.lxrt.modeling_base import QUERY_LENGTH
from src.lxrt.entry import LXRTEncoder
import src.lxrt.modeling_modified as mod_modeling

# Number of output classes
LOGIT_ANSWER_LENGTH = 1

logger = logging.getLogger(__name__)


class ModLXRT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention_scores = None
        if args.baseline:
            # Using orginal LXMERT model
            self.lxrt_encoder = LXRTEncoder(
                args, max_seq_length=QUERY_LENGTH + 2, mode="x"
            )
            self.hid_dim = self.lxrt_encoder.dim

            # VQA Answer heads
            self.logit_fc = nn.Sequential(
                nn.Linear(self.hid_dim, self.hid_dim * 2),
                modeling.GeLU(),
                modeling.BertLayerNorm(self.hid_dim * 2, eps=1e-12),
                nn.Linear(self.hid_dim * 2, LOGIT_ANSWER_LENGTH),
            )
        else:
            # Using the modified version
            self.lxrt_encoder = LXRTEncoder(
                args, max_seq_length=QUERY_LENGTH + 2, mode="x"
            )
            self.hid_dim = self.lxrt_encoder.dim

            # VQA Answer heads
            self.logit_fc = nn.Sequential(
                nn.Linear(self.hid_dim, self.hid_dim * 2),
                mod_modeling.GeLU(),
                mod_modeling.BertLayerNorm(self.hid_dim * 2, eps=1e-12),
                nn.Linear(self.hid_dim * 2, LOGIT_ANSWER_LENGTH),
            )
        self.attention_scores = self.lxrt_encoder.attention_scores

    def forward(self, imgid, feat, pos, sent):
        output = self.lxrt_encoder(imgid, sent, (feat, pos))
        return self.logit_fc(output)

    def load_weights(self) -> None:
        """Load pre-trained weights"""

        if args.load:
            # Load specified weights, usually the fine-tuned weights for this task for testing
            assert args.load_lxmert is None, "Do not load weights twice."

            logger.info(f"Load weights from {args.load}.")
            state_dict = torch.load(f"{args.load}.pth")
            self.load_state_dict(state_dict)
            logger.info(f"weights from {args.load} loaded.")

        elif args.load_lxmert:
            # Load pre-trained weights given by LXMERT
            logger.info(f"Load pre-trained weights of LXMERT from {args.load_lxmert}.")
            self.lxrt_encoder.load(args.load_lxmert)
            logger.info(f"Weights from {args.load_lxmert} loaded.")

        else:
            raise ValueError(
                "No weights were loaded, the model was going to be trained from scratch."
            )
