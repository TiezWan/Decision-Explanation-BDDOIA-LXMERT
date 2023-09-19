from src.utils.param import args
from src.utils.logger import logger_setup
from src.optimizer import Optimizer
from src.model import ModLXRT
from src.dataset import Dataset
import warnings
import logging
import pdb
import os

os.environ[
    "CUDA_VISIBLE_DEVICES"
] = "2"  # ToDo: extract the ENV.variable outside the codes

# Attempts to ignore deprecated warnings from Pytorch, as those haven't been fixed in this torch version
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # Initialize logger
    logger_setup()
    logger = logging.getLogger(__name__)
    # Train and validation datasets
    assert not (args.train and args.test), "Perform either training or testing."
    assert not (not args.train and not args.test), "Perform either training or testing."

    if args.train:
        training_set = Dataset("train")
        val_set = Dataset("val")
    elif args.test:
        test_set = Dataset("test")

    # Create model and load fine-tuned weights
    model = ModLXRT()  # ToDo: do we have a choice for Baseline LXRT?
    model.load_weights()

    if args.train:
        # Start training
        optimizer = Optimizer((training_set, val_set), model)
        optimizer.train()  # to train the model on the loaded dataset

    elif args.test:
        optimizer = Optimizer((test_set,), model)
        optimizer.test()  # to test the model on the loaded dataset
