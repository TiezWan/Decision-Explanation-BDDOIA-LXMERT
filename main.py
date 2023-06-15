import warnings, logging, pdb, os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

from src.dataset import Dataset
from src.model import ModLXRT
from src.optimizer import Optimizer
from src.utils.logger import logger_setup
from src.utils.param import args

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
    model = ModLXRT()
    model.load_weights()

    if args.train:
        # Start training
        optimizer = Optimizer((training_set, val_set), model)
        optimizer.train() # to train the model on the loaded dataset

    elif args.test:
        # ToDo
        pass

