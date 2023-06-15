import argparse, random, torch
import numpy as np

def get_optimizer(optim):
    """Bind the optimizer"""
    
    if optim == 'rms':
        print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamax':
        print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        print("Optimizer: sgd")
        optimizer = torch.optim.SGD
    elif 'bert' in optim:
        # The bert optimizer will be bind later.
        optimizer = 'bert'
    else:
        assert False, f"Please add your optimizer {optim} in the list."

    return optimizer

def parse_args():
    parser = argparse.ArgumentParser()

    # Data Splits
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--baseline", action='store_true')

    # Training Hyper-parameters
    parser.add_argument('--batchSize', dest='batch_size', type=int, default=10)
    parser.add_argument('--optim', type=str, default='bert')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--samples_num', type=int, default=None)
    parser.add_argument('--seed', type=int, default=9595, help='random seed')

    # Debugging
    parser.add_argument('--output', type=str, default='/root/Documents/xISVQA/output') # ! Default for debug
    parser.add_argument('--input', type=str, default='/root/Documents/xISVQA/input/bdd100k') # ! Default for debug
    # ? parser.add_argument("--fast", action='store_const', default=False, const=True)
    # ? parser.add_argument("--tiny", action='store_const', default=False, const=True)
    # ? parser.add_argument("--tqdm", action='store_const', default=False, const=True)
    parser.add_argument("--saveHeatmap", dest='save_heatmap', action='store_true')
    parser.add_argument("--savePredictions", dest='save_predictions', action='store_true')

    # Model Loading
    parser.add_argument('--load', type=str, default=None, help='Load specified weights, usually the fine-tuned weights for our task for testing.')
    parser.add_argument('--loadLxmert', dest='load_lxmert', type=str, default=None, help='Load pre-trained weights given by LXMERT.')
    parser.add_argument("--fromScratch", dest='from_scratch', action='store_const', default=False, const=True,
                        help='If none of the --load, --loadLXMERT, --loadLXMERTQA is set, the model would be trained from scratch. If --fromScratch is'
                        ' not specified, the model would load BERT-pre-trained weights by default.')

    # Optimization
    # ? parser.add_argument("--mceLoss", dest='mce_loss', action='store_const', default=False, const=True)

    # LXRT Model Config
    # Note: LXRT = L, X, R (three encoders), Transformer
    parser.add_argument("--llayers", default=9, type=int, help='Number of Language layers')
    parser.add_argument("--xlayers", default=5, type=int, help='Number of Cross-Modality layers.')
    parser.add_argument("--rlayers", default=5, type=int, help='Number of object Relationship layers.')
    parser.add_argument("--heads", default=12, type=int, help='Number of attention heads')
    # ? parser.add_argument("--num_decoderlayers", default=3, type=int, help='Number of layers in the decoder')

    # LXMERT Pre-training Config
    # ? parser.add_argument("--taskMatched", dest='task_matched', action='store_const', default=False, const=True)
    # ? parser.add_argument("--taskMaskLM", dest='task_mask_lm', action='store_const', default=False, const=True)
    # ? parser.add_argument("--taskObjPredict", dest='task_obj_predict', action='store_const', default=False, const=True)
    # ? parser.add_argument("--taskQA", dest='task_qa', action='store_const', default=False, const=True)
    # ? parser.add_argument("--visualLosses", dest='visual_losses', default='obj,attr,feat', type=str)
    # ? parser.add_argument("--qaSets", dest='qa_sets', default=None, type=str)
    # ? parser.add_argument("--wordMaskRate", dest='word_mask_rate', default=0.15, type=float)
    # ? parser.add_argument("--objMaskRate", dest='obj_mask_rate', default=0.15, type=float)

    # Training configuration
    parser.add_argument("--multiGPU", dest='multi_GPU', action='store_true')
    parser.add_argument("--numWorkers", dest='num_workers', default=0)

    # Parse the arguments.
    args = parser.parse_args()

    # Bind optimizer class.
    args.optimizer = get_optimizer(args.optim)

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args


args = parse_args()