import os, logging

from src.utils.param import args

logger = logging.getLogger(__name__)

class Document():  # ToDo: why and how do we use this class?
    @staticmethod
    def docu_test_set(content: dict, file_name: str='val_acc', mode: str='w', shown=False) -> None:
        """Document results using test set"""

        if not os.path.exists(f"{args.output}/logs"):
            os.makedirs(f"{args.output}/logs")
        
        with open(f"{args.output}/logs/{file_name}_epoch_{content['epoch']}.log", mode) as f:
            f.write(f"Pre-trained weights: \t")
            if args.load:
                f.write(args.load)
            elif args.load_lxmert:
                f.write(args.load_lxmert)
            else:
                f.write(f"From scratch")

            f.write("\n")
            f.write(f"Validation set:")
            f.write("\n")

            for key in content.keys():
                f.write(f"{key.replace('_', ' ').title()}: \t{str(content[key])}")
                f.write("\n")
            f.write("\n")
            f.flush()

        if shown:
            from_scratch = "From scratch"
            text = f"Pre-trained weights: \t{args.load if args.load else (args.load_lxmert if args.load_lxmert else from_scratch)}."
            text += f"\nValidation set:\n"
            for key in content.keys():
                text += f"{key.replace('_', ' ').title()}: \t{str(content[key])}"
                text += "\n"
            text += "\n"

            logger.info(text)

    @staticmethod
    def docu_eval_hist(content: dict, file_name: str, mode: str='w', shown=False) -> None:
        """Document the evaluation result"""

        if not os.path.exists(f"{args.output}/logs"):
            os.makedirs(f"{args.output}/logs")
        
        with open(f"{args.output}/logs/{file_name}_epoch_{content['epoch']}.log", mode) as f:
            for key in content.keys():
                f.write(f"{key.replace('_', ' ').title()}: \t{str(content[key])}")
                f.write("\n")
            f.write("\n")
            f.flush()

        if shown:
            text = ""
            for key in content.keys():
                text += f"{key.replace('_', ' ').title()}: \t{str(content[key])}"
                text += "\n"
            text += "\n"

            logger.info(text)

    @staticmethod
    def docu_training_loss_hist(content: dict, file_name: str='training_loss_hist', mode: str='w', shown=False) -> None:
        """Document the loss history"""

        if not os.path.exists(f"{args.output}/logs"):
            os.makedirs(f"{args.output}/logs")
        
        with open(f"{args.output}/logs/{file_name}_epoch_{content['epoch']}.log", mode) as f:
            f.write(f"Loss in epoch: \t{str(content['epoch'])}")
            f.write("\n")
            f.write(f"Loss: \t{str(content['loss'][0])}\n")
            for i in range(1, len(content['loss'])):
                f.write(f"\t\t{str(content['loss'][i])}\n")
            f.write("\n")
            f.flush()

        if shown:
            text = f"Loss in epoch: \t{str(content['epoch'])}"
            text += f"\n"
            text += f"Loss: \t{str(content['loss'])}"
            text += f"\n \n"

            logger.info(text)
