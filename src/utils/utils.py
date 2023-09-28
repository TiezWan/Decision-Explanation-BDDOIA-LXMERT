import logging

from src.utils.param import args

logger = logging.getLogger(__name__)


class Document:
    @staticmethod
    def docu_eval_hist(content: dict, file_name: str, mode: str = "w") -> None:
        """Document the evaluation result"""

        with open(
            f"{args.output}/logs/{file_name}_epoch_{content['epoch']}.log", mode
        ) as f:
            for key in content.keys():
                f.write(f"{key.replace('_', ' ').title()}: \t{str(content[key])}")
                f.write("\n")
            f.write("\n")
            f.flush()

    @staticmethod
    def docu_training_loss_hist(
        content: dict, file_name: str = "training_loss_hist", mode: str = "w"
    ) -> None:
        """Document the loss history during training"""

        with open(
            f"{args.output}/logs/{file_name}_epoch_{content['epoch']}.log", mode
        ) as f:
            f.write(f"Loss in epoch: \t{str(content['epoch'])}")
            f.write("\n")
            f.write(f"Loss: \t{str(content['loss'][0])}\n")
            for i in range(1, len(content["loss"])):
                f.write(f"\t\t{str(content['loss'][i])}\n")
            f.write("\n")
            f.flush()
