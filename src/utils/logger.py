import os, logging
from src.utils.param import args


def logger_setup() -> None:
    """Set up the logger"""

    if not os.path.exists(f"{args.output}/logs"):
        os.makedirs(f"{args.output}/logs")

    if args.save_heatmap:
        if not os.path.exists(f"{args.output}/visualization_images"):
            os.makedirs(f"{args.output}/visualization_images")

    logs_dir = f"{args.output}/logs"

    # Define log format
    log_format = "[%(levelname)s]: %(asctime)s (%(name)s): %(message)s"
    date_format = "%H:%M:%S"

    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    # Define log files
    info_log_name = "info.log"
    warning_log_name = "warning.log"
    error_log_name = "error.log"

    info_log_path = os.path.join(logs_dir, info_log_name)
    warning_log_path = os.path.join(logs_dir, warning_log_name)
    error_log_path = os.path.join(logs_dir, error_log_name)

    info_log_fh = logging.FileHandler(info_log_path, "w", "utf8")
    warning_log_fh = logging.FileHandler(warning_log_path, "w", "utf8")
    error_log_fh = logging.FileHandler(error_log_path, "w", "utf8")

    info_log_fh.setLevel(logging.INFO)
    warning_log_fh.setLevel(logging.WARNING)
    error_log_fh.setLevel(logging.ERROR)

    info_log_fh.setFormatter(formatter)
    warning_log_fh.setFormatter(formatter)
    error_log_fh.setFormatter(formatter)

    # Define log that be sent to streams
    console_h = logging.StreamHandler()
    console_h.setLevel(logging.DEBUG)
    console_h.setFormatter(formatter)

    # Define the logger
    logger_root = logging.getLogger()
    logger_root.setLevel(logging.DEBUG)
    logger_root.addHandler(info_log_fh)
    logger_root.addHandler(warning_log_fh)
    logger_root.addHandler(error_log_fh)
    logger_root.addHandler(console_h)
