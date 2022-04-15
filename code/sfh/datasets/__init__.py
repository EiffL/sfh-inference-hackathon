"""Datasets for AstroInfo 2021 SFH Hackathon."""
import os

import tensorflow_datasets as tfds


def setup_environment():
    """Set up the environment for using the datasets.

    This function configures tensorflow_datasets to find the datasets in the
    correct location.
    """
    # Set TNG100_DATA_PATH on Jean Zay if no environment variable
    os.environ['TNG100_DATA_PATH'] = os.getenv(
        'TNG100_DATA_PATH',
        f"{os.getenv('ALL_CCFRWORK')}/SFH/tng100/"
    )

    # Set EAGLE_DATA_PATH on Jean Zay if no EnvironmentError variable
    os.environ['EAGLE_DATA_PATH'] = os.getenv(
        'EAGLE_DATA_PATH',
        f"{os.getenv('ALL_CCFRWORK')}/SFH/eagle/"
    )

    # Set the location of the tensorflow datasets.
    tfds.core.constants.DATA_DIR = os.getenv(
        'TFDS_DATA_DIR',
        f"{os.getenv('ALL_CCFRWORK')}/tensorflow_datasets/"
    )
