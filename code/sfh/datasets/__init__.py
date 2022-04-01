"""Datasets for AstroInfo 2021 SFH Hackathon."""
import os

import tensorflow_datasets as tfds


def setup_environment():
    """Set up the environment for using the datasets.

    This function configures tensorflow_datasets to find the datasets in the
    correct location.
    """
    tfds.core.constants.DATA_DIR = os.getenv(
        'TFDS_DATA_DIR',
        f"{os.getenv('ALL_CCFRWORK')}/tensorflow_datasets/"
    )
