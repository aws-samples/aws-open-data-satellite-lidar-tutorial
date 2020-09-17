#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# This script setup a conda environment that contains dependencies of
# the 'solaris' module.

# Exit when error occurs
set -e

# Create conda environment
export ENV_NAME=$@
conda create -n $ENV_NAME -y --channel conda-forge

# Activate the environment in Bash shell
. /home/ec2-user/anaconda3/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Install dependencies
conda install --file conda-requirements.txt -y --channel conda-forge
pip install -r pip-requirements.txt

