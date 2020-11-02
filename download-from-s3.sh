#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# This script downloads data used in this tutorial from S3 buckets.

# Exit when error occurs
set -e

echo "===== Downloading RGB+LiDAR merged data (~22GB) ... ====="
aws s3 cp s3://aws-satellite-lidar-tutorial/data/ ./data/ --recursive --no-sign-request

echo "===== Downloading pretrained model weights (617MB) ... ====="
aws s3 cp s3://aws-satellite-lidar-tutorial/models/ ./models/ --recursive --no-sign-request

echo "===== Downloading completes. ====="
