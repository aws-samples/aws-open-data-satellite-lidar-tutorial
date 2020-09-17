# Deep Learning on AWS Open Data Registry: Automatic Building and Road Extraction from Satellite and LiDAR

This is the repository for OpenData tutorial content by MLSL.

## Setup

### Create a SageMaker instance
The tutorial can be run with any SageMaker instance type, but we highly recommend instance type with GPU support. For example, `ml.p?.?xlarge` series. The EBS volume size should be more than 60GB in order to store all necessary data.

Network training/inference is a memory-intensive process. If you run into out of GPU memory or out of RAM error, consider decrease the number of `batch_size` in the `yml` config files in the `configs` folder.

### Clone this repository
Once the SageMaker instance is successfully launched, open a terminal and follow the commands below:
```shell
$ cd ~/SageMaker/
$ git clone https://git-codecommit.us-east-2.amazonaws.com/v1/repos/SpaceNet-LIDAR
$ cd SpaceNet-LIDAR
```
This will download the repository and take you to the repository directory.

### Create Conda environment
Next, set up a Conda environment by running `setup-env.sh` as shown below. You can change the environment name from `tutorial_env` to any other names.
```shell
$ ./setup-env.sh tutorial_env
```
This may take 10--15 minutes to complete.

Then check to make sure you have a new Jupyter kernel called `conda_tutorial_env`, or `conda_[name]` if you change the environment name to `[name]`. You may need to wait for a couple of minutes and refresh the Jupyter page.

### Download from S3 buckets
Next, download necessary files from S3 bucket prepared for this tutorial by running `download-from-s3.sh`:
```shell
$ ./download-from-s3.sh
```
This may take 5 minutes to complete, and requires at least 23GB of EBS disk size.

## Launch notebook
Finally, you can launch the notebooks `Building-Footprint.ipynb` or `Road-Network.ipynb` and learn to reproduce the tutorial. Note that if the notebook shows "No Kernel", or prompts to "Select Kernel", select the Jupyter kernel created in the previous step.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.
The [NOTICE](THIRD-PARTY) includes third-party licenses used in this repository.

