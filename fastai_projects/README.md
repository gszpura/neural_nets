# Installation and Set Up (Ubuntu 22.04)

Following steps need to be done:
1. install nvidia driver with:
> sudo ubuntu-drivers install

this should automatically install best driver
> https://ubuntu.com/server/docs/nvidia-drivers-installation

it is enough to install general purpose driver

after installtion of nvidia driver it should be possible to
run commands from `Verify Nvidia` section.

2. Download and install cuDNN, follow instructions from:
> https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local

Installation is quick, download is about 2GB.

3. Download and install cuda toolkit
Follow instructions from:
> https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local


4. mkvirtualenv `aiproj`
5. pip install -r requirements.txt
   1. this might take a while (more than 2GB to download)

After those steps everything should work out of the box.


# Work
1. workon `aiproj`
2. jupyter notebook

Start with CatOrDogClassifier notebook.


# Verify Nvidia driver
>nvidia-settings

> nvidia-settings -q NvidiaDriverVersion

> cat /proc/driver/nvidia/version


# GPU usage during training
Run to see if GPU is used:
> nvidia-smi -q -g 0 -d UTILIZATION -l

