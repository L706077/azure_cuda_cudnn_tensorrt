# azure_cuda_cudnn_tensorrt

### reference website to install Nvidia driver (367.57),than you can use "nvidia-smi":
https://docs.microsoft.com/en-us/azure/virtual-machines/virtual-machines-linux-n-series-driver-setup

##############################################################
before you install anything:

1.##check ubuntu version

$ lsb_release -a 


2.check nvidia graphy card

$ lspci | grep -i NVIDIA

##############################################################

(1).###install cuda toolkit

$ CUDA_REPO_PKG=cuda-repo-ubuntu1604_8.0.44-1_amd64.deb

$ wget -O /tmp/${CUDA_REPO_PKG} http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/${CUDA_REPO_PKG} 

$ sudo dpkg -i /tmp/${CUDA_REPO_PKG}

$ rm -f /tmp/${CUDA_REPO_PKG}

$ sudo apt-get update

$ sudo apt-get install cuda

#sudo apt-get install cuda-drivers  #or #sudo apt-get install cuda
###

##############################################################

(2).###after install cuda toolkit:

$ nano ~/.bashrc
$ export CUDA_HOME=/usr/local/cuda-8.0
$ export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
$ export PATH=${CUDA_HOME}/bin:${PATH}

enter "ctrl+x" to save

$ source ~/.bashrc


# than you can use "nvcc -V"

##############################################################



(3).##cudnn5.0
##from notebook download cudnn-8.0-linux-x64-v5.0-ga.tgz, than upload to azure
tar xvzf cudnn-8.0-linux-x64-v5.0-ga.tgz
sudo cp -P cuda/include/cudnn.h /usr/local/cuda/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

###if have bug ref as following:
https://github.com/tensorflow/tensorflow/issues/5591

##############################################################

(4).##install pip3

$ sudo apt-get install build-essential gfortran libatlas-base-dev python-pip python-dev
$ sudo pip3 install --upgrade pip3

$ sudo apt-get -y install python3-pip



(5).
---install numpy/scipy/sklearn/matplotlib
$ sudo pip3 install numpy
$ sudo pip3 install scipy
$ sudo pip3 install -U scikit-learn
$ sudo pip3 install matplotlib


##############################################################

(6).### install tensorflow
https://www.tensorflow.org/get_started/os_setup

# Ubuntu/Linux 64-bit, GPU enabled, Python 3.5
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp35-cp35m-linux_x86_64.whl

# Python 3
$ sudo pip3 install --upgrade $TF_BINARY_URL


### for test:

$ python3
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))

Hello, TensorFlow!

##############################################################


(7).

###Install nltk
https://www.quora.com/How-do-I-install-NLTK-on-Ubuntu

(tqdm  僅把指令改成tqdm)

$ sudo pip3 install -U nltk
$ sudo pip3 install -U tqdm


##############################################################

(8).
###necessary to be installed

$ python3 -m nltk.downloader punkt

##############################################################

(9).
###install DeepQA:

$ git clone https://github.com/Conchylicultor/DeepQA

$ python3 main.py

##############################################################

### some command line:
$ rm -rf
$cp (original file) (will be done file)

##############################################################

(10).
## https://www.microway.com/hpc-tech-tips/nvidia-smi_control-your-gpus/
##persistence mode
$ sudo nvidia-smi -pm 1

$ nvidia-smi  -q -i 0 -d CLOCK

$ sudo nvidia-smi -ac 3004,875 -i 0


#############################################################

*** Remove any other installation (include card driver, nvidia-smi)
$ sudo apt-get purge nvidia-cuda* 

 
*** if you want to install the drivers too, then 
$ sudo apt-get purge nvidia-*.


$ sudo apt-get remove --purge nvidia-WHATEVER


*** Remove only CUDA (nvcc -V)***
$ sudo apt-get purge --auto-remove nvidia-cuda-toolkit


#############################################################

*** install tensorrt ***
$ sudo dpkg -i nv-tensorrt-repo-ubuntu1404-7-ea-cuda8.0_2.0.1-1_amd64.deb

$ sudo apt-get update

$ sudo apt-get install tensorrt-2


*** test tensorrt if be installed or not ***

$ dpkg -l | grep tensorrt-2
tensorrt-2 2.0.0-1+cuda amd64 Meta package of TensorRT


$ dpkg -l | grep nvinfer2
libnvinfer2 2.0.0-1+cuda amd64 TensorRT runtime libraries


 




