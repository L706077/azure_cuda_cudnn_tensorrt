# azure_cuda_cudnn_tensorrt

### reference website to install Nvidia driver (367.57),than you can use "nvidia-smi":
https://docs.microsoft.com/en-us/azure/virtual-machines/virtual-machines-linux-n-series-driver-setup

############################################################## <br/>
before you install anything:

1.##check ubuntu version <br/>

$ lsb_release -a  <br/>


2.check nvidia graphy card <br/>

$ lspci | grep -i NVIDIA <br/>

############################################################## <br/>

(1).###install cuda toolkit <br/>

$ CUDA_REPO_PKG=cuda-repo-ubuntu1604_8.0.44-1_amd64.deb <br/>

$ wget -O /tmp/${CUDA_REPO_PKG} http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/${CUDA_REPO_PKG} <br/>

$ sudo dpkg -i /tmp/${CUDA_REPO_PKG} <br/>

$ rm -f /tmp/${CUDA_REPO_PKG} <br/>

$ sudo apt-get update <br/>

$ sudo apt-get install cuda <br/>

#sudo apt-get install cuda-drivers  #or #sudo apt-get install cuda <br/>

##############################################################

(2).###after install cuda toolkit: <br/>

$ nano ~/.bashrc <br/> 
$ export CUDA_HOME=/usr/local/cuda-8.0 <br/>
$ export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH <br/>
$ export PATH=${CUDA_HOME}/bin:${PATH} <br/>

enter "ctrl+x" to save <br/>

$ source ~/.bashrc <br/>


# than you can use "nvcc -V" <br/>

##############################################################



(3).##cudnn5.0 <br/>
 
##from notebook download cudnn-8.0-linux-x64-v5.0-ga.tgz, than upload to azure <br/>

$ tar xvzf cudnn-8.0-linux-x64-v5.0-ga.tgz <br/>
$ sudo cp -P cuda/include/cudnn.h /usr/local/cuda/include <br/>
$ sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64 <br/>
$ sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn* <br/>

###if have bug ref as following: <br/>
https://github.com/tensorflow/tensorflow/issues/5591 <br/>

##############################################################

(4).##install pip3 <br/>

$ sudo apt-get install build-essential gfortran libatlas-base-dev python-pip python-dev <br/>
$ sudo pip3 install --upgrade pip3 <br/>

$ sudo apt-get -y install python3-pip <br/>



(5).---install numpy/scipy/sklearn/matplotlib <br/>

$ sudo pip3 install numpy <br/>
$ sudo pip3 install scipy <br/>
$ sudo pip3 install -U scikit-learn <br/>
$ sudo pip3 install matplotlib <br/>


#############################################################

(6).### install tensorflow <br/>

https://www.tensorflow.org/get_started/os_setup <br/>
 
# Ubuntu/Linux 64-bit, GPU enabled, Python 3.5 <br/>
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp35-cp35m-linux_x86_64.whl <br/>

# Python 3 <br/>
$ sudo pip3 install --upgrade $TF_BINARY_URL <br/>


### for test: <br/>

$ python3 <br/>
>>> import tensorflow as tf <br/>
>>> hello = tf.constant('Hello, TensorFlow!') <br/>
>>> sess = tf.Session() <br/>
>>> print(sess.run(hello)) <br/>

Hello, TensorFlow! <br/>

##############################################################


(7).###Install nltk <br/>
https://www.quora.com/How-do-I-install-NLTK-on-Ubuntu <br/>

(tqdm  僅把指令改成tqdm) <br/>

$ sudo pip3 install -U nltk <br/>
$ sudo pip3 install -U tqdm <br/>


##############################################################

(8).###necessary to be installed <br/>

$ python3 -m nltk.downloader punkt <br/>

##############################################################

(9).###install DeepQA: <br/>

$ git clone https://github.com/Conchylicultor/DeepQA <br/>

$ python3 main.py <br/>

##############################################################

### some command line: <br/>
$ rm -rf <br/>
$cp (original file) (will be done file) <br/>

##############################################################

(10).
## https://www.microway.com/hpc-tech-tips/nvidia-smi_control-your-gpus/  <br/>
##persistence mode <br/>
$ sudo nvidia-smi -pm 1 <br/>

$ nvidia-smi  -q -i 0 -d CLOCK <br/>

$ sudo nvidia-smi -ac 3004,875 -i 0 <br/>


#############################################################

*** Remove any other installation (include card driver, nvidia-smi) <br/>
$ sudo apt-get purge nvidia-cuda* <br/>

 
*** if you want to install the drivers too, then  <br/>
$ sudo apt-get purge nvidia-*. <br/>


$ sudo apt-get remove --purge nvidia-WHATEVER <br/>


*** Remove only CUDA (nvcc -V)*** <br/>
$ sudo apt-get purge --auto-remove nvidia-cuda-toolkit <br/>


#############################################################

*** install tensorrt *** <br/>
$ sudo dpkg -i nv-tensorrt-repo-ubuntu1404-7-ea-cuda8.0_2.0.1-1_amd64.deb <br/>

$ sudo apt-get update <br/>

$ sudo apt-get install tensorrt-2 <br/>


*** test tensorrt if be installed or not *** <br/>

$ dpkg -l | grep tensorrt-2 <br/>
tensorrt-2 2.0.0-1+cuda amd64 Meta package of TensorRT <br/>


$ dpkg -l | grep nvinfer2 <br/>
libnvinfer2 2.0.0-1+cuda amd64 TensorRT runtime libraries <br/>


 




