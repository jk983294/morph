# Ubuntu 20.04 Software & Updates > Additional Drivers > Using NVIDIA driver (proprietary, tested)
sudo apt install nvidia-cuda-toolkit
nvcc -V   # check if CUDA is successfully installed
conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# prepare data
wget www.di.ens.fr/~lelarge/MNIST.tar.gz
tar -zxvf MNIST.tar.gz
mv MNIST ~/junk/