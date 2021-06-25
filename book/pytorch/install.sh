# Ubuntu 20.04 Software & Updates > Additional Drivers > Using NVIDIA driver (proprietary, tested)
sudo apt install graphviz
sudo apt install nvidia-cuda-toolkit
nvcc -V   # check if CUDA is successfully installed
conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip3 install opencv-python
pip3 install torchviz

# prepare data
wget www.di.ens.fr/~lelarge/MNIST.tar.gz
tar -zxvf MNIST.tar.gz
mv MNIST ~/junk/

# install opencv
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev
sudo apt-get install  libcairo2-dev
sudo add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
sudo apt update
sudo apt install libjasper1 libjasper-dev

git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

cd opencv && mkdir build && cd build
cmake -D INSTALL_C_EXAMPLES=OFF -D BUILD_opencv_java=OFF -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
make
sudo make install
sudo sh -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf'
sudo ldconfig
