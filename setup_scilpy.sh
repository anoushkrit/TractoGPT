# https://github.com/scilus/containers-scilus?tab=readme-ov-file
sudo apt install libblas-dev liblapack-dev gfortran
sudo apt-get install git g++ python libeigen3-dev zlib1g-dev libqt5opengl5-dev libqt5svg5-dev libgl1-mesa-dev libfftw3-dev libtiff5-dev libpng-dev
sudo apt-get install python3-distutils build-essential jq
sudo apt install libfreetype6-dev
ENV=tractolearn
PROGRAM_PATH=/neuro/programs
conda create -n $ENV python=3.10 && source ~/.bashrc
conda activate ${ENV}
#mrtrix
conda install -c mrtrix3 mrtrix3
conda update -c mrtrix3 mrtrix3
#scilpy
pip install --upgrade setuptools wheel
pip install numpy
git clone https://github.com/getspams/spams-python && cd spams-python && pip install -e .
cd .. && rm -rf spams-python 
cd $PROGRAM_PATH
git clone https://github.com/scilus/scilpy.git
export SETUPTOOLS_USE_DISTUTILS=stdlib
cd scilpy/ && pip install -e .
# dmriQCpy
pip install git+https://github.com/scilus/dmriqcpy.git
# ants
cd $PROGRAM_PATH
wget https://github.com/ANTsX/ANTs/releases/download/v2.5.3/ants-2.5.3-ubuntu-20.04-X64-gcc.zip
unzip ants-2.5.3-ubuntu-20.04-X64-gcc.zip
export PATH=$PROGRAM_PATH/ants-2.5.3/bin:$PATH && source ~/.bashrc 
conda activate $ENV
# echo 'export PATH=$PROGRAM_PATH/ants-2.5.3/bin:$PATH' >> ~/.bashrc
#VTK
wget https://www.vtk.org/files/release/9.3/VTK-9.3.1.tar.gz
tar xf VTK-9.3.1.tar.gz
# MESA
# https://github.com/mesonbuild/meson
cd $PROGRAM_PATH
sudo apt install ninja-build
# python3 -m pip install meson==1.1.0
pip install meson==1.1.0
# sudo apt install meson
# wget https://archive.mesa3d.org/mesa-24.2.5.tar.xz
# tar xf mesa-24.2.5.tar.xz
# cd mesa-24.2.5.tar.xz
# source ~/.bashrc && conda activate $ENV
# meson setup builddir/

# VTK is installed already 
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
# WMA 
pip install git+https://github.com/SlicerDMRI/whitematteranalysis.git
