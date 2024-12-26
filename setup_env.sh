ENV_NAME=tractogpt
conda create --name $ENV_NAME python=3.9
conda activate $ENV_NAME
conda install -c nvidia/label/cuda-12.1.0 cuda-toolkit
export PATH=$PATH:$CONDA_PREFIX/bin
# pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
# pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# 11.8
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
# pip install git+https://github.com/erikwijmans/Pointnet2_PyTorch.git@b5ceb6d9ca0467ea34beb81023f96ee82228f626#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib
# pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
cd $CONDA_PREFIX/envs/
# TODO: Change TORCH_CUDA_ARCH_LIST in the pointnet2_ops_lib 
# # os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"
# os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
cd Pointnet2_PyTorch/pointnet2_ops_lib/
pip install -e .
cd ~
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl