#!/bin/bash  
echo "Creating a virtual environment called metric-env"  
python3 -m venv ./metric
echo "I have created a virtual environment called metric"
pip3 install awscli==1.27.36 
pip3 install boto3==1.26.32 
pip3 install botocore==1.29.36 
pip3 install matplotlib==3.5.3 
pip3 install matplotlib-inline==0.1.6  
pip3 install numpy==1.21.6 
pip3 install opencv-python==4.6.0.66 
pip3 install pandas==1.3.5  
pip3 install Pillow==9.3.0  
pip3 install scipy==1.7.3 
pip3 install seaborn==0.12.1 
pip3 install torch==1.13.1 
pip3 install torch-fidelity==0.3.0 
pip3 install torchmetrics==0.11.0 
pip3 install torchvision==0.14.1   
pip3 install tqdm==4.64.1  
pip3 install gdown
echo "I have created a completed installing the packages in the virtual environment called metric"
git clone https://github.com/mlfoundations/open_clip.git
cd open_clip
$pip3 install pip install open_clip_torch
cd ..
echo "I have installed the open_clip library"
git clone https://github.com/TabithaKO/face-parsing.PyTorch.git
cd face-parsing.PyTorch
gdown the_weights
