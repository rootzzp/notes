conda install --use-local
conda create --name mmdetect python=3.8 -y
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
conda config --add channels
conda config --remove channels
conda config --remove-key channels
conda remove -n xxx --all
conda create --name myclone --clone root

mmdetect:
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install -v -e .
