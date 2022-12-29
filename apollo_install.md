sudo apt-get update

安装docker
curl https://get.docker.com | sh
sudo systemctl start docker && sudo systemctl enable docker

测试docker
sudo docker run hello-world

添加docker用户组
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
测试：
docker run hello-world

安装 NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get -y update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

源码 clone
sudo apt install git -y
git clone https://github.com/ApolloAuto/apollo.git

启动docker容器
cd ~/apollo
bash docker/scripts/dev_start.sh -g cn
当提示无法pull镜像时，添加网易的镜像源：
vim /etc/docker/daemon.json 
{
  "registry-mirrors": [
    "https://hub-mirror.c.163.com"
  ],
  "live-restore": true
}

进入docker容器
bash docker/scripts/dev_into.sh

输入nvidia-smi来校验 NVIDIA GPU 在容器内是否能正常运行

编译 Apollo 源码
bash apollo.sh build

启动 dreamview
bash scripts/bootstrap.sh
浏览器中输入：http://localhost:8888

回放数据包：
wget https://apollo-system.cdn.bcebos.com/dataset/6.0_edu/demo_3.5.record
cyber_recorder play -f demo_3.5.record --loop