```sh
export HF_ENDPOINT=https://hf-mirror.com

export HF_HOME=/mnt/workspace/huggingface
echo $HF_HOME

huggingface-cli download --resume-download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

llamafactory-cli webui

deepseek-r1模型位置: /mnt/workspace/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/530ca3e1ad39d440e182c2e4317aa40f012512fa

模型导出位置: /mnt/workspace/LLAMA-Factory/merge
```

登录hugging face
```sh
pip install --upgrade huggingface_hub
huggingface-cli login
```
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

环境
## Requirement

|Mandatory|Minimum|Recommend|
|---|---|---|
|python|3.9|3.10|
|torch|2.0.0|2.6.0|
|torchvision|0.15.0|0.21.0|
|transformers|4.45.0|4.50.0|
|datasets|2.16.0|3.2.0|
|accelerate|0.34.0|1.2.1|
|peft|0.14.0|0.15.1|
|trl|0.8.6|0.9.6|

|Optional|Minimum|Recommend|
|---|---|---|
|CUDA|11.6|12.2|
|deepspeed|0.10.0|0.16.4|
|bitsandbytes|0.39.0|0.43.1|
|vllm|0.4.3|0.8.2|
|flash-attn|2.5.6|2.7.2|

### Hardware Requirement
* _estimated_

|Method|Bits|7B|14B|30B|70B|`x`B|
|---|---|---|---|---|---|---|
|Full (`bf16` or `fp16`)|32|120GB|240GB|600GB|1200GB|`18x`GB|
|Full (`pure_bf16`)|16|60GB|120GB|300GB|600GB|`8x`GB|
|Freeze/LoRA/GaLore/APOLLO/BAdam|16|16GB|32GB|64GB|160GB|`2x`GB|
|QLoRA|8|10GB|20GB|40GB|80GB|`x`GB|
|QLoRA|4|6GB|12GB|24GB|48GB|`x/2`GB|
|QLoRA|2|4GB|8GB|16GB|24GB|`x/4`GB|
安装
```sh
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```
使用docker安装
```sh
docker run -it --rm --gpus=all --ipc=host hiyouga/llamafactory:latest
```
启动
```sh
llamafactory-cli webui
```
下载(这里默认是modelscope)
### Download from ModelScope Hub

If you have trouble with downloading models and datasets from Hugging Face, you can use ModelScope.

```shell
export USE_MODELSCOPE_HUB=1 # `set USE_MODELSCOPE_HUB=1` for Windows
```

Train the model by specifying a model ID of the ModelScope Hub as the `model_name_or_path`. You can find a full list of model IDs at [ModelScope Hub](https://modelscope.cn/models), e.g., `LLM-Research/Meta-Llama-3-8B-Instruct`.
日志监控
### Use SwanLab Logger

To use [SwanLab](https://github.com/SwanHubX/SwanLab) for logging experimental results, you need to add the following arguments to yaml files.

```yaml
use_swanlab: true
swanlab_run_name: test_run # optional
```

When launching training tasks, you can log in to SwanLab in three ways:

1. Add `swanlab_api_key=<your_api_key>` to the yaml file, and set it to your [API key](https://swanlab.cn/settings).
2. Set the environment variable `SWANLAB_API_KEY` to your [API key](https://swanlab.cn/settings).
3. Use the `swanlab login` command to complete the login.
### 整体环境配置
你这张图展示的是一个虚拟环境（或者镜像）的命名规范：

```
ubuntu22.04-cuda12.1.0-py311-torch2.3.1-tf2.16.1-1.29.0
```

意思是：

- **操作系统**：Ubuntu 22.04
    
- **CUDA 版本**：12.1.0
    
- **Python 版本**：3.11
    
- **PyTorch 版本**：2.3.1
    
- **TensorFlow 版本**：2.16.1
    
- **可能的环境/镜像版本号**：1.29.0
    

---

## 安装对应虚拟环境的思路

### 方法一：使用 Conda（推荐）

```bash
# 1. 创建 Python 3.11 环境
conda create -n dl_env python=3.11 -y
conda activate dl_env

# 2. 安装 CUDA 12.1 对应的 PyTorch
# （官网推荐安装命令，需根据CUDA版本确认）
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. 安装 TensorFlow 2.16.1（支持 CUDA 12.x）
pip install tensorflow==2.16.1

# 4. 检查 CUDA、PyTorch、TensorFlow
python -c "import torch; print('Torch CUDA:', torch.cuda.is_available())"
python -c "import tensorflow as tf; print('TF GPU:', len(tf.config.list_physical_devices('GPU')))"
```

---

### 方法二：使用 Docker 镜像

如果你的系统环境不稳定，直接拉取官方或定制的镜像更简单。

```bash
docker pull nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# 进入容器后再安装 Python 与深度学习框架
docker run --gpus all -it nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 bash

# 容器中执行：
apt update && apt install -y python3.11 python3.11-venv python3-pip
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install tensorflow==2.16.1
```

---

### 方法三：用 requirements.txt 管理

你可以写一个 `requirements.txt`：

```
torch==2.3.1+cu121
torchvision
torchaudio
tensorflow==2.16.1
```

然后：

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

---
