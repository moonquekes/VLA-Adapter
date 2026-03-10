<div align="center">
  <img src="figure/LOGO2.png" width="70%" style="vertical-align:-7px;" />


[![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2509.09372) [![Hugging Face Collection](https://img.shields.io/badge/Models-fcd022?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/VLA-Adapter) [![Twitter](https://img.shields.io/badge/AK-%23000000.svg?style=for-the-badge&logo=x&logoColor=white)](https://x.com/_akhaliq/status/1966610780838621241) [![WeChat](https://img.shields.io/badge/WeChat--Group-07C160?style=for-the-badge&logo=wechat&logoColor=white)](https://github.com/OpenHelix-Team/VLA-Adapter/issues/1)

</div>

### **VLA-Adapter** 官方实现。
<br/>

<div id="top" align="center">
<p align="center">
<img src=figure/Framework.png width=90% />
</p>
</div>

> **📝 论文: https://arxiv.org/abs/2509.09372**<br/>
> **🌍 项目主页: https://vla-adapter.github.io/**<br/>
> **🤗 HuggingFace: https://huggingface.co/VLA-Adapter**<br/>
> **Github: https://github.com/OpenHelix-Team/VLA-Adapter**

<br/>

## :loudspeaker: 新闻!
- **[2025/09/22]** 我们发布了代码！同时发布了增强的 **Pro** 版本（该版本遵循原论文的流水线，但在实现上进行了优化）。欢迎大家使用！🎉
- **[2025/09/13]** 我们的论文在 HF 的 [日榜](https://huggingface.co/papers/date/2025-09-12) 中获得🥇**第一名**，[周榜](https://huggingface.co/papers/week/2025-W37)中获得🥈**第二名**，[月榜](https://huggingface.co/papers/month/2025-09)中获得🥉**第三名**！ ⭐
- **[2025/09/13]** 我们的论文列入了 HF 的 [趋势论文](https://huggingface.co/papers/trending)！ ⭐
- **[2025/09/12]** 我们在 [HuggingFace](https://huggingface.co/VLA-Adapter) 上发布了针对四个 LIBERO 模型的 VLA-Adapter 原始版本。
- **[2025/09/11]** 我们在 [ArXiv](https://arxiv.org/abs/2509.09372) 上发布了论文。

<br/>

## :black_nib: 待办任务清单<a name="todo"></a>

- [x] 发布用于复现的 **权重 checkpoints**。
- [x] 发布 [VLA-Adapter v2 论文](https://arxiv.org/abs/2509.09372)。
- [ ] 更强大的版本 **VLA-Adapter++** 和详细的 **技术报告** 📝 即将发布。<br/>
- [ ] 继续更新代码以适应各种 **真实世界系统** 的部署，包括论文中的配置：Franka, UR-5, 和 AGILE Piper。<br/>
- [ ] 即将兼容 **各种基础模型**，包括但不限于 [VPP](https://arxiv.org/abs/2412.14803), [π0.5](https://arxiv.org/abs/2504.16054)。<br/>
- [ ] 以后将更新 **diffusion transformers** 和 **flow matching** 策略网络，结果将在随后的 VLA-Adapter++ 技术报告中更新。
- [ ] 我们还将更新并在 **冻结主干网络 (Frozen backbone)** 上进行更多实验。
- [ ] 未来我们将进一步扩展其 **泛化性**。工作正在进行中！敬请期待！
- [ ] **强化学习后续训练 (RL post-training)** 也在进行中。欢迎感兴趣的研究者加入我们，共同构建这个基石！
- [ ] **Dual-system 兼容性** 正在探索中！


<br/>

## 🌟 目录

- [:rocket: 快速开始](#rocket-quick-start) 
  - [VLA-Adapter 的 Conda 环境](#conda-environment-of-vla-adapter)
  - [安装依赖](#install-dependencies)
- [:pencil: 数据准备](#pencil-data-preparation) 
  - [LIBERO 基准测试](#libero-benchmark)
  - [CALVIN 基准测试](#calvin-benchmark)
  - [:video_game: 我们的依赖](#video_game-our-dependencies)
  - [:pushpin: 基准测试数据位置](#pushpin-benchmark-location)
- [⚓ VLM 主干网络](#vlm)
- [:fire: 不同配置的训练](#fire-training-for-different-configurations) &emsp; => 提供给显存从 **10GB** 到 **80GB** 的 GPU 的 **训练配置**。
  - [:books: 训练相关文件](#books-related-file-for-training)
  - [:ledger: 如何在显存极度受限的 GPU 上训练](#ledger-how-to-train-on-extremely-limited-vram-gpus) &emsp; => 10GB-12GB 显卡的显存 *(例如 NVIDIA GeForce RTX 2080Ti, 3060, 3080, 4070, 4080, 和 5070)*
  - [:ledger: 如何在低显存 GPU 上训练](#ledger-how-to-train-on-low-vram-gpus) &emsp; => 24GB 显存的显卡 *(例如 NVIDIA GeForce RTX 3090 和 4090)*
  - [:ledger: 如何在较大显存 GPU 上训练](#ledger-how-to-train-on-larger-vram-gpus) &emsp; => 32GB 显存的消费级 GPU *(例如 NVIDIA GeForce RTX 5090)* &emsp; 40GB-48GB 显存的专业级 GPU *(例如 NVIDIA A100-40GB, A800-40GB, L20, 和 RTX A6000).*
  - [:ledger: 如何在显存充足的 GPU 上训练](#ledger-how-to-train-on-sufficient-vram-gpus) &emsp; => ≥80GB 显存的专业级 GPU *(例如 NVIDIA A100-80GB, A800-80GB, H100, H800, H20-NVLink, 和 GB200).*
- [:mechanical_arm: 推理](#mechanical_arm-inference)
  - [:books: 推理相关文件](#books-related-file-for-inference)
  - [🤗 VLA-Adapter 权重 Checkpoint](#ckpts)
  - [:notebook: 如何进行评估](#evals)
- [🌈 成功率对比](#results)
- [📝 引用](#cite)
- [:heart: 致谢](#heart-acknowledgment)

<br/>

## :rocket: 快速开始


### VLA-Adapter 的 Conda 环境

```bash
# 创建并激活 conda 环境
conda create -n vla-adapter python=3.10.16 -y
conda activate vla-adapter
```

### 安装依赖

```bash
# 安装 PyTorch
# 请使用适合您机器的命令：https://pytorch.org/get-started/locally/
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

# 克隆 vla-adapter 仓库并使用 pip 安装以下载依赖
git clone https://github.com/OpenHelix-Team/VLA-Adapter.git
cd VLA-Adapter
pip install -e .

pip install packaging ninja
ninja --version; echo $?  # 验证 Ninja --> 应该返回退出代码 "0"

# 安装用于训练的 Flash Attention 2 (https://github.com/Dao-AILab/flash-attention)
pip install "flash-attn==2.5.5" --no-build-isolation
# 如果遇到困难，请先尝试 `pip cache remove flash_attn`，或者访问网站下载。
# (https://github.com/Dao-AILab/flash-attention/releases/tag/v2.5.5)
# 您可以根据 `nvidia-smi` 显示的 cuda 版本下载对应的 `.whl` 文件，
# 然后运行 `pip install flash_attn-2.5.5+cuXX...whl` 进行安装。
# 我们使用的是 `flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl` 文件。
```

<br/>
<br/>


## :pencil: 数据准备

### LIBERO 基准测试

- **(可选)**

克隆并安装 [LIBERO 仓库](https://github.com/Lifelong-Robot-Learning/LIBERO) 及所需包：

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt  # 在 vla-adapter 根目录下运行
```

下载我们在微调实验中使用的 [LIBERO 数据集](https://huggingface.co/datasets/openvla/modified_libero_rlds)，请运行以下命令。这将以 `RLDS` 格式下载 `Spatial`, `Object`, `Goal`, 和 `Long` 数据集，即 `libero_spatial_no_noops`, `libero_object_no_noops`, `libero_goal_no_noops`, `libero_10_no_noops`。（`"_no_noops"` 表示没有空操作动作，即过滤掉了动作接近于零的训练样本）。这些数据集总共需要 `~10GB` 内存。如果需要，请参阅[此处](https://github.com/openvla/openvla?tab=readme-ov-file#libero-setup)了解如何下载原始的非 RLDS 数据集。您可以使用这些数据集来微调 Prismatic-VLMs（基于 Qwen2.5-0.5B）或其他 VLM。

```bash
git clone git@hf.co:datasets/openvla/modified_libero_rlds
```

🌟 注意！以此方式下载的数据集需要删除名称中的 ``modified_`` 字样，以适应 - [:pushpin: 基准测试数据位置](#pushpin-benchmark-location) 的路径！！！

在使用 LIBERO 时，您可能会遇到类似 `AttributeError: 'NoneType' object has no attribute 'eglQueryString'` 的错误消息。您可以使用以下命令解决：

```bash
sudo apt-get update
sudo apt-get install libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev libglew-dev
```

### CALVIN 基准测试

- **(可选)**

```bash
git clone --recurse-submodules https://github.com/mees/calvin.git
export CALVIN_ROOT=$(pwd)/calvin
cd $CALVIN_ROOT

# `pyhash` 的安装在某些机器上可能会失败。如果失败，您可以通过降低 `setuptools` 版本来解决：`pip install setuptools==57.5.0`
sh install.sh
```

下载我们在微调实验中使用的 [CALVIN ABC→D 数据集](https://github.com/mees/calvin/tree/main/dataset)，请运行以下命令。

```bash
cd $CALVIN_ROOT/dataset
sh download_data.sh ABC
```

如果您想下载 RLDS 格式，可以访问 [此处](https://huggingface.co/datasets/zhouhongyi/calvin_abc_rlds) 下载。该数据集需要 `~50GB` 显存。

在使用 CALVIN 时，您可能会遇到类似 `AttributeError: 'NoneType' object has no attribute 'eglQueryString'` 的错误消息。您可以使用以下命令解决：

```bash
sudo apt-get update
sudo apt-get install libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev libglew-dev
```


### :video_game: 我们的依赖 

- **(包括 LIBERO 和 CALVIN)**

至此，环境已完全安装。如果您想确认环境是否正确，可以参考我们发布的 `our_envs.txt` 文件。


### :pushpin: 基准测试数据位置

下载好的数据集可以放在 `/data` 文件夹下。整体目录结构如下：

```
·
├── data
·   ├── libero
    │   ├── libero_10_no_noops
    │   │   └── 1.0.0  (包含一些 json 文件和 32 个 tfrecord 文件)
    │   ├── libero_goal_no_noops
    │   │   └── 1.0.0  (包含一些 json 文件和 16 个 tfrecord 文件)
    │   ├── libero_object_no_noops
    │   │   └── 1.0.0  (包含一些 json 文件和 32 个 tfrecord 文件)
    │   ├── libero_spatial_no_noops
    │   │   └── 1.0.0  (包含一些 json 文件和 16 个 tfrecord 文件)
    │
    ├── calvin_abc
    │   └── 1.0.0  (包含一些 json 文件，512 个训练 tfrecord 文件和 32 个验证 tfrecord 文件)
    │
    └── 其他基准测试 ...
```

<br/>
<br/>

## ⚓ VLM 主干网络 <a name="vlm"></a>
我们使用 `Prismatic-VLMs` 架构。由于文件较大，请从[此处](https://huggingface.co/Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b)下载。然后将其放入 `/pretrained_models` 文件夹中。文件结构为：

```
·
├── pretrained_models
·   ├── configs
    └── prism-qwen25-extra-dinosiglip-224px-0_5b
```


<br/>
<br/>

## :fire: 不同配置的训练

**我们为不同用户提供了不同的训练配置。您可以根据自己的 GPU 类型选择合适的训练配置。**

### :books: 训练相关文件
* `vla-scripts/finetune.py`: VLA 微调脚本


### :ledger: 如何在显存极度受限的 GPU 上训练

***=> 显存极度受限 (10GB-12GB 显卡) (例如 NVIDIA GeForce RTX 2080Ti, 3060, 3080, 4070, 4080, 和 5070)。***

>***关于 `batch_size`, `lora_rank`, `grad_accumulation_steps`, 和 `max_steps`。***

如果您的资源极其有限，可以将 `--batch_size 1` 和 `--lora_rank 64` 设置为微调参数，它只需要 `9.6GB` 显存。当然，`batch size = 1` 会导致梯度更新受极端值影响较大，Loss 收敛不稳定。在这种情况下，您可以修改 `grad_accumulation_steps` 参数来模拟类似的效果。例如，`--batch_size 1` 配合 `--grad_accumulation_steps 8` 的效果类似于 `--batch_size 8`，但训练速度会慢一些。这意味着您无法在 `10GB` 的卡上运行 [OpenVLA-OFT](https://github.com/moojink/openvla-oft) 模型，因为即使 `batch size = 1` 也需要 `25GB` 显存。幸运的是，您可以使用 VLA-Adapter。尽管 `batch size` 仍然较小，但您可以增加 `--max_steps` 来达到论文中报告性能。

>***关于 `vlm_path`。***

VLA-Adapter 中的 VLM 使用 Prismatic-VLMs 架构，LLM 主干为 `Qwen2.5-0.5B`。您可以从 https://huggingface.co/Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b 下载并将其放置在 `/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b`。

>***关于 `data_name`。***

使用下面的 vla-adapter 配置启动微调脚本。它可以在后台运行，运行进度可以在 `/logs` 文件夹中查看。您可以将 `libero_spatial_no_noops` 替换为 `libero_object_no_noops`, `libero_goal_no_noops`, 或 `libero_10_no_noops`。如果您使用的是 `CALVIN` 基准测试，则需要删除 `--data_root_dir` 中的 `\libero` 并将 `libero_spatial_no_noops` 替换为 `calvin_abc`。

>***关于 `use_pro_version`。***

此外，我们最近发布了 VLA-Adapter 的增强版 `Pro`。虽然其框架与原论文一致，但在实现上进行了增强，性能显著提高。**因此，我们强烈建议使用 Pro 版本！** `Pro` 版本的 `Policy` 大小为 `207MB`，训练速度几乎没有变化。`原始版本` 比 `pro 版本` 小近 `1GB`，仅需要 `8.6GB` 显存。您可以通过设置 `use_pro_version` 参数来选择是否使用 `Pro` 版本，即 `Pro` 版本设置 `--use_pro_version True`。

 ```bash
data_name=libero_spatial_no_noops

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
--vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
--config_file_path pretrained_models/configs \
--data_root_dir data/libero \
--dataset_name $data_name \
--run_root_dir outputs \
--use_film False \
--num_images_in_input 2 \
--use_proprio True \
--use_lora True \
--use_fz False \
--use_minivlm True \
--image_aug True \
--num_steps_before_decay 400000 \
--max_steps 400005 \
--save_freq 5000 \
--save_latest_checkpoint_only False \
--merge_lora_during_training True \
--batch_size 1 \
--grad_accumulation_steps 8 \
--learning_rate 2e-4 \
--lora_rank 64 \
--use_pro_version True \
--wandb_entity "YOUR_WANDB_ENTITY" \
--wandb_project "$data_name" \
--run_id_note VLA-Adapter--libero_spatial_no_noops--$current_time \
> logs/VLA-Adapter--libero_spatial_no_noops--$current_time.log 2>&1 &
```

请注意，获取的模型将存储在 `/outputs` 文件夹中。每个模型将占用近 `3GB` 的内存，因此您需要预留足够的空间。我们强烈建议您从 [VLA-Adapter HuggingFace](https://huggingface.co/VLA-Adapter) 获取我们训练好的模型，并将其放在此文件夹中进行推理。

<br/>

### :ledger: 如何在低显存 GPU 上训练

***=> 低显存 (24GB 显卡) (例如 NVIDIA GeForce RTX 3090 和 4090)。***

>***关于 `batch_size`, `lora_rank`, `grad_accumulation_steps`, 和 `max_steps`。***

如果您拥有此类设备，可以增加 `batch size` 和 `lora rank`：`--batch_size 4` 和 `--lora_rank 64`。这仅消耗近 `20GB`。这与我们论文中的 rank 一致。这意味着您无法在 `24GB` 的卡上运行 [OpenVLA-OFT](https://github.com/moojink/openvla-oft) 模型，因为即使 `batch size = 1` 也需要 `25GB` 显存。幸运的是，您可以使用 VLA-Adapter。尽管 `batch size` 仍然偏小，但您可以增加 `--max_steps` 来实现论文中报告的性能。

>***关于 `vlm_path`。***

VLA-Adapter 中的 VLM 使用 Prismatic-VLMs 架构，LLM 主干为 `Qwen2.5-0.5B`。您可以从 https://huggingface.co/Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b 下载并将其放置在 `/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b`。

>***关于 `data_name`。***

使用下面的 vla-adapter 配置启动微调脚本。它可以在后台运行，运行进度可以在 `/logs` 文件夹中查看。您可以将 `libero_spatial_no_noops` 替换为 `libero_object_no_noops`, `libero_goal_no_noops`, 或 `libero_10_no_noops`。如果您使用的是 `CALVIN` 基准测试，则需要删除 `--data_root_dir` 中的 `\libero` 并将 `libero_spatial_no_noops` 替换为 `calvin_abc`。

>***关于 `use_pro_version`。***

此外，我们最近发布了 VLA-Adapter 的增强版 `Pro`。虽然其框架与原论文一致，但在实现上进行了增强，性能显著提高。**因此，我们强烈建议使用 Pro 版本！** `Pro` 版本的 `Policy` 大小为 `207MB`，训练速度几乎没有变化。`原始版本` 比 `pro 版本` (1 batch) 小近 `1GB`，仅需要 `17.6GB` 显存。您可以通过设置 `use_pro_version` 参数来选择是否使用 `Pro` 版本，即 `Pro` 版本设置 `--use_pro_version True`。


 ```bash
data_name=libero_spatial_no_noops

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
--vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
--config_file_path pretrained_models/configs \
--data_root_dir data/libero \
--dataset_name $data_name \
--run_root_dir outputs \
--use_film False \
--num_images_in_input 2 \
--use_proprio True \
--use_lora True \
--use_fz False \
--use_minivlm True \
--image_aug True \
--num_steps_before_decay 200000 \
--max_steps 200005 \
--save_freq 5000 \
--save_latest_checkpoint_only False \
--merge_lora_during_training True \
--batch_size 4 \
--grad_accumulation_steps 4 \
--learning_rate 2e-4 \
--lora_rank 64 \
--use_pro_version True \
--wandb_entity "YOUR_WANDB_ENTITY" \
--wandb_project "$data_name" \
--run_id_note VLA-Adapter--libero_spatial_no_noops--$current_time \
> logs/VLA-Adapter--libero_spatial_no_noops--$current_time.log 2>&1 &
```

请注意，获取的模型将存储在 `/outputs` 文件夹中。每个模型将占用近 `3GB` 的内存，因此您需要预留足够的空间。我们强烈建议您从 [VLA-Adapter HuggingFace](https://huggingface.co/VLA-Adapter) 获取我们训练好的模型，并将其放在此文件夹中进行推理。



<br/>

### :ledger: 如何在较大显存 GPU 上训练

***=> 32GB 显存显卡 (例如 NVIDIA GeForce RTX 5090) <br/> => 40GB-48GB 显存的专业级 GPU (例如 NVIDIA A100-40GB, A800-40GB, L20, 和 RTX A6000)。***


>***关于 `batch_size`, `lora_rank`, `grad_accumulation_steps`, 和 `max_steps`。***

如果您拥有此类设备，可以增加 `batch size` 和 `lora rank`：`--batch_size 8` 和 `--lora_rank 64`。这仅占用近 `29GB` 显存。

>***关于 `vlm_path`。***

VLA-Adapter 中的 VLM 使用 Prismatic-VLMs 架构，LLM 主干为 `Qwen2.5-0.5B`。您可以从 https://huggingface.co/Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b 下载并将其放置在 `/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b`。

>***关于 `data_name`。***

使用下面的 vla-adapter 配置启动微调脚本。它可以在后台运行，运行进度可以在 `/logs` 文件夹中查看。您可以将 `libero_spatial_no_noops` 替换为 `libero_object_no_noops`, `libero_goal_no_noops`, 或 `libero_10_no_noops`。如果您使用的是 `CALVIN` 基准测试，则需要删除 `--data_root_dir` 中的 `\libero` 并将 `libero_spatial_no_noops` 替换为 `calvin_abc`。

通过此配置，您可以在 `LIBERO-Object` 基准测试上获得与论文相同的结果，仅需 `8 小时` 即可达到 `99.2%` 的成功率。`LIBERO-Spatial` 基准测试大约需要 10 小时的训练。但 `LIBERO-Long` 基准测试需要更长时间，因为其任务更长且更具挑战性，需要更多的训练步骤才能达到优异的性能。

>***关于 `use_pro_version`。***

此外，我们最近发布了 VLA-Adapter 的增强版 `Pro`。虽然其框架与原论文一致，但在实现上进行了增强，性能显著提高。**因此，我们强烈建议使用 Pro 版本！** `Pro` 版本的 `Policy` 大小为 `207MB`，训练速度几乎没有变化。`原始版本` 比 `pro 版本` (1 batch) 小近 `1GB`。您可以通过设置 `use_pro_version` 参数来选择是否使用 `Pro` 版本，即 `Pro` 版本设置 `--use_pro_version True`。

 ```bash
data_name=libero_spatial_no_noops

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
--vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
--config_file_path pretrained_models/configs \
--data_root_dir data/libero \
--dataset_name $data_name \
--run_root_dir outputs \
--use_film False \
--num_images_in_input 2 \
--use_proprio True \
--use_lora True \
--use_fz False \
--use_minivlm True \
--image_aug True \
--num_steps_before_decay 200000 \
--max_steps 200005 \
--save_freq 5000 \
--save_latest_checkpoint_only False \
--merge_lora_during_training True \
--batch_size 8 \
--grad_accumulation_steps 2 \
--learning_rate 2e-4 \
--lora_rank 64 \
--use_pro_version True \
--wandb_entity "YOUR_WANDB_ENTITY" \
--wandb_project "$data_name" \
--run_id_note VLA-Adapter--libero_spatial_no_noops--$current_time \
> logs/VLA-Adapter--libero_spatial_no_noops--$current_time.log 2>&1 &
```

请注意，获取的模型将存储在 `/outputs` 文件夹中。每个模型将占用近 `3GB` 的内存，因此您需要预留足够的空间。我们强烈建议您从 [VLA-Adapter HuggingFace](https://huggingface.co/VLA-Adapter) 获取我们训练好的模型，并将其放在此文件夹中进行推理。



<br/>

### :ledger: 如何在显存充足的 GPU 上训练

***=> 专业级 GPU (≥80GB 显存) (例如 NVIDIA A100-80GB, A800-80GB, H100, H800, H20-NVLink, 和 GB200)。***

>***关于 `batch_size`, `lora_rank`, `grad_accumulation_steps`, 和 `max_steps`。***

您可以通过将 `CUDA_VISIBLE_DEVICES` 中的 GPU 号更改为您欲使用的 GPU 序号，并将 `--nproc-per-node` 后的 GPU 数量更改为您欲使用的 GPU 数量，来使用 1 到 8 个 GPU 进行训练。在我们的论文中，我们使用了 4×H100 GPU 进行训练。在这种配置下，LIBERO 基准测试的四个任务套件分别需要：`Spatial`（仅五小时）、`Object`（不到一小时）、`Goal`（三小时）和 `Long`（半天）；`CALVIN` 基准测试（八小时）。

>***关于 `vlm_path`。***

VLA-Adapter 中的 VLM 使用 Prismatic-VLMs 架构，LLM 主干为 `Qwen2.5-0.5B`。您可以从 https://huggingface.co/Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b 下载并将其放置在 `/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b`。

>***关于 `data_name`。***

使用下面的 vla-adapter 配置启动微调脚本。它可以在后台运行，运行进度可以在 `/logs` 文件夹中查看。您可以将 `libero_spatial_no_noops` 替换为 `libero_object_no_noops`, `libero_goal_no_noops`, 或 `libero_10_no_noops`。如果您使用的是 `CALVIN` 基准测试，则需要删除 `--data_root_dir` 中的 `\libero` 并将 `libero_spatial_no_noops` 替换为 `calvin_abc`。


>***关于 `use_pro_version`。***

此外，我们最近发布了 VLA-Adapter 的增强版 `Pro`。虽然其框架与原论文一致，但在实现上进行了增强，性能显著提高。**因此，我们强烈建议使用 Pro 版本！** `Pro` 版本的 `Policy` 大小为 `207MB`，训练速度几乎没有变化。`原始版本` 比 `pro 版本` (1 batch) 小近 `1GB`。您可以通过设置 `use_pro_version` 参数来选择是否使用 `Pro` 版本，即 `Pro` 版本设置 `--use_pro_version True`。

```bash
data_name=libero_spatial_no_noops

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
--vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
--config_file_path pretrained_models/configs \
--data_root_dir data/libero \
--dataset_name $data_name \
--run_root_dir outputs \
--use_film False \
--num_images_in_input 2 \
--use_proprio True \
--use_lora True \
--use_fz False \
--use_minivlm True \
--image_aug True \
--num_steps_before_decay 150000 \
--max_steps 150005 \
--save_freq 5000 \
--save_latest_checkpoint_only False \
--merge_lora_during_training True \
--batch_size 16 \
--grad_accumulation_steps 1 \
--learning_rate 2e-4 \
--lora_rank 64 \
--use_pro_version True \
--wandb_entity "YOUR_WANDB_ENTITY" \
--wandb_project "$data_name" \
--run_id_note VLA-Adapter--spatial--$current_time \
> logs/VLA-Adapter--spatial--$current_time.log 2>&1 &
```

请注意，获取的模型将存储在 `/outputs` 文件夹中。每个模型将占用近 `3GB` 的内存，因此您需要预留足够的空间。我们强烈建议您从 [VLA-Adapter HuggingFace](https://huggingface.co/VLA-Adapter) 获取我们训练好的模型，并将其放在此文件夹中进行推理。

## :mechanical_arm: 推理

### :books: 推理相关文件
* `experiments/robot/libero/`: LIBERO 评估文件
  * `run_libero_eval.py`: LIBERO 评估脚本
  * `libero_utils.py`: LIBERO 评估工具
* `experiments/robot/`: 通用评估工具文件
  * `openvla_utils.py`: VLA 特定评估工具
  * `robot_utils.py`: 其他评估工具

<br/>

### 🤗 VLA-Adapter 权重 Checkpoint <a name="ckpts"></a>
我们使用微调 bridge 范式在四个 LIBERO 任务套件上独立微调了 `Qwen2.5-0.5B`：`LIBERO-Spatial`, `LIBERO-Object`, `LIBERO-Goal`, 和 `LIBERO-Long`。
针对 LIBERO 的四个 VLA-Adapter checkpoint 可在 Hugging Face 上获取：
* [VLA-Adapter/LIBERO-Spatial](https://huggingface.co/VLA-Adapter/LIBERO-Spatial) 
* [VLA-Adapter/LIBERO-Object](https://huggingface.co/VLA-Adapter/LIBERO-Object)
* [VLA-Adapter/LIBERO-Goal](https://huggingface.co/VLA-Adapter/LIBERO-Goal)
* [VLA-Adapter/LIBERO-Long](https://huggingface.co/VLA-Adapter/LIBERO-Long)

此外，我们还提供了 `Pro` 版本，我们使用了 `4*H100` GPU 进行训练，参数为 `--batch_size 16`, `--lora rank 64`, 和 `--max_steps 100000`。Pro 版本的 checkpoint 如下：

* [VLA-Adapter/LIBERO-Spatial-Pro](https://huggingface.co/VLA-Adapter/LIBERO-Spatial-Pro) `(97.8 -> 99.6)`
* [VLA-Adapter/LIBERO-Object-Pro](https://huggingface.co/VLA-Adapter/LIBERO-Object-Pro) `(99.2 -> 99.6)`
* [VLA-Adapter/LIBERO-Goal-Pro](https://huggingface.co/VLA-Adapter/LIBERO-Goal-Pro) `(97.2 -> 98.2)`
* [VLA-Adapter/LIBERO-Long-Pro](https://huggingface.co/VLA-Adapter/LIBERO-Long-Pro) `(95.0 -> 96.4)`
* [VLA-Adapter/CALVIN-ABC-Pro](https://huggingface.co/VLA-Adapter/CALVIN-ABC-Pro) `(4.42 -> 4.50)`

这些文件需要放置在 `/output` 文件夹中。如果您训练了自己的模型，它也将存储在这里。随后的评估代码将调用此文件夹中的模型进行推理。


<br/>


### :notebook: 如何评估 <a name="evals"></a>

**我们强烈建议您使用我们开源的性能更强的 `Pro` 版本模型。** 要使用其中一个 checkpoint 开始评估，请运行以下命令之一。每个命令都会自动下载上面列出的对应 checkpoint。如果您想使用模型的原始版本，只需将 `--use_pro_version` 参数调整为 `False`，并向 `--pretrained_checkpoint` 参数传递模型原始版本所在的路径即可。最后，推理结果将显示在 `/eval_logs` 文件夹中，推理视频将显示在 `/rollouts/vla-adapter` 文件夹中。


```bash
# 启动 LIBERO-Spatial-Pro 评估 (后台运行)
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --pretrained_checkpoint outputs/LIBERO-Spatial-Pro \
  --task_suite_name libero_spatial \
  --use_pro_version True \
  > eval_logs/Spatial--chkpt.log 2>&1 &


# 启动 LIBERO-Object-Pro 评估 (后台运行)
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --pretrained_checkpoint outputs/LIBERO-Object-Pro \
  --task_suite_name libero_object \
  --use_pro_version True \
  > eval_logs/Object--chkpt.log 2>&1 &


# 启动 LIBERO-Goal-Pro 评估 (后台运行)
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --pretrained_checkpoint outputs/LIBERO-Goal-Pro \
  --task_suite_name libero_goal \
  --use_pro_version True \
  > eval_logs/Goal--chkpt.log 2>&1 &


# 启动 LIBERO-Long-Pro (LIBERO-10) 评估 (后台运行)
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --pretrained_checkpoint outputs/LIBERO-long-Pro \
  --task_suite_name libero_10 \
  --use_pro_version True \
  > eval_logs/Long--chkpt.log 2>&1 &


# 启动 CALVIN ABC→D-Pro 评估 (后台运行)
CUDA_VISIBLE_DEVICES=0 python vla-scripts/evaluate_calvin.py \
  --pretrained_checkpoint outputs/CALVIN-ABC-Pro \
  > eval_logs/CALVIN--ABC.log 2>&1 &
```

如果您想获取推理 **吞吐量 (throughput)**，可以在 `run_libero_eval.py` 文件中运行。您可以在 `334--345 行` 前后分别添加 `start = time.time()` 和 `end = time.time()` 并计算两者之差。这个差值即为生成 `8 个 chunk` 的时间。据此您可以得到推理吞吐量。我们进行了多次测量，取平均值为 `0.036s`。

<br/>

## 🌈 成功率对比 <a name="results"></a>

我们所有的结果都是在 `H100` 上推理得出的。您可以在 [HF](https://huggingface.co/VLA-Adapter) 上发布的模型中找到推理 `log` 文件进行查看。默认情况下，LIBERO 评估脚本运行 500 次试验（10 个任务 x 每个任务 50 个 episode），CALVIN 评估脚本运行 1,000 个任务序列。请尽可能使用同一张卡进行训练和推理。**注意，如果您使用与 H100 不同的 GPU，结果可能会略有波动。** 此现象在 OpenVLA-OFT 的 Readme 文件中也有提到。

### 在 LIBERO 基准测试上的表现。

<b><i>XX</i></b> 代表最佳表现，<b>XX</b> 代表次佳表现，<i><u>XX*</u></i> 代表第三佳表现。
<table>
  <tr>
   <td><strong>LIBERO</strong></td>  <td><strong>方法</strong></td>
   <td><strong>参数规模</strong></td>  <td><strong>Spatial</strong></td>
   <td><strong>Object</strong></td>  <td><strong>Goal</strong></td>
   <td><strong>Long</strong></td>  <td><strong>平均</strong></td>
  </tr>

  <tr><td rowspan="10">大规模 (Large-scale)</td><td>FlowVLA (Zhong et al., 2025)</td>
   <td>8.5B</td><td>93.2</td><td>95.0</td><td>91.6</td><td>72.6</td><td>88.1</td></tr>

  <tr><td>UnifiedVLA (Wang et al., 2025)</td>
   <td>8.5B</td><td>95.4</td><td><i><u>98.8*</u></i></td><td> 93.6 </td><td>94.0 </td><td>95.5</td></tr>

  <tr><td>OpenVLA (Kim et al., 2024)</td>
   <td>7B</td><td>84.7</td><td>88.4</td><td>79.2</td><td>53.7</td><td>76.5</td></tr>

  <tr><td>OpenVLA-OFT (Kim et al., 2025)</td>
   <td>7B</td><td><i><u>97.6*</u></i></td><td>98.4</td><td><b>97.9</b></td><td><i><u>94.5*</u></i></td><td><i><u>97.1*</u></i></td></tr>

  <tr><td>UniVLA (Bu et al., 2025)</td>
   <td>7B</td><td>96.5</td><td> 96.8</td><td> 95.6 </td><td>92.0 </td><td>95.2</td></tr>

  <tr><td>CoT-VLA (Zhao et al., 2025)</td>
   <td>7B</td><td>87.5 </td><td>91.6 </td><td>87.6</td><td> 69.0</td><td> 81.1</td></tr>

  <tr><td>WorldVLA (Cen et al., 2025)</td>
   <td>7B</td><td>87.6</td><td> 96.2</td><td> 83.4</td><td> 60.0</td><td> 81.8</td></tr>

  <tr><td>TraceVLA (Zheng et al., 2025)</td>
   <td>7B</td><td>84.6</td><td> 85.2</td><td> 75.1</td><td> 54.1</td><td> 74.8</td></tr>

  <tr><td>MolmoAct (Lee et al., 2025)</td>
   <td>7B</td><td>87.0</td><td> 95.4 </td><td>87.6</td><td> 77.2 </td><td>86.6</td></tr>

  <tr><td>ThinkAct (Huang et al., 2025)</td>
   <td>7B</td><td>88.3 </td><td>91.4</td><td> 87.1</td><td> 70.9</td><td> 84.4</td></tr>

  <tr><td rowspan="7">小规模 (Small-scale)</td><td>4D-VLA (Zhang et al., 2025)</td>
   <td>4B</td><td>88.9</td><td> 95.2</td><td> 90.9</td><td> 79.1 </td><td>88.6</td></tr>

  <tr><td>SpatialVLA (Qu et al., 2025)</td>
   <td>4B</td><td>88.2</td><td> 89.9</td><td> 78.6</td><td> 55.5 </td><td>78.1</td></tr>

  <tr><td>π0 (Black et al., 2024)</td>
   <td>3B</td><td>96.8</td><td><i><u>98.8*</u></i></td><td>95.8</td><td> 85.2</td><td> 94.2</td></tr>

  <tr><td>π0-FAST (Pertsch et al., 2025)</td>
   <td>3B</td><td>96.4</td><td> 96.8 </td><td>88.6</td><td> 60.2</td><td> 85.5</td></tr>

  <tr><td>NORA (Hung et al., 2025)</td>
   <td>3B</td><td>92.2 </td><td>95.4 </td><td>89.4</td><td> 74.6 </td><td>87.9</td></tr>

  <tr><td>SmolVLA (Shukor et al., 2025)</td>
   <td>2.2B</td><td>93.0</td><td> 94.0 </td><td>91.0</td><td> 77.0 </td><td>88.8</td></tr>

  <tr><td>GR00T N1 (NVIDIA et al., 2025)</td>
   <td>2B</td><td>94.4</td><td> 97.6 </td><td>93.0 </td><td>90.6</td><td> 93.9</td></tr>

  <tr><td rowspan="5">极小规模 (Tiny-scale)</td><td>Seer (Tian et al., 2025)</td>
   <td>0.57B</td><td>-</td><td> - </td><td>- </td><td>78.7</td><td> 78.7</td></tr>

  <tr><td>VLA-OS (Gao et al., 2025)</td>
   <td>0.5B</td><td>87.0 </td><td>96.5</td><td> 92.7 </td><td>66.0</td><td> 85.6</td></tr>

  <tr><td>Diffusion Policy (Chi et al., 2023)</td>
   <td>-</td><td>78.3</td><td> 92.5</td><td> 68.3 </td><td>50.5 </td><td>72.4</td></tr>

  <tr><td><b>VLA-Adapter (本研究)</b></td>
   <td><b>0.5B</b></td><td><b>97.8</b></td><td><b>99.2</b></td><td><i><u>97.2*</u></i></td><td> <b>95.0 </b></td><td><b>97.3</b></td></tr>

  <tr><td><b>VLA-Adapter-Pro (本研究)</b></td>
   <td><b>0.5B</b></td><td><b><i>99.6</i></b></td><td><b><i>99.6</i></b> </td><td><b><i>98.2</i></b></td><td><b><i>96.4</i></b></td><td><b><i>98.5</i></b></td></tr>
  
</table>

### 在 CALVIN ABC→D 基准测试上的表现。

<b><i>XX</i></b> 代表最佳表现，<b>XX</b> 代表次佳表现，<i><u>XX*</u></i> 代表第三佳表现。

<table>
  <tr>
   <td><strong>CALVIN</strong></td>  <td><strong>方法</strong></td>
   <td><strong>参数规模</strong></td>  <td><strong>1</strong></td>
   <td><strong>2</strong></td>  <td><strong>3</strong></td>
   <td><strong>4</strong></td>  <td><strong>5</strong></td> <td><strong>平均长度</strong></td>
  </tr>

  <tr><td rowspan="8">大规模</td><td>UniVLA (Bu et al., 2025) </td><td>7B </td><td>95.5 </td><td>85.8 </td><td>75.4</td><td> 66.9 </td><td>56.5 </td><td>3.80</td></tr>

  <tr><td>OpenVLA (Kim et al., 2024) </td><td> 7B</td><td> 91.3</td><td> 77.8 </td><td>62.0 </td><td>52.1 </td><td>43.5</td><td> 3.27</td></tr>

  <tr><td>OpenVLA-OFT (Kim et al., 2025)</td><td> 7B</td><td> 96.3</td><td> 89.1 </td><td>82.4</td><td> 75.8</td><td> 66.5</td><td> 4.10</td></tr>

  <tr><td>VLAS (Zhao et al., 2025b) </td><td> 7B</td><td> 87.2 </td><td>64.2</td><td> 40.9 </td><td>28.1</td><td> 19.6 </td><td>2.40</td></tr>

  <tr><td>LCB (Shentu et al., 2024) </td><td> 7B</td><td> 73.6 </td><td>50.2 </td><td>28.5 </td><td>16.0 </td><td>9.9 </td><td>1.78</td></tr>

  <tr><td>RoboDual (Bu et al., 2024a) </td><td> 7B</td><td> 94.4</td><td> 82.7</td><td> 72.1</td><td> 62.4 </td><td>54.4</td><td> 3.66</td></tr>

  <tr><td>OpenHelix (Cui et al., 2025)  </td><td> 7B</td><td> <i><u>97.1*</u></i> </td><td>91.4 </td><td>82.8</td><td> 72.6</td><td> 64.1 </td><td>4.08</td></tr>

  <tr><td>ReconVLA (Song et al., 2025c)  </td><td> 7B</td><td> 95.6 </td><td>87.6 </td><td>76.9</td><td> 69.3</td><td> 64.1 </td><td>3.95</td></tr>

  <tr><td rowspan="4">小规模</td><td>DeeR (Yue et al., 2024) </td><td> 3B</td><td> 86.2</td><td> 70.1 </td><td>51.8</td><td> 41.5</td><td> 30.4 </td><td>2.82</td></tr>

  <tr><td>RoboFlamingo (Li et al., 2024b) </td><td> 3B</td><td> 82.4 </td><td>61.9</td><td> 46.6 </td><td>33.1</td><td> 23.5</td><td> 2.48</td></tr>

  <tr><td>VPP (Hu et al., 2025)</td><td>  1.5B</td><td>  95.7</td><td>  91.2</td><td>  <i><u>86.3*</u></i></td><td>  <i><u>81.0*</u></i></td><td>  <i><u>75.0*</u></i></td><td>  <i><u>4.33*</u></i></td></tr>

  <tr><td>SuSIE (Black et al., 2024)</td><td>1.3B</td><td> 87.0</td><td> 69.0</td><td> 49.0 </td><td>38.0</td><td> 26.0</td><td> 2.69</td></tr>

  <tr><td rowspan="5">极小规模</td><td>Seer-Large (Tian et al., 2025)</td><td>0.57B</td><td> 96.3 </td><td><i><u>91.6*</u></i></td><td> 86.1 </td><td>80.3 </td><td>74.0</td><td> 4.28</td></tr>

  <tr><td>MoDE (Reuss et al., 2025) </td><td> 0.44B </td><td>96.2</td><td> 88.9</td><td> 81.1</td><td> 71.8 </td><td>63.5 </td><td>4.01</td></tr>

  <tr><td>Seer (Tian et al., 2025) </td><td> 0.32B</td><td> 94.4 </td><td>87.2 </td><td>79.9 </td><td>72.2 </td><td>64.3</td><td> 3.98</td></tr>

  <tr><td><b>VLA-Adapter (本研究)</b></td>
   <td><b>0.5B</b></td><td><b><i>99.1</i></b> </td><td><b>94.6</b> </td><td><b>88.8</b></td><td> <b>82.8</b> </td><td><b>76.5</b> </td><td><b>4.42</b></td></tr>

  <tr><td><b>VLA-Adapter-Pro (本研究)</b></td>
   <td><b>0.5B</b></td><td><b>98.5</b></td><td><b><i>95.0</i></b> </td><td><b><i>90.5</i></b></td><td><b><i>85.3</i></b></td><td><b><i>80.0</i></b></td><td><b><i>4.50</i></b></td></tr>
  
</table>


<br/>


## 📝 引用 <a name="cite"></a>

### 🫶 如果您觉得本论文、模型或代码对您有帮助，请引用我们的论文，感谢您对 VLA-Adapter 的支持！

```bibtex
@article{wang2025vlaadapter,
  author={Wang, Yihao and Ding, Pengxiang and Li, Lingxiao and Cui, Can and Ge, Zirui and Tong, Xinyang and Song, Wenxuan and Zhao, Han and Zhao, Wei and Hou, Pengxu and Huang, Siteng and Tang, Yifan and Wang, Wenhui and Zhang, Ru and Liu, Jianyi and Wang, Donglin},
  title={VLA-Adapter: An Effective Paradigm for Tiny-Scale Vision-Language-Action Model},
  journal={arXiv preprint arXiv:2509.09372},
  year={2025}
}
```

## :heart: 致谢

感谢 [OpenVLA-OFT](https://github.com/moojink/openvla-oft), [MiniVLA](https://github.com/Stanford-ILIAD/openvla-mini), 和 [RoboDual](https://github.com/OpenDriveLab/RoboDual) 的开源工作！

## 🌟 Star History

<a href="https://www.star-history.com/#OpenHelix-Team/VLA-Adapter&Date">
  <img src="https://api.star-history.com/svg?repos=OpenHelix-Team/VLA-Adapter&type=Date" width="400" height="250" />
</a>
