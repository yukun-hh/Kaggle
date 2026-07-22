# 部署指南 —— Qwen2.5-1.5B 材料科学问答模型（bf16 全量微调）

本项目基于 `Qwen/Qwen2.5-1.5B-Instruct`，在 `样本收集.csv`（材料科学问答对）上做因果语言模型微调，单卡 16GB 显存可训练。

## 一、环境与依赖

```bash
# Python 3.10+
pip install torch torchvision torchaudio
pip install transformers>=4.44.0 accelerate
pip install pandas tqdm
```

验证 CUDA：

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> 国内加速下载：在脚本顶部已设置 `os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"`，无需额外操作。
> 若想更快可安装 `hf_transfer` 并设 `HF_HUB_ENABLE_HF_TRANSFER=1`。

## 二、文件结构

```
gpt-chemistry/
├── 样本收集.csv      # 训练数据（列：问题、回答）
├── main.py           # 训练脚本（bf16 全量微调）
├── test.py           # 推理回测脚本
├── download.py       # 模型预热下载脚本
├── deploy.md         # 本指南
└── output/           # 训练产物（自动生成）
    ├── model.pth     # state_dict
    └── best_model/   # 完整模型 + tokenizer（可直接 from_pretrained）
```

## 三、方案说明

| 项目         | 配置                              |
| ------------ | --------------------------------- |
| 基座模型     | Qwen/Qwen2.5-1.5B-Instruct        |
| 精度         | bfloat16                           |
| 微调方式     | 全量参数微调                       |
| 梯度检查点   | 开启                               |
| 显存占用     | 约 8~10GB（单卡 16GB 富余）       |
| 训练参数量   | 1.5B 全部                          |

> 1.5B 用 bf16 全量微调约需 8~10GB 显存，16GB 卡无需量化即可跑通。相比 QLoRA 全量微调收敛更稳、效果更好。

## 四、下载模型（可选预热）

```bash
python download.py
```

`download.py` 会带进度条把模型拉到本地缓存，之后 `main.py` / `test.py` 自动命中缓存，不再重复下载。

## 五、训练流程

1. 确认 `样本收集.csv` 与 `main.py` 同目录，列名为 `问题`、`回答`。
2. 启动训练：

   ```bash
   python main.py
   ```

3. 训练产物保存在 `./output/best_model/`，可直接 `from_pretrained` 加载。
4. 终端 `tqdm` 实时显示 `loss`、`avg_loss`、`lr`，每个 epoch 结束保存当前最优。

## 六、推理 / 测试

```bash
python test.py
```

`test.py` 自动从 csv 取前 10 条问题回测，对比「标准答案」与「模型回答」。

交互式问答：

```python
from test import chat
print(chat("MOF 是什么材料？"))
```

## 七、常见问题

- **OOM（显存不足）**：降低 `batch_size` 为 2 或 1 → 减小 `max_length` 至 384 → 换 Qwen2.5-0.5B。
- **下载慢**：确认已换源（脚本内已设 hf-mirror），或安装 `hf_transfer` 加速。
- **中文乱码**：确认 `样本收集.csv` 为 UTF-8 编码。
- **生成重复**：推理时提高 `repetition_penalty`(1.1~1.3) 或降低 `temperature`。
- **训练不收敛**：全量微调学习率建议 `1e-5 ~ 3e-5`，默认 `2e-5`。
- **想用 LoRA**：1.5B 全量微调显存足够，一般无需 LoRA；若要省显存可参考 PEFT 文档注入 LoRA。
