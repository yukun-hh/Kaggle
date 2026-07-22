# %%
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# 加载微调后的完整模型（main.py 保存的 final_model 目录）
model_path = "./output/final_model"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model.to(device)
model.eval()
print("模型加载完成:", model_path)

# %%
PROMPT_PREFIX = (
    "你是一名材料科学助手，请根据问题给出准确、专业的回答。\n"
    "问题：{q}\n"
    "回答："
)

EOS_TOKEN = tokenizer.eos_token


def build_prompt(question):
    return PROMPT_PREFIX.format(q=question)


@torch.no_grad()
def chat(
    question,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    system_prompt=None,
):
    prefix = system_prompt if system_prompt else PROMPT_PREFIX
    input_text = prefix.format(q=question)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


def interactive():
    """交互式问答（输入 quit/exit/q 退出）"""
    print("=" * 60)
    print("材料科学问答助手（输入 quit / exit / q 退出）")
    print("=" * 60)
    while True:
        try:
            q = input("\n问题：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            break
        print("回答：", chat(q))
    print("已退出。")


# %%
# 从原始 csv 取问题做批量回测，对比模型回答与标准答案
def batch_eval(sample_n=10):
    df = pd.read_csv("样本收集.csv", encoding="utf-8").dropna(subset=["问题", "回答"]).reset_index(drop=True)
    df["问题"] = df["问题"].astype(str).str.replace("\u3000", " ").str.replace(r"\s+", " ", regex=True).str.strip()
    df["回答"] = df["回答"].astype(str).str.replace("\u3000", " ").str.replace(r"\s+", " ", regex=True).str.strip()

    sample_n = min(sample_n, len(df))
    for i in tqdm(range(sample_n), desc="回测中"):
        q = str(df.loc[i, "问题"])
        a = str(df.loc[i, "回答"])
        pred = chat(q)
        print(f"\n{'='*60}")
        print(f"[{i+1}] Q: {q}")
        print(f"    标准答案: {a}")
        print(f"    模型回答: {pred}")


# %%
if __name__ == "__main__":
    import sys

    # 用法：
    #   python test.py                 -> 默认进入交互式问答
    #   python test.py --eval          -> 跑 csv 批量回测
    #   python test.py "你的问题"      -> 单次问答
    if len(sys.argv) > 1:
        if sys.argv[1] == "--eval":
            batch_eval(sample_n=10)
        else:
            q = " ".join(sys.argv[1:])
            print("回答：", chat(q))
    else:
        interactive()
