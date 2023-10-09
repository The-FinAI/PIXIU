<p align="center" width="100%">
<img src="https://i.postimg.cc/xTpWgq3L/pixiu-logo.png"  width="100%" height="100%">
</p>

<div>
<div align="left">
    <a target='_blank'>Qianqian Xie<sup>1</sup></span>&emsp;
    <a target='_blank'>Weiguang Han<sup>1</sup></span>&emsp;
    <a target='_blank'>Xiao Zhang<sup>2</sup></a>&emsp;
    <a target='_blank'>Ruoyu Xiang<sup>7</sup></a>&emsp;
    <a target='_blank'>Gang Hu<sup>5</sup></a>&emsp;
    <a target='_blank'>Ke Qin<sup>5</sup></a>&emsp;
    <a target='_blank'>Duanyu Feng<sup>3</sup></a>&emsp;
    <a target='_blank'>Yongfu Dai<sup>3</sup></a>&emsp;
    <a target='_blank'>Hao Wang<sup>3</sup></a>&emsp;
    <a target='_blank'>Yanzhao Lai<sup>4</sup></a>&emsp;
    <a target='_blank'>Min Peng<sup>1</sup></a>&emsp;
    <a href='https://warrington.ufl.edu/directory/person/12693/' target='_blank'>Alejandro Lopez-Lira<sup>6</sup></a>&emsp;
    <a href='https://jimin.chancefocus.com/' target='_blank'>Jimin Huang*<sup>,8</sup></a>
</div>
<div>
<div align="left">
    <sup>1</sup>Wuhan University&emsp;
    <sup>2</sup>Sun Yat-Sen University&emsp;
    <sup>3</sup>Sichuan University&emsp;
    <sup>4</sup>Southwest Jiaotong University&emsp;
    <sup>5</sup>Yunan University&emsp;
    <sup>6</sup>University of Florida&emsp;
    <sup>7</sup>New York University&emsp;
  <sup>8</sup>ChanceFocus AMC.
</div>
<div align="left">
    <img src='https://i.postimg.cc/CLtkBwz7/57-EDDD9-FB0-DF712-F3-AB627163-C2-1-EF15655-13-FCA.png' alt='Wuhan University Logo' height='100px'>&emsp;
    <img src='https://i.postimg.cc/C1XnZNK1/Sun-Yat-sen-University-Logo.png' alt='Sun Yat-Sen University Logo' height='100px'>&emsp;
    <img src='https://i.postimg.cc/DfB8jxV1/ynu.png' alt='Yunnan University Logo' height='100px'>&emsp;
    <img src='https://i.postimg.cc/vTHJdYxN/NYU-Logo.png' alt='New York University' height='100px'>&emsp;
    <img src='https://i.postimg.cc/NjKhDkGY/DFAF986-CCD6529-E52-D7830-F180-D-C37-C7-DEE-4340.png' alt='Sichuan University Logo' height='100px'>&emsp;
    <img src='https://i.postimg.cc/k5WpYj0r/SWJTULogo.png' alt='Southwest Jiaotong University Logo' height='100px'>&emsp;
    <img src='https://i.postimg.cc/XY1s2RHD/University-of-Florida-Logo-1536x864.jpg' alt='University of Florida Logo' height='100px'>&emsp;
    <img src='https://i.postimg.cc/xTsgsrqN/logo11.png' alt='ChanceFocus AMC Logo' height='100px'>
</div>




-----------------

![](https://img.shields.io/badge/pixiu-v0.1-gold)
![](https://black.readthedocs.io/en/stable/_static/license.svg)
[![Discord](https://img.shields.io/discord/1146837080798933112)](https://discord.gg/HRWpUmKB)

[Pixiu Paper](https://arxiv.org/abs/2306.05443) | [FLARE Leaderboard](https://huggingface.co/spaces/ChanceFocus/FLARE)

**Disclaimer**

This repository and its contents are provided for **academic and educational purposes only**. None of the material constitutes financial, legal, or investment advice. No warranties, express or implied, are offered regarding the accuracy, completeness, or utility of the content. The authors and contributors are not responsible for any errors, omissions, or any consequences arising from the use of the information herein. Users should exercise their own judgment and consult professionals before making any financial, legal, or investment decisions. The use of the software and information contained in this repository is entirely at the user's own risk.

**By using or accessing the information in this repository, you agree to indemnify, defend, and hold harmless the authors, contributors, and any affiliated organizations or persons from any and all claims or damages.**



**Checkpoints:** 

- [FinMA v0.1 (Full 7B version)](https://huggingface.co/ChanceFocus/finma-7b-full)

**Languages**

- [English](README.md)
- [Chinese](README.zh.md)

**Evaluations** (More details on FLARE section):

- [FLARE (flare-zh-afqmc)](https://huggingface.co/datasets/ChanceFocus/flare-zh-afqmc)
- [FLARE (flare-zh-stocka)](https://huggingface.co/datasets/ChanceFocus/flare-zh-stocka)
- [FLARE (flare-zh-corpus)](https://huggingface.co/datasets/ChanceFocus/flare-zh-corpus)
- [FLARE (flare-zh-fineval)](https://huggingface.co/datasets/ChanceFocus/flare-zh-fineval)
- [FLARE (flare-zh-fe)](https://huggingface.co/datasets/ChanceFocus/flare-zh-fe)
- [FLARE (flare-zh-nl)](https://huggingface.co/datasets/ChanceFocus/flare-zh-nl)
- [FLARE (flare-zh-nl2)](https://huggingface.co/datasets/ChanceFocus/flare-zh-nl2)
- [FLARE (flare-zh-nsp)](https://huggingface.co/datasets/ChanceFocus/flare-zh-nsp)
- [FLARE (flare-zh-re)](https://huggingface.co/datasets/ChanceFocus/flare-zh-re)
- [FLARE (flare-zh-stockb)](https://huggingface.co/datasets/ChanceFocus/flare-zh-stockb)


## Overview

**FLARE_ZH** is a cornerstone initiative focusing on the Chinese financial domain, FLARE_ZH aims to bolster the progress, refinement, and assessment of Large Language Models (LLMs) tailored specifically for Chinese financial contexts. As a vital segment of the broader PIXIU endeavor, FLARE_ZH stands as a testament to the commitment in harnessing the capabilities of LLMs, ensuring that financial professionals and enthusiasts in the Chinese-speaking world have top-tier linguistic tools at their disposal.

### Key Features

- **Open resources**: PIXIU openly provides the financial LLM, instruction tuning data, and datasets included in the evaluation benchmark to encourage open research and transparency.
- **Multi-task**: The instruction tuning data and benchmark in PIXIU cover a diverse set of financial tasks.
- **Multi-modality**: PIXIU's instruction tuning data and benchmark consist of multi-modality financial data, including time series data from the stock movement prediction task. It covers various types of financial texts, including reports, news articles, tweets, and regulatory filings.
- **Diversity**: Unlike previous benchmarks focusing mainly on financial NLP tasks, PIXIU's evaluation benchmark includes critical financial prediction tasks aligned with real-world scenarios, making it more challenging.

---

## FLARE_ZH: Financial Language Understanding and Prediction Evaluation Benchmark

In this section, we provide a detailed performance analysis of FinMA compared to other leading models, including ChatGPT, GPT-4, lince-zero et al. For this analysis, we've chosen a range of tasks and metrics that span various aspects of financial Natural Language Processing and financial prediction.

### Tasks

| Data                  | Task                            | Raw     | Data Types                       | Modalities        | License           | Paper |
| --------------------- | ------------------------------- | ------- | -------------------------------- | ----------------- | ----------------- | ----- |
| AFQMC                 | semantic matching               | 42,511  | question data, chat              | text              | Apache-2.0        | [1]   |
| corpus                | semantic matching               | 120,000 | question data, chat              | text              | Public            | [2]   |
| stockA                | stock classification            | 40,963  | news, historical prices          | text, time series | Public            | [3]   |
| Fineval               | multiple-choice                 | 4,661   | financial exam                   | text              | Apache-2.0        | [4]   |
| NL                    | news classification             | 10,000  | news articles                    | text              | Public            | [5]   |
| NL2                   | news classification             | 10,000  | news articles                    | text              | Public            | [5]   |
| NSP                   | negative news judgment          | 6,000   | news, social media text          | text              | Public            | [5]   |
| RE                    | relationship identification     | 12,670  | news, entity pair                | text              | Public            | [5]   |
| FE                    | sentiment analysis              | 10,000  | financial social media text      | text              | Public            | [5]   |
| stockB                | sentiment analysis              | 270,601 | financial social media text      | text              | Apache-2.0        | [6]   |


1. Xu L, Hu H, Zhang X, et al. CLUE: A Chinese language understanding evaluation benchmark[J]. arXiv preprint arXiv:2004.05986, 2020.
2. Jing Chen, Qingcai Chen, Xin Liu, Haijun Yang, Daohe Lu, and Buzhou Tang. 2018. The BQ Corpus: A Large-scale Domain-specific Chinese Corpus For Sentence Semantic Equivalence Identification. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 4946–4951, Brussels, Belgium. Association for Computational Linguistics.
3. Jinan Zou, Haiyao Cao, Lingqiao Liu, Yuhao Lin, Ehsan Abbasnejad, and Javen Qinfeng Shi. 2022. Astock: A New Dataset and Automated Stock Trading based on Stock-specific News Analyzing Model. In Proceedings of the Fourth Workshop on Financial Technology and Natural Language Processing (FinNLP), pages 178–186, Abu Dhabi, United Arab Emirates (Hybrid). Association for Computational Linguistics.
4. Zhang L, Cai W, Liu Z, et al. FinEval: A Chinese Financial Domain Knowledge Evaluation Benchmark for Large Language Models[J]. arxiv preprint arxiv:2308.09975, 2023.
5. Lu D, Liang J, Xu Y, et al. BBT-Fin: Comprehensive Construction of Chinese Financial Domain Pre-trained Language Model, Corpus and Benchmark[J]. arxiv preprint arxiv:2302.09432, 2023.
6. https://huggingface.co/datasets/kuroneko5943/stock11

### Evaluation

#### Preparation
##### Locally install
```bash
git clone https://github.com/chancefocus/PIXIU.git --recursive
cd PIXIU
pip install -r requirements.txt
cd PIXIU/src/financial-evaluation
pip install -e .[multilingual]
```
##### Docker image
```bash
sudo bash scripts/docker_run.sh
```
Above command starts a docker container, you can modify `docker_run.sh` to fit your environment. We provide pre-built image by running `sudo docker pull tothemoon/pixiu:latest`

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --network host \
    --env https_proxy=$https_proxy \
    --env http_proxy=$http_proxy \
    --env all_proxy=$all_proxy \
    --env HF_HOME=$hf_home \
    -it [--rm] \
    --name pixiu \
    -v $pixiu_path:$pixiu_path \
    -v $hf_home:$hf_home \
    -v $ssh_pub_key:/root/.ssh/authorized_keys \
    -w $workdir \
    $docker_user/pixiu:$tag \
    [--sshd_port 2201 --cmd "echo 'Hello, world!' && /bin/bash"]
```
Arguments explain:
- `[]` means ignoreable arguments
- `HF_HOME`: huggingface cache dir
- `sshd_port`: sshd port of the container, you can run `ssh -i private_key -p $sshd_port root@$ip` to connect to the container, default to 22001
- `--rm`: remove the container when exit container (ie.`CTRL + D`)

#### Automated Task Assessment
Before evaluation, please download [BART checkpoint](https://drive.google.com/u/0/uc?id=1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m&export=download) to `src/metrics/BARTScore/bart_score.pth`.

For automated evaluation, please follow these instructions:

1. Huggingface Transformer

   To evaluate a model hosted on the HuggingFace Hub (for instance, finma-7b-full), use this command:

```bash
python eval.py \
    --model "hf-causal-llama" \
    --model_args "use_accelerate=True,pretrained=chancefocus/finma-7b-full,tokenizer=chancefocus/finma-7b-full,use_fast=False" \
    --tasks "flare_ner,flare_sm_acl,flare_fpb"
```

More details can be found in the [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness) documentation.

2. Commercial APIs


Please note, for tasks such as NER, the automated evaluation is based on a specific pattern. This might fail to extract relevant information in zero-shot settings, resulting in relatively lower performance compared to previous human-annotated results.

```bash
export OPENAI_API_SECRET_KEY=YOUR_KEY_HERE
python eval.py \
    --model gpt-4 \
    --tasks flare_ner,flare_sm_acl,flare_fpb
```


## License

PIXIU is licensed under [MIT]. For more details, please see the [MIT](LICENSE) file.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=chancefocus/pixiu&type=Date)](https://star-history.com/#chancefocus/pixiu&Date)



-----------------

![](https://img.shields.io/badge/pixiu-v0.1-gold)
![](https://black.readthedocs.io/en/stable/_static/license.svg)
[![Discord](https://img.shields.io/discord/1146837080798933112)](https://discord.gg/HRWpUmKB)

[Pixiu Paper](https://arxiv.org/abs/2306.05443) | [FLARE Leaderboard](https://huggingface.co/spaces/ChanceFocus/FLARE)

**免责声明**

本资料库及其内容仅用于**学术和教育目的**。所有资料均不构成金融、法律或投资建议。不对内容的准确性、完整性或实用性提供任何明示或暗示的保证。作者和撰稿人不对任何错误、遗漏或因使用本网站信息而产生的任何后果负责。用户在做出任何财务、法律或投资决定之前，应自行判断并咨询专业人士。使用本资料库所含软件和信息的风险完全由用户自行承担。

**使用或访问本资源库中的信息，即表示您同意对作者、撰稿人以及任何附属组织或个人的任何及所有索赔或损害进行赔偿、为其辩护并使其免受损害。**




**检查点:** 

- [FinMA v0.1 (Full 7B version)](https://huggingface.co/ChanceFocus/finma-7b-full)

**语言**

- [英文](README.md)
- [中文](README.zh.md)

**评估** (更多详情，请参阅FLARE部分):

- [FLARE (flare-zh-afqmc)](https://huggingface.co/datasets/ChanceFocus/flare-zh-afqmc)
- [FLARE (flare-zh-stocka)](https://huggingface.co/datasets/ChanceFocus/flare-zh-stocka)
- [FLARE (flare-zh-corpus)](https://huggingface.co/datasets/ChanceFocus/flare-zh-corpus)
- [FLARE (flare-zh-fineval)](https://huggingface.co/datasets/ChanceFocus/flare-zh-fineval)
- [FLARE (flare-zh-fe)](https://huggingface.co/datasets/ChanceFocus/flare-zh-fe)
- [FLARE (flare-zh-nl)](https://huggingface.co/datasets/ChanceFocus/flare-zh-nl)
- [FLARE (flare-zh-nl2)](https://huggingface.co/datasets/ChanceFocus/flare-zh-nl2)
- [FLARE (flare-zh-nsp)](https://huggingface.co/datasets/ChanceFocus/flare-zh-nsp)
- [FLARE (flare-zh-re)](https://huggingface.co/datasets/ChanceFocus/flare-zh-re)
- [FLARE (flare-zh-stockb)](https://huggingface.co/datasets/ChanceFocus/flare-zh-stockb)

## 概述

**FLARE_ZH** 是一项专注于中文金融领域的基石计划，旨在促进专为中文金融环境定制的大型语言模型（LLMs）的进展、完善和评估。FLARE_ZH 是 PIXIU 更大范围工作的一个重要部分，证明了我们在利用 LLMs 能力方面的承诺，确保中文世界的金融专业人士和爱好者拥有顶级的语言工具。

### 主要特征

- **公开资源**: PIXIU 公开提供财务 LLM、教学调整数据和评估基准中的数据集，以鼓励公开研究和透明度。
- **多任务**: PIXIU 中的指令调整数据和基准涵盖了一系列不同的金融任务。
- **多模态**: PIXIU 的指令调整数据和基准由多模态金融数据组成，包括股票走势预测任务的时间序列数据。它涵盖各种类型的金融文本，包括报告、新闻报道、推特和监管文件。
- **多样性**: 与以往主要侧重于金融 NLP 任务的基准不同，PIXIU 的评估基准包括与真实世界场景相一致的关键金融预测任务，因此更具挑战性。

---

## FLARE_ZH: 金融语言理解和预测评估基准

在本节中，我们将提供 FinMA 与其他领先模型（包括 ChatGPT、GPT-4、ince-zero 等）相比的详细性能分析。为了进行分析，我们选择了一系列任务和指标，涵盖了金融自然语言处理和金融预测的各个方面。

### 任务

| 数据                  | 任务类型                       | 原始数据| 数据类型                         | 模式              | 许可证            | 论文  |
| --------------------- | ------------------------------- | ------- | -------------------------------- | ----------------- | ----------------- | ----- |
| AFQMC                 | 语义匹配                        | 42,511  | 提问数据, 对话                   | 文本               | Apache-2.0        | [1]   |
| corpus                | 语义匹配                        | 120,000 | 提问数据, 对话                   | 文本               | Public            | [2]   |
| stockA                | 股票分类                        | 40,963  | 新闻, 历史价格                   | 文本, 时间序列     | Public            | [3]   |
| Fineval               | 多项选择                        | 4,661   | 金融考试                         | 文本               | Apache-2.0        | [4]   |
| NL                    | 新闻分类                        | 10,000  | 新闻报道                         | 文本               | Public            | [5]   |
| NL2                   | 新闻分类                        | 10,000  | 新闻报道                         | 文本               | Public            | [5]   |
| NSP                   | 负面新闻判断                    | 6,000   | 新闻、社交媒体文本               | 文本               | Public            | [5]   |
| RE                    | 关系识别                        | 12,670  | 新闻、实体对                     | 文本               | Public            | [5]   |
| FE                    | 情感分析                        | 10,000  | 金融社交媒体文本                 | 文本               | Public            | [5]   |
| stockB                | 情感分析                        | 270,601 | 金融社交媒体文本                 | 文本               | Apache-2.0        | [6]   |


1. Xu L, Hu H, Zhang X, et al. CLUE: A Chinese language understanding evaluation benchmark[J]. arXiv preprint arXiv:2004.05986, 2020.
2. Jing Chen, Qingcai Chen, Xin Liu, Haijun Yang, Daohe Lu, and Buzhou Tang. 2018. The BQ Corpus: A Large-scale Domain-specific Chinese Corpus For Sentence Semantic Equivalence Identification. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 4946–4951, Brussels, Belgium. Association for Computational Linguistics.
3. Jinan Zou, Haiyao Cao, Lingqiao Liu, Yuhao Lin, Ehsan Abbasnejad, and Javen Qinfeng Shi. 2022. Astock: A New Dataset and Automated Stock Trading based on Stock-specific News Analyzing Model. In Proceedings of the Fourth Workshop on Financial Technology and Natural Language Processing (FinNLP), pages 178–186, Abu Dhabi, United Arab Emirates (Hybrid). Association for Computational Linguistics.
4. Zhang L, Cai W, Liu Z, et al. FinEval: A Chinese Financial Domain Knowledge Evaluation Benchmark for Large Language Models[J]. arxiv preprint arxiv:2308.09975, 2023.
5. Lu D, Liang J, Xu Y, et al. BBT-Fin: Comprehensive Construction of Chinese Financial Domain Pre-trained Language Model, Corpus and Benchmark[J]. arxiv preprint arxiv:2302.09432, 2023.
6. https://huggingface.co/datasets/kuroneko5943/stock11

### 评估

#### 准备工作
##### 本地安装
```bash
git clone https://github.com/chancefocus/PIXIU.git --recursive
cd PIXIU
pip install -r requirements.txt
cd PIXIU/src/financial-evaluation
pip install -e .[multilingual]
```
##### Docker 镜像
```bash
sudo bash scripts/docker_run.sh
```
以上命令会启动一个 docker 容器，你可以根据自己的环境修改 `docker_run.sh`。我们通过运行 `sudo docker pull tothemoon/pixiu:latest` 来提供预编译镜像。

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --network host \
    --env https_proxy=$https_proxy \
    --env http_proxy=$http_proxy \
    --env all_proxy=$all_proxy \
    --env HF_HOME=$hf_home \
    -it [--rm] \
    --name pixiu \
    -v $pixiu_path:$pixiu_path \
    -v $hf_home:$hf_home \
    -v $ssh_pub_key:/root/.ssh/authorized_keys \
    -w $workdir \
    $docker_user/pixiu:$tag \
    [--sshd_port 2201 --cmd "echo 'Hello, world!' && /bin/bash"]
```
参数说明:
- `[]` 表示可忽略的参数
- `HF_HOME`: huggingface 缓存目录
- `sshd_port`: 容器的 sshd 端口，可以运行 `ssh -i private_key -p $sshd_port root@$ip` 来连接容器，默认为 22001
- `--rm`: 退出容器时移除容器（即 `CTRL + D`）

#### 自动化任务评估
在评估前, 请下载 [punto de control BART](https://drive.google.com/u/0/uc?id=1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m&export=download) 到 `src/metrics/BARTScore/bart_score.pth`.

如需进行自动评估，请按照以下说明操作：

1. Transformador Huggingface

   要评估 HuggingFace Hub 上托管的模型（例如，finma-7b-full），请使用此命令：

```bash
python eval.py \
    --model "hf-causal-llama" \
    --model_args "use_accelerate=True,pretrained=chancefocus/finma-7b-full,tokenizer=chancefocus/finma-7b-full,use_fast=False" \
    --tasks "flare_ner,flare_sm_acl,flare_fpb"
```

更多详情，请参阅 [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness) 文档。

2. 商用接口


请注意，对于 NER 等任务，自动评估是基于特定模式进行的。这可能无法提取零镜头设置中的相关信息，导致性能相对低于之前的人工标注结果。

```bash
export OPENAI_API_SECRET_KEY=YOUR_KEY_HERE
python eval.py \
    --model gpt-4 \
    --tasks flare_ner,flare_sm_acl,flare_fpb
```

---


## 许可证

PIXIU 采用 [MIT] 许可。有关详细信息，请参阅 [MIT](LICENSE) 文件。

## 星标历史

[![Star History Chart](https://api.star-history.com/svg?repos=chancefocus/pixiu&type=Date)](https://star-history.com/#chancefocus/pixiu&Date)

