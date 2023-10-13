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

**📢 Update (Date: 09-22-2023)**

🚀 We're thrilled to announce that our paper, "PIXIU: A Comprehensive Benchmark, Instruction Dataset and Large Language Model for Finance", has been accepted by NeurIPS 2023 Track Datasets and Benchmarks!

**📢 Update (Date: 10-08-2023)**

🌏 We're proud to share that the enhanced versions of FLARE, which now support both Chinese and Spanish!

**Checkpoints:** 

- [FinMA v0.1 (NLP 7B version)](https://huggingface.co/ChanceFocus/finma-7b-nlp)
- [FinMA v0.1 (Full 7B version)](https://huggingface.co/ChanceFocus/finma-7b-full)

**Languages**

- [English](README.md)
- [Spainish](README.es.md)
- [Chinese](README.zh.md)

**Evaluations**:

- [English Evaluation Datasets](https://huggingface.co/collections/ChanceFocus/flare-evaluation-datasets-english-6529286a147d9119a64689c0) (More details on FLARE section)
- [Spanish Evaluation Datasets](https://huggingface.co/collections/ChanceFocus/flare-evaluation-datasets-spanish-652929c34f8fe1bea9cd5a66)
- [Chinese Evaluation Datasets](https://huggingface.co/collections/ChanceFocus/flare-evalution-datasets-chinese-65292963a8cd8847517204a2)

## Overview

Welcome to the **PIXIU** project! This project is designed to support the development, fine-tuning, and evaluation of Large Language Models (LLMs) in the financial domain. PIXIU is a significant step towards understanding and harnessing the power of LLMs in the financial domain.

### Structure of the Repository

The repository is organized into several key components, each serving a unique purpose in the financial NLP pipeline:

- **FLARE**: Our Financial Language Understanding and Prediction Evaluation Benchmark. FLARE serves as the evaluation suite for financial LLMs, with a focus on understanding and prediction tasks across various financial contexts.
- **FIT**: Our Financial Instruction Dataset. FIT is a multi-task and multi-modal instruction dataset specifically tailored for financial tasks. It serves as the training ground for fine-tuning LLMs for these tasks.

- **FinMA**: Our Financial Large Language Model (LLM). FinMA is the core of our project, providing the learning and prediction power for our financial tasks.

### Key Features

- **Open resources**: PIXIU openly provides the financial LLM, instruction tuning data, and datasets included in the evaluation benchmark to encourage open research and transparency.
  
- **Multi-task**: The instruction tuning data and benchmark in PIXIU cover a diverse set of financial tasks, including four financial NLP tasks and one financial prediction task.
- **Multi-modality**: PIXIU's instruction tuning data and benchmark consist of multi-modality financial data, including time series data from the stock movement prediction task. It covers various types of financial texts, including reports, news articles, tweets, and regulatory filings.
- **Diversity**: Unlike previous benchmarks focusing mainly on financial NLP tasks, PIXIU's evaluation benchmark includes critical financial prediction tasks aligned with real-world scenarios, making it more challenging.

---

## FLARE 2.0: Financial Language Understanding and Prediction Evaluation Benchmark

In this section, we provide a detailed performance analysis of FinMA compared to other leading models, including ChatGPT, GPT-4, and BloombergGPT et al. For this analysis, we've chosen a range of tasks and metrics that span various aspects of financial Natural Language Processing and financial prediction. All model results of FLARE can be found on our [leaderboard](https://huggingface.co/spaces/ChanceFocus/FLARE)!

### Tasks

| Data                  | Task                             | Raw    | Data Types                | Modalities        | License         | Paper |
| --------------------- | -------------------------------- | ------ | ------------------------- | ----------------- | --------------- | ----- |
| FPB                   | sentiment analysis               | 4,845  | news                      | text              | CC BY-SA 3.0    | [1]   |
| FiQA-SA               | sentiment analysis               | 1,173  | news headlines, tweets    | text              | Public          | [2]   |
| FOMC                  | hawkish-dovish classification    | 496    | FOMC transcripts          | text              | CC BY-NC 4.0    | [3]   |
| Headlines             | news headline classification     | 11,412 | news headlines            | text              | CC BY-SA 3.0    | [4]   |
| NER                   | named entity recognition         | 1,366  | financial agreements      | text              | CC BY-SA 3.0    | [5]  |
| Finer Ord             | named entity recognition         | 1,080  | news articles             | text              | CC BY-NC 4.0    | [6]  |
| FinQA                 | question answering               | 8,281  | earnings reports          | text, table       | MIT License     | [7]  |
| ConvFinQA             | multi-turn question answering    | 1,490  | earnings reports          | text, table       | MIT License     | [8]  |
| ECTSUM                | text summarization               | 495    | earning call transcipts   | text              | Public          | [9]  |
| EDTSUM                | text summarization               | 2000   | news articles             | text              | Public          | [10]  |
| German                | credit scoring                   | 1000   | credit records            | table             | CC BY 4.0       | [11]  |
| Australian            | credit scoring                   | 690    | credit records            | table             | CC BY 4.0       | [12]  |
| BigData22             | stock movement prediction        | 7,164  | tweets, historical prices | text, time series | Public          | [13]  |
| ACL18                 | stock movement prediction        | 27,053 | tweets, historical prices | text, time series | MIT License     | [14]  |
| CIKM18                | stock movement prediction        | 4,967  | tweets, historical prices | text, time series | Public          | [15]  |

1. Pekka Malo, Ankur Sinha, Pekka Korhonen, Jyrki Wallenius, and Pyry Takala. 2014. Good debt or bad debt: Detecting semantic orientations in economic texts. Journal of the Association for Information Science and Technology 65, 4 (2014), 782–796.
2. Macedo Maia, Siegfried Handschuh, André Freitas, Brian Davis, Ross McDermott, Manel Zarrouk, and Alexandra Balahur. 2018. Www’18 open challenge: financial opinion mining and question answering. In Companion proceedings of the the web conference 2018. 1941–1942
3. Agam Shah, Suvan Paturi, and Sudheer Chava. 2023. [Trillion Dollar Words: A New Financial Dataset, Task & Market Analysis](https://aclanthology.org/2023.acl-long.368). In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 6664–6679, Toronto, Canada. Association for Computational Linguistics.
4. Ankur Sinha and Tanmay Khandait. 2021. Impact of news on the commodity market: Dataset and results. In Advances in Information and Communication: Proceedings of the 2021 Future of Information and Communication Conference (FICC), Volume 2. Springer, 589–601
5. Julio Cesar Salinas Alvarado, Karin Verspoor, and Timothy Baldwin. 2015. Domain adaption of named entity recognition to support credit risk assessment. In Proceedings of the Australasian Language Technology Association Workshop 2015. 84–90.
6. Shah A, Vithani R, Gullapalli A, et al. Finer: Financial named entity recognition dataset and weak-supervision model[J]. arXiv preprint arXiv:2302.11157, 2023.
7. Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan R Routledge, et al . 2021. FinQA: A Dataset of Numerical Reasoning over Financial Data. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. 3697–3711.
8. Zhiyu Chen, Shiyang Li, Charese Smiley, Zhiqiang Ma, Sameena Shah, and William Yang Wang. 2022. ConvFinQA: Exploring the Chain of Numerical Reasoning in Conversational Finance Question Answering. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 6279–6292, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.
9. Rajdeep Mukherjee, Abhinav Bohra, Akash Banerjee, Soumya Sharma, Manjunath Hegde, Afreen Shaikh, Shivani Shrivastava, Koustuv Dasgupta, Niloy Ganguly, Saptarshi Ghosh, and Pawan Goyal. 2022. [ECTSum: A New Benchmark Dataset For Bullet Point Summarization of Long Earnings Call Transcripts](https://aclanthology.org/2022.emnlp-main.748). In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, pages 10893–10906, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.
10. Zhihan Zhou, Liqian Ma, and Han Liu. 2021. [Trade the Event: Corporate Events Detection for News-Based Event-Driven Trading](https://aclanthology.org/2021.findings-acl.186). In *Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021*, pages 2114–2124, Online. Association for Computational Linguistics.
11. Hofmann,Hans. (1994). Statlog (German Credit Data). UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77.
12. Quinlan,Ross. Statlog (Australian Credit Approval). UCI Machine Learning Repository. https://doi.org/10.24432/C59012.
13. Yejun Soun, Jaemin Yoo, Minyong Cho, Jihyeong Jeon, and U Kang. 2022. Accurate Stock Movement Prediction with Self-supervised Learning from Sparse Noisy Tweets. In 2022 IEEE International Conference on Big Data (Big Data). IEEE, 1691–1700.
14. Yumo Xu and Shay B Cohen. 2018. Stock movement prediction from tweets and historical prices. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 1970–1979.
15. Huizhe Wu, Wei Zhang, Weiwei Shen, and Jun Wang. 2018. Hybrid deep sequential modeling for social text-driven stock prediction. In Proceedings of the 27th ACM international conference on information and knowledge management. 1627–1630.

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

3. Self-Hosted Evaluation

To run inference backend:

```bash
bash scripts/run_interface.sh
```

Please adjust run_interface.sh according to your environment requirements.

To evaluate:

```bash
python data/*/evaluate.py
```

### Create new tasks

Creating a new task for FLARE involves creating a Huggingface dataset and implementing the task in a Python file. This guide walks you through each step of setting up a new task using the FLARE framework.

#### Creating your dataset in Huggingface

Your dataset should be created in the following format:

```python
{
    "query": "...",
    "answer": "...",
    "text": "..."
}
```

In this format:

- `query`: Combination of your prompt and text
- `answer`: Your label

For **Multi-turn** tasks (such as )

For **Classification** tasks (such as [FPB (flare_fpb)](https://huggingface.co/datasets/ChanceFocus/flare-fpb)), additional keys should be defined:

- `choices`: Set of labels
- `gold`: Index of the correct label in choices (Start from 0)

For **Sequential Labeling** tasks (such as [Finer Ord (flare_finer_ord)](https://huggingface.co/datasets/ChanceFocus/flare-finer-ord)), additional keys should be defined:

- `label`: List of token labels

- `token`: List of tokens

For **Extractive Summarization** tasks (such as [ECTSUM (flare_ectsum)](https://huggingface.co/datasets/ChanceFocus/flare-ectsum)), additional keys should be defined:

- `label`: List of sentence labels

For **abstractive Summarization** and **Question Answering** tasks (such as [EDTSUM (flare_edtsum)](https://huggingface.co/datasets/ChanceFocus/flare-edtsum)), no additional keys should be defined

#### Implementing the task

Once your dataset is ready, you can start implementing your task. Your task should be defined within a new class in flare.py or any other Python file located within the tasks directory.

To cater to a range of tasks, we offer several specialized base classes, including `Classification`, `SequentialLabeling`, `RelationExtraction`, `ExtractiveSummarization`, `AbstractiveSummarization` and `QA`.

For instance, if you are embarking on a classification task, you can directly leverage our `Classification` base class. This class allows for efficient and intuitive task creation. To better demonstrate this, let's delve into an example of crafting a task named FLARE-FPB using the `Classification` base class:

```python
class FlareFPB(Classification):
    DATASET_PATH = "flare-fpb"
```

And that's it! Once you've created your task class, the next step is to register it in the `src/tasks/__init__.py` file. To do this, add a new line following the format `"task_name": module.ClassName`. Here is how it's done:

```python
TASK_REGISTRY = {
    "flare_fpb": flare.FPB,
    "your_new_task": your_module.YourTask,  # This is where you add your task
}
```

#### Predefined task metrics

| Task                                     | Metric                                 | Illustration                                                 |
| ---------------------------------------- | -------------------------------------- | ------------------------------------------------------------ |
| Classification                           | Accuracy                               | This metric represents the ratio of correctly predicted observations to total observations. It is calculated as (True Positives + True Negatives) / Total Observations. |
| Classification                           | F1 Score                               | The F1 Score represents the harmonic mean of precision and recall, thereby creating an equilibrium between these two factors. It proves particularly useful in scenarios where one factor bears more significance than the other. The score ranges from 0 to 1, with 1 signifying perfect precision and recall, and 0 indicating the worst case. Furthermore, we provide both 'weighted' and 'macro' versions of the F1 score. |
| Classification                           | Missing Ratio                          | This metric calculates the proportion of responses where no options from the given choices in the task are returned. |
| Classification                           | Matthews Correlation Coefficient (MCC) | The MCC is a metric that assesses the quality of binary classifications, producing a score ranging from -1 to +1. A score of +1 signifies perfect prediction, 0 denotes a prediction no better than random chance, and -1 indicates a completely inverse prediction. |
| Sequential Labeling                      | F1 score                               | In the context of Sequential Labeling tasks, we utilize the F1 Score as computed by the `seqeval` library, a robust entity-level evaluation metric. This metric mandates an exact match of both the entity's span and type between the predicted and ground truth entities for a correct evaluation. True Positives (TP) represent correctly predicted entities, False Positives (FP) denote incorrectly predicted entities or entities with mismatched spans/types, and False Negatives (FN) signify missed entities from the ground truth. Precision, recall, and F1-score are then computed using these quantities, with the F1 Score representing the harmonic mean of precision and recall. |
| Sequential Labeling                      | Label F1 score                         | This metric evaluates model performance based solely on the correctness of the labels predicted, without considering entity spans. |
| Relation Extraction                      | Precision                              | Precision measures the proportion of correctly predicted relations out of all predicted relations. It is calculated as the number of True Positives (TP) divided by the sum of True Positives and False Positives (FP). |
| Relation Extraction                      | Recall                                 | Recall measures the proportion of correctly predicted relations out of all actual relations. It is calculated as the number of True Positives (TP) divided by the sum of True Positives and False Negatives (FN). |
| Relation Extraction                      | F1 score                               | The F1 Score is the harmonic mean of precision and recall, and it provides a balance between these two metrics. The F1 Score is at its best at 1 (perfect precision and recall) and worst at 0. |
| Extractive and Abstractive Summarization | Rouge-N                                | This measures the overlap of N-grams (a contiguous sequence of N items from a given sample of text) between the system-generated summary and the reference summary. 'N' can be 1, 2, or more, with ROUGE-1 and ROUGE-2 being commonly used to assess unigram and bigram overlaps respectively. |
| Extractive and Abstractive Summarization | Rouge-L                                | This metric evaluates the longest common subsequence (LCS) between the system and the reference summaries. LCS takes into account sentence level structure similarity naturally and identifies longest co-occurring in-sequence n-grams automatically. |
| Question Answering                       | EmACC                                  | EMACC assesses the exact match between the model-generated response and the reference answer. In other words, the model-generated response is considered correct only if it matches the reference answer exactly, word-for-word. |

>  Additionally, you can determine if the labels should be lowercased during the matching process by specifying `LOWER_CASE` in your class definition. This is pertinent since labels are matched based on their appearance in the generated output. For tasks like examinations where the labels are a specific set of capitalized letters such as 'A', 'B', 'C', this should typically be set to False.

---

## FIT: Financial Instruction Dataset

Our instruction dataset is uniquely tailored for the domain-specific LLM, FinMA. This dataset has been meticulously assembled to fine-tune our model on a diverse range of financial tasks. It features publicly available multi-task and multi-modal data derived from the multiple open released financial datasets.

The dataset is multi-faceted, featuring tasks including sentiment analysis, news headline classification, named entity recognition, question answering, and stock movement prediction. It covers both textual and time-series data modalities, offering a rich variety of financial data. The task specific instruction prompts for each task have been carefully degined by domain experts.

### Modality and Prompts

The table below summarizes the different tasks, their corresponding modalities, text types, and examples of the instructions used for each task:

| **Task**                     | **Modalities**    | **Text Types**        | **Instructions Examples**                                    |
| ---------------------------- | ----------------- | --------------------- | ------------------------------------------------------------ |
| Sentiment Analysis           | Text              | news headlines,tweets | "Analyze the sentiment of this statement extracted from a financial news article.Provide your answer as either negative, positive or neutral. For instance, 'The company's stocks plummeted following the scandal.' would be classified as negative." |
| News Headline Classification | Text              | News Headlines        | "Consider whether the headline mentions the price of gold. Is there a Price or Not in the gold commodity market indicated in the news headline? Please answer Yes or No." |
| Named Entity Recognition     | Text              | financial agreements  | "In the sentences extracted from financial agreements in U.S. SEC filings, identify the named entities that represent a person ('PER'), an organization ('ORG'), or a location ('LOC'). The required answer format is: 'entity name, entity type'. For instance, in 'Elon Musk, CEO of SpaceX, announced the launch from Cape Canaveral.', the entities would be: 'Elon Musk, PER; SpaceX, ORG; Cape Canaveral, LOC'" |
| Question Answering           | Text              | earnings reports      | "In the context of this series of interconnected finance-related queries and the additional information provided by the pretext, table data, and post text from a company's financial filings, please provide a response to the final question. This may require extracting information from the context and performing mathematical calculations. Please take into account the information provided in the preceding questions and their answers when formulating your response:" |
| Stock Movement Prediction    | Text, Time-Series | tweets, Stock Prices  | "Analyze the information and social media posts to determine if the closing price of *\{tid\}* will ascend or descend at *\{point\}*. Please respond with either Rise or Fall." |

### Dataset Statistics

The dataset contains a vast amount of instruction data samples (136K), allowing FinMA to capture the nuances of the diverse financial tasks. The table below provides the statistical details of the instruction dataset:

| Data      | Task                         | Raw    | Instruction | Data Types                | Modalities        | License      | Original Paper |
| --------- | ---------------------------- | ------ | ----------- | ------------------------- | ----------------- | ------------ | -------------- |
| FPB       | sentiment analysis           | 4,845  | 48,450      | news                      | text              | CC BY-SA 3.0 | [1]            |
| FiQA-SA   | sentiment analysis           | 1,173  | 11,730      | news headlines, tweets    | text              | Public       | [2]            |
| Headline  | news headline classification | 11,412 | 11,412      | news headlines            | text              | CC BY-SA 3.0 | [3]            |
| NER       | named entity recognition     | 1,366  | 13,660      | financial agreements      | text              | CC BY-SA 3.0 | [4]            |
| FinQA     | question answering           | 8,281  | 8,281       | earnings reports          | text, table       | MIT License  | [5]            |
| ConvFinQA | question answering           | 3,892  | 3,892       | earnings reports          | text, table       | MIT License  | [6]            |
| BigData22 | stock movement prediction    | 7,164  | 7,164       | tweets, historical prices | text, time series | Public       | [7]            |
| ACL18     | stock movement prediction    | 27,053 | 27,053      | tweets, historical prices | text, time series | MIT License  | [8]            |
| CIKM18    | stock movement prediction    | 4,967  | 4,967       | tweets, historical prices | text, time series | Public       | [9]            |

1. Pekka Malo, Ankur Sinha, Pekka Korhonen, Jyrki Wallenius, and Pyry Takala. 2014. Good debt or bad debt: Detecting semantic orientations in economic texts. Journal of the Association for Information Science and Technology 65, 4 (2014), 782–796.
2. Macedo Maia, Siegfried Handschuh, André Freitas, Brian Davis, Ross McDermott, Manel Zarrouk, and Alexandra Balahur. 2018. Www’18 open challenge: financial opinion mining and question answering. In Companion proceedings of the the web conference 2018. 1941–1942
3. Ankur Sinha and Tanmay Khandait. 2021. Impact of news on the commodity market: Dataset and results. In Advances in Information and Communication: Proceedings of the 2021 Future of Information and Communication Conference (FICC), Volume 2. Springer, 589–601
4. Julio Cesar Salinas Alvarado, Karin Verspoor, and Timothy Baldwin. 2015. Domain adaption of named entity recognition to support credit risk assessment. In Proceedings of the Australasian Language Technology Association Workshop 2015. 84–90.
5. Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan R Routledge, et al . 2021. FinQA: A Dataset of Numerical Reasoning over Financial Data. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. 3697–3711.
6. Zhiyu Chen, Shiyang Li, Charese Smiley, Zhiqiang Ma, Sameena Shah, and William Yang Wang. 2022. Convfinqa: Exploring the chain of numerical reasoning in conversational finance question answering. arXiv preprint arXiv:2210.03849 (2022).
7. Yejun Soun, Jaemin Yoo, Minyong Cho, Jihyeong Jeon, and U Kang. 2022. Accurate Stock Movement Prediction with Self-supervised Learning from Sparse Noisy Tweets. In 2022 IEEE International Conference on Big Data (Big Data). IEEE, 1691–1700.
8. Yumo Xu and Shay B Cohen. 2018. Stock movement prediction from tweets and historical prices. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 1970–1979.
9. Huizhe Wu, Wei Zhang, Weiwei Shen, and Jun Wang. 2018. Hybrid deep sequential modeling for social text-driven stock prediction. In Proceedings of the 27th ACM international conference on information and knowledge management. 1627–1630.

### Generating Datasets for FIT

When you are working with the Financial Instruction Dataset (FIT), it's crucial to follow the prescribed format for training and testing models.

The format should look like this:

```json
{
    "id": "unique id",
    "conversations": [
        {
            "from": "human",
            "value": "Your prompt and text"
        },
        {
            "from": "agent",
            "value": "Your answer"
        }
    ],
    "text": "Text to be classified",
    "label": "Your label"
}
```

Here's what each field means:

- "id": a unique identifier for each example in your dataset.
- "conversations": a list of conversation turns. Each turn is represented as a dictionary, with "from" representing the speaker, and "value" representing the text spoken in the turn.
- "text": the text to be classified.
- "label": the ground truth label for the text.


The first turn in the "conversations" list should always be from "human", and contain your prompt and the text. The second turn should be from "agent", and contain your answer.

---

## FinMA v0.1: Financial Large Language Model

We are pleased to introduce the first version of FinMA, including three models FinMA-7B, FinMA-7B-full, FinMA-30B, fine-tuned on LLaMA 7B and LLaMA-30B. FinMA-7B and FinMA-30B are trained with the NLP instruction data, while FinMA-7B-full is trained with the full instruction data from FIT covering both NLP and prediction tasks. 

FinMA v0.1 is now available on [Huggingface](https://huggingface.co/ChanceFocus/finma-7b-nlp) for public use. We look forward to the valuable contributions that this initial version will make to the financial NLP field and encourage users to apply it to various financial tasks and scenarios. We also invite feedback and shared experiences to help improve future versions.

### How to fine-tune a new large language model using PIXIU based on FIT?

Coming soon.

---


## Citation

If you use PIXIU in your work, please cite our paper.

```
@misc{xie2023pixiu,
      title={PIXIU: A Large Language Model, Instruction Data and Evaluation Benchmark for Finance}, 
      author={Qianqian Xie and Weiguang Han and Xiao Zhang and Yanzhao Lai and Min Peng and Alejandro Lopez-Lira and Jimin Huang},
      year={2023},
      eprint={2306.05443},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License

PIXIU is licensed under [MIT]. For more details, please see the [MIT](LICENSE) file.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=chancefocus/pixiu&type=Date)](https://star-history.com/#chancefocus/pixiu&Date)

