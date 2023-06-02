# PIXIU: A Comprehensive Benchmark, Instruction Dataset and Large Language Model for Finance

This repository introduces PIXIU, an open-source resource featuring the first financial large language models (LLMs), instruction tuning data, and evaluation benchmarks to holistically assess financial LLMs. Our goal is to continually push forward the open-source development of financial artificial intelligence (AI).

## Overview

The advancement of Natural Language Processing (NLP) and machine learning (ML) techniques in financial technology (FinTech) has enabled a diverse set of capabilities from predicting stock price movements to advanced financial analytics. However, to effectively understand the complex financial language and concepts, domain-specific LLMs are necessary.

Despite prior efforts, there is a lack of open-source financial LLMs and benchmarks to evaluate them. Additionally, these models are not fine-tuned to follow natural language instructions, limiting their performance in downstream financial tasks.

To address these gaps, we introduce PIXIU, providing:

1. Open-source LLMs tailored for finance, FinMA, by fine-tuning LLaMA with the dataset constructed in PIXIU.
2. Large-scale, high-quality financial instruction data.
3. Holistic financial evaluation benchmarks FLUPE for assessing financial LLMs.

## Key Features

- **Open resources**: PIXIU openly provides the financial LLM, instruction tuning data, and datasets included in the evaluation benchmark to encourage open research and transparency.
  
- **Multi-task**: The instruction tuning data in PIXIU cover a diverse set of financial tasks, including four financial NLP tasks and two financial prediction tasks.

- **Multi-modality**: PIXIU's instruction tuning data consist of multi-modality financial data, including time series data from stock movement predictions and portfolio management tasks. It covers various types of financial texts, including reports, news articles, tweets, and regulatory filings.
  
- **Diversity**: Unlike previous benchmarks focusing mainly on financial NLP tasks, PIXIU's evaluation benchmark includes critical financial prediction tasks aligned with real-world scenarios, making it more challenging.

## Building PIXIU

To construct the multi-task and multi-modal instruction data, we collected publicly available training data from diverse tasks. We wrote task-specific instructions for each task and assembled these with data samples to create a large-scale instruction tuning data.

Using this dataset, we conducted multi-task instruction tuning on LLaMA to create FinMA, a domain-specific LLM.

We built the Financial Language Understanding and Prediction Evaluation Benchmark (FLUPE), covering 4 financial NLP tasks with 5 datasets, and 2 financial prediction tasks with 6 datasets under 2 different scenarios. This benchmark allows us to compare the performance of our model with BloombergGPT and general domain LLMs such as ChatGPT and GPT-4.

## Structure of Repository

The repository is organized as follows:

1. **Models**: Contains the FinMA model trained on our dataset.
2. **Instruction Tuning Data**: Multi-task and multi-modal instruction data for financial tasks.
3. **Evaluation Benchmark**: FLUPE for evaluating financial LLMs.

## Finma v0.1: Financial Fine-tuned Model

We are pleased to introduce the first version of Finma, a specialized model fine-tuned on Llama 7B. 

Finma v0.1 has been trained specifically on our NLP instruction dataset. This dataset is drawn from diverse and respected financial resources, ensuring that the model has a comprehensive understanding of financial language and scenarios.

This version is particularly designed to enhance the performance on financial tasks compared to the base Llama 7B model. We utilized the original data sources and trained Finma v0.1 with a significant number of examples, covering a wide range of financial topics.

Finma v0.1 is now available on Huggingface for public use. We look forward to the valuable contributions that this initial version will make to the financial NLP field and encourage users to apply it to various financial tasks and scenarios. We also invite feedback and shared experiences to help improve future versions.

## Instruction Dataset

Our instruction dataset is uniquely tailored for the domain-specific LLM, FinMA. This dataset has been meticulously assembled to fine-tune our model on a diverse range of financial tasks. It features multi-task and multi-modal data derived from the same large-scale financial corpus used to train BloombergGPT.

The dataset is multi-faceted, featuring tasks such as sentiment analysis, news headline classification, named entity recognition, question answering, stock movement prediction, and portfolio management. It covers both textual and time-series data modalities, offering a rich variety of financial data.

The table below summarizes the different tasks, their corresponding modalities, text types, and examples of the instructions used for each task:

| **Task** | **Modalities** | **Text Types** | **Instructions Examples** |
| --- | --- | --- | --- |
| Sentiment Analysis | Text | Reports, News Articles | "Identify the sentiment of the given report text." |
| News Headline Classification | Text | News Headlines | "Classify the financial topic of this news headline." |
| Named Entity Recognition | Text | Reports, Regulatory Filings | "Highlight the financial entities in this text." |
| Question Answering | Text | All Text Types | "Answer the following question based on the given text." |
| Stock Movement Prediction | Text, Time-Series | Reports, News, Stock Prices | "Predict the stock movement based on the given historical prices and news." |
| Portfolio Management | Text, Time-Series | Reports, News, Stock Prices | "Suggest a portfolio adjustment based on the given market information." |

The dataset contains a vast amount of data samples for each task, allowing FinMA to capture the nuances of the diverse financial tasks. The table below provides the statistical details of the instruction dataset of NLP tasks:


| Task            | Data Source                | Example Number (Train) | Example Number (Test) | Original Paper                               |
|-----------------|----------------------------|------------------------|-----------------------|----------------------------------------------|
| Sentiment Analysis    | Financial Phrasebank (FPB)   | 3,876                  | 970                   | Malo et al., 2014                            |
| Aspect-specific Sentiment Analysis | FiQA Sentiment Analysis (FiQA SA) | 938                    | 235                   | Maia et al., 2018                            |
| Binary Classification   | Headline                   | 9,129                  | 2,283                 | Sinha and Khandait, 2020                     |
| Named Entity Recognition   | NER                        | 504                    | 98                    | Salinas Alvarado et al., 2015; Tjong Kim Sang and De Meulder, 2003 |
| Conversational Question Answering | ConvFinQA                  | 11,104                 | 1,490                 | Chen et al., 2022                            |


This dataset, along with the model and evaluation benchmark, is available in an open-source format to support future research in the financial AI sector. More details about the dataset and its usage can be found in the README in the "Instruction Tuning Data" directory.

## Usage

Please refer to the individual README files in the respective directories for usage instructions for the models, datasets, and benchmark.

## Citation

If you use PIXIU in your work, please cite our paper.

```
(Bibtex format of the paper will go here)
```

## License

PIXIU is licensed under [LICENSE NAME]. For more details, please see the [LICENSE](LICENSE) file.

## Contributions

We welcome contributions to the PIXIU project. Please refer to our [CONTRIBUTING](CONTRIBUTING.md) guide for more details.
