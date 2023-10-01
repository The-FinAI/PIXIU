<p align="center" width="100%">
<img src="https://i.postimg.cc/xTpWgq3L/pixiu-logo.png"  width="100%" height="100%">
</p>

<div>
<div align="left">
    <a target='_blank'>Qianqian Xie<sup>1</sup></span>&emsp;
    <a target='_blank'>Weiguang Han<sup>1</sup></span>&emsp;
    <a target='_blank'>Xiao Zhang<sup>2</sup></a>&emsp;
    <a target='_blank'>Ruoyu Xiang<sup>6</sup></a>&emsp;
    <a target='_blank'>Duanyu Feng<sup>3</sup></a>&emsp;
    <a target='_blank'>Yongfu Dai<sup>3</sup></a>&emsp;
    <a target='_blank'>Hao Wang<sup>3</sup></a>&emsp;
    <a target='_blank'>Yanzhao Lai<sup>4</sup></a>&emsp;
    <a target='_blank'>Min Peng<sup>1</sup></a>&emsp;
    <a href='https://warrington.ufl.edu/directory/person/12693/' target='_blank'>Alejandro Lopez-Lira<sup>5</sup></a>&emsp;
    <a href='https://jimin.chancefocus.com/' target='_blank'>Jimin Huang*<sup>,7</sup></a>
</div>
<div>
<div align="left">
    <sup>1</sup>Universidad de Wuhany&emsp;
    <sup>2</sup>Universidad Sun Yat-sen&emsp;
    <sup>3</sup>Universidad de Sichuan&emsp;
    <sup>4</sup>Universidad Southwest Jiaotong&emsp;
    <sup>5</sup>Universidad de Florida&emsp;
    <sup>6</sup>New York Universidad&emsp;
  <sup>7</sup>ChanceFocus AMC.
</div>
<div align="left">
    <img src='https://i.postimg.cc/CLtkBwz7/57-EDDD9-FB0-DF712-F3-AB627163-C2-1-EF15655-13-FCA.png' alt='Wuhan University Logo' height='100px'>&emsp;
    <img src='https://i.postimg.cc/C1XnZNK1/Sun-Yat-sen-University-Logo.png' alt='Sun Yat-Sen University Logo' height='100px'>&emsp;
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
- [Spainish](README.es.md)

**Evaluations** (More details on FLARE section):

- [FLARE (flare-es-financees)](https://huggingface.co/datasets/ChanceFocus/flare-es-financees)
- [FLARE (flare-es-tsa)](https://huggingface.co/datasets/ChanceFocus/flare-es-tsa)
- [FLARE (flare-es-fns)](https://huggingface.co/datasets/ChanceFocus/flare-es-fns)
- [FLARE (flare-es-efpa)](https://huggingface.co/datasets/ChanceFocus/flare-es-efpa)
- [FLARE (flare-es-efp)](https://huggingface.co/datasets/ChanceFocus/flare-es-efp)
- [FLARE (flare-es-multifin)](https://huggingface.co/datasets/ChanceFocus/flare-es-multifin)


## Overview

**FLARE_ES** is a cornerstone initiative focusing on the Spanish financial domain, FLARE_ES aims to bolster the progress, refinement, and assessment of Large Language Models (LLMs) tailored specifically for Spanish financial contexts. As a vital segment of the broader PIXIU endeavor, FLARE_ES stands as a testament to the commitment in harnessing the capabilities of LLMs, ensuring that financial professionals and enthusiasts in the Spanish-speaking world have top-tier linguistic tools at their disposal.

### Key Features

- **Open resources**: PIXIU openly provides the financial LLM, instruction tuning data, and datasets included in the evaluation benchmark to encourage open research and transparency.
- **Multi-task**: The instruction tuning data and benchmark in PIXIU cover a diverse set of financial tasks, including four financial NLP tasks and one financial prediction task.
- **Multi-modality**: PIXIU's instruction tuning data and benchmark consist of multi-modality financial data, including time series data from the stock movement prediction task. It covers various types of financial texts, including reports, news articles, tweets, and regulatory filings.
- **Diversity**: Unlike previous benchmarks focusing mainly on financial NLP tasks, PIXIU's evaluation benchmark includes critical financial prediction tasks aligned with real-world scenarios, making it more challenging.

---

## FLARE_ES: Financial Language Understanding and Prediction Evaluation Benchmark

In this section, we provide a detailed performance analysis of FinMA compared to other leading models, including ChatGPT, GPT-4, lince-zero et al. For this analysis, we've chosen a range of tasks and metrics that span various aspects of financial Natural Language Processing and financial prediction.

### Tasks

| Data                  | Task                             | Raw    | Data Types                       | Modalities        | License         | Paper |
| --------------------- | -------------------------------- | ------ | -------------------------------- | ----------------- | --------------- | ----- |
| MultiFin              | news headline classification     | 230    | news headlines                   | text              | CC BY 4.0       | [1]   |
| FNS                   | question answering               | 50     | earnings reports                 | text              | Public          | [2]   |
| TSA                   | sentiment analysis               | 3,829  | news headlines                   | text              | CC BY 4.0       | [3]   |
| Financees             | sentiment analysis               | 6,539  | news headlines                   | text              | Public          | [4]   |
| EFP                   | question answering               | 37     | business assessment questions    | text              | Public          |       |
| EFPA                  | question answering               | 228    | business assessment questions    | text              | Public          |       |


1. Rasmus Jørgensen, Oliver Brandt, Mareike Hartmann, Xiang Dai, Christian Igel, and Desmond Elliott. 2023. MultiFin: A Dataset for Multilingual Financial NLP. In Findings of the Association for Computational Linguistics: EACL 2023, 894–909. Association for Computational Linguistics, Dubrovnik, Croatia.
2. [FNS 2023. FNP 2023.](http://wp.lancs.ac.uk/cfie/fns2023/).
3. Pan R, García-Díaz JA, Garcia-Sanchez F, and Valencia-García R. 2023. Evaluation of transformer models for financial targeted sentiment analysis in Spanish. In PeerJ Computer Science, 9:e1377. https://doi.org/10.7717/peerj-cs.1377.
4. CodaLab. 2023. [Competition](https://codalab.lisn.upsaclay.fr/competitions/10052)

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

**Descargo de responsabilidad**

Este repositorio y su contenido se proporcionan **únicamente con fines académicos y educativos**. Ninguno de los materiales constituye asesoramiento financiero, legal o de inversión. No se ofrecen garantías, explícitas o implícitas, respecto a la precisión, integridad o utilidad del contenido. Los autores y colaboradores no son responsables de errores, omisiones o cualquier consecuencia derivada del uso de la información aquí contenida. Los usuarios deben ejercer su propio juicio y consultar a profesionales antes de tomar cualquier decisión financiera, legal o de inversión. El uso del software e información contenida en este repositorio es bajo el propio riesgo del usuario.

**Al utilizar o acceder a la información de este repositorio, usted acepta indemnizar, defender y eximir de responsabilidad a los autores, colaboradores y cualquier organización o persona afiliada por cualquier reclamo o daño.**




**Puntos de control:** 

- [FinMA v0.1 (Full 7B version)](https://huggingface.co/ChanceFocus/finma-7b-full)

**Idiomas**

- [Inglés](README.md)
- [Español](README.es.md)

**Evaluaciones** (más detalles en la sección FLARE):

- [FLARE (flare-es-financees)](https://huggingface.co/datasets/ChanceFocus/flare-es-financees)
- [FLARE (flare-es-tsa)](https://huggingface.co/datasets/ChanceFocus/flare-es-tsa)
- [FLARE (flare-es-fns)](https://huggingface.co/datasets/ChanceFocus/flare-es-fns)
- [FLARE (flare-es-efpa)](https://huggingface.co/datasets/ChanceFocus/flare-es-efpa)
- [FLARE (flare-es-efp)](https://huggingface.co/datasets/ChanceFocus/flare-es-efp)
- [FLARE (flare-es-multifin)](https://huggingface.co/datasets/ChanceFocus/flare-es-multifin)

## Descripción general

**FLARE_ES** es una iniciativa fundamental enfocada en el dominio financiero español. FLARE_ES busca reforzar el progreso, perfeccionamiento y evaluación de Modelos de Lenguaje a Gran Escala (MLGs) diseñados específicamente para contextos financieros españoles. Como un segmento vital del esfuerzo más amplio de PIXIU, FLARE_ES se erige como un testimonio del compromiso por aprovechar las capacidades de los MLGs, asegurando que los profesionales y entusiastas financieros del mundo hispanohablante tengan a su disposición herramientas lingüísticas de primera categoría.

### Características clave

- **Recursos abiertos**: PIXIU proporciona abiertamente el LLM financiero, los datos de instrucción de ajuste fino y los conjuntos de datos incluidos en el conjunto de evaluación de referencia para fomentar la investigación abierta y la transparencia. 
- **Multitarea**: Los datos de instrucción y el conjunto de referencia en PIXIU cubren un diverso conjunto de tareas financieras, que incluyen cuatro tareas de NLP financiero y una tarea de predicción financiera.
- **Multimodalidad**: Los datos de instrucción y el conjunto de referencia de PIXIU consisten en datos financieros multimodales, que incluyen datos de series de tiempo de la tarea de predicción de movimientos de acciones. Cubre varios tipos de textos financieros, que incluyen informes, artículos de noticias, tweets y presentaciones regulatorias.
- **Diversidad**: A diferencia de conjuntos de referencia anteriores que se centran principalmente en tareas de NLP financiero, el conjunto de evaluación de referencia de PIXIU incluye tareas críticas de predicción financiera alineadas con escenarios del mundo real, lo que lo hace más desafiante.

---

## FLARE_ES: Conjunto de evaluación de comprensión y predicción del lenguaje financiero

En esta sección, proporcionamos un análisis de rendimiento detallado de FinMA en comparación con otros modelos líderes, incluyendo ChatGPT, GPT-4, lince-zero et al. Para este análisis, hemos elegido una gama de tareas y métricas que abarcan varios aspectos del Procesamiento del Lenguaje Natural financiero y de la predicción financiera.

### Tareas

| Datos                 | Tarea                          | Bruto  | Tipos de Datos                      | Modalidades       | Licencia        | Artículo |
| --------------------- | ------------------------------ | ------ | ----------------------------------- | ----------------- | --------------- | -------- |
| MultiFin              | clasificación de titulares     | 230    | titulares de noticias               | texto             | CC BY 4.0       | [1]      |
| FNS                   | respuesta a preguntas          | 50     | informes de ganancias               | texto             | Público         | [2]      |
| TSA                   | análisis de sentimientos       | 3,829  | titulares de noticias               | texto             | CC BY 4.0       | [3]      |
| Financees             | análisis de sentimientos       | 6,539  | titulares de noticias               | texto             | Público         | [4]      |
| EFP                   | respuesta a preguntas          | 37     | preguntas de evaluación empresarial | texto             | Público         |          |
| EFPA                  | respuesta a preguntas          | 228    | preguntas de evaluación empresarial | texto             | Público         |          |

1. Rasmus Jørgensen, Oliver Brandt, Mareike Hartmann, Xiang Dai, Christian Igel, and Desmond Elliott. 2023. MultiFin: A Dataset for Multilingual Financial NLP. In Findings of the Association for Computational Linguistics: EACL 2023, 894–909. Association for Computational Linguistics, Dubrovnik, Croatia.
2. [FNS 2023. FNP 2023.](http://wp.lancs.ac.uk/cfie/fns2023/).
3. Pan R, García-Díaz JA, Garcia-Sanchez F, and Valencia-García R. 2023. Evaluation of transformer models for financial targeted sentiment analysis in Spanish. In PeerJ Computer Science, 9:e1377. https://doi.org/10.7717/peerj-cs.1377.
4. CodaLab. 2023. [Competition](https://codalab.lisn.upsaclay.fr/competitions/10052)


### Evaluación

#### Preparación
##### Instalación local
```bash
git clone https://github.com/chancefocus/PIXIU.git --recursive
cd PIXIU
pip install -r requirements.txt
cd PIXIU/src/financial-evaluation
pip install -e .[multilingual]
```
##### Imagen de Docker
```bash
sudo bash scripts/docker_run.sh
```
El comando anterior inicia un contenedor docker, puede modificar docker_run.sh para adaptarlo a su entorno. Proporcionamos una imagen precompilada ejecutando sudo docker pull tothemoon/pixiu:latest

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
Argumentos de explicación:
- `[]` significa argumentos ignorables
- `HF_HOME`: directorio de caché huggingface
- `sshd_port`: puerto sshd del contenedor, puede ejecutar `ssh -i private_key -p $sshd_port root@$ip` para conectarse al contenedor, el valor predeterminado es 22001
- `--rm`: elimina el contenedor al salir del contenedor (es decir,`CTRL + D`)

#### Evaluación automatizada de tareas
Antes de la evaluación, descargue el [punto de control BART](https://drive.google.com/u/0/uc?id=1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m&export=download) en `src/metrics/BARTScore/bart_score.pth`.

Para la evaluación automatizada, siga estas instrucciones:

1. Transformador Huggingface

   Para evaluar un modelo alojado en HuggingFace Hub (por ejemplo, finma-7b-full), use este comando:

```bash
python eval.py \
    --model "hf-causal-llama" \
    --model_args "use_accelerate=True,pretrained=chancefocus/finma-7b-full,tokenizer=chancefocus/finma-7b-full,use_fast=False" \
    --tasks "flare_ner,flare_sm_acl,flare_fpb"
```

Puede encontrar más detalles en la documentación de [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness).

2. API comerciales


Tenga en cuenta que para tareas como NER, la evaluación automatizada se basa en un patrón específico. Esto podría no extraer información relevante en entornos de cero disparos, dando como resultado un rendimiento relativamente más bajo en comparación con los resultados anteriores anotados manualmente.

```bash
export OPENAI_API_SECRET_KEY=YOUR_KEY_HERE
python eval.py \
    --model gpt-4 \
    --tasks flare_ner,flare_sm_acl,flare_fpb
```

---


## License

PIXIU tiene licencia [MIT]. Para más detalles, consulte el archivo [MIT](LICENSE).

## Historial de estrellas

[![Star History Chart](https://api.star-history.com/svg?repos=chancefocus/pixiu&type=Date)](https://star-history.com/#chancefocus/pixiu&Date)

