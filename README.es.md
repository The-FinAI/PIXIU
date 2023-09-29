<p align="center" width="100%">
<img src="https://i.postimg.cc/xTpWgq3L/pixiu-logo.png"  width="100%" height="100%">
</p>

- [English](README.md)

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



-----------------

![](https://img.shields.io/badge/pixiu-v0.1-gold)
![](https://black.readthedocs.io/en/stable/_static/license.svg)
[![Discord](https://img.shields.io/discord/1146837080798933112)](https://discord.gg/HRWpUmKB)

[Pixiu Paper](https://arxiv.org/abs/2306.05443) | [FLARE Leaderboard](https://huggingface.co/spaces/ChanceFocus/FLARE)

**Descargo de responsabilidad**

Este repositorio y su contenido se proporcionan **únicamente con fines académicos y educativos**. Ninguno de los materiales constituye asesoramiento financiero, legal o de inversión. No se ofrecen garantías, explícitas o implícitas, respecto a la precisión, integridad o utilidad del contenido. Los autores y colaboradores no son responsables de errores, omisiones o cualquier consecuencia derivada del uso de la información aquí contenida. Los usuarios deben ejercer su propio juicio y consultar a profesionales antes de tomar cualquier decisión financiera, legal o de inversión. El uso del software e información contenida en este repositorio es bajo el propio riesgo del usuario.

**Al utilizar o acceder a la información de este repositorio, usted acepta indemnizar, defender y eximir de responsabilidad a los autores, colaboradores y cualquier organización o persona afiliada por cualquier reclamo o daño.**




**Puntos de control:** 

- [FinMA v0.1 (NLP 7B version)](https://huggingface.co/ChanceFocus/finma-7b-nlp)
- [FinMA v0.1 (Full 7B version)](https://huggingface.co/ChanceFocus/finma-7b-full)

**Evaluaciones** (más detalles en la sección FLARE):

> Análisis de sentimientos

- [FPB (flare_fpb)](https://huggingface.co/datasets/ChanceFocus/flare-fpb)
- [FIQASA (flare_fiqasa)](https://huggingface.co/datasets/ChanceFocus/flare-fiqasa)
- [FOMC (flare_fomc)](https://huggingface.co/datasets/ChanceFocus/flare-fomc)
- [Headlines (flare_headlines)](https://huggingface.co/datasets/ChanceFocus/flare-headlines)

> Extracción de conocimiento 

- [NER (flare_ner)](https://huggingface.co/datasets/ChanceFocus/flare-ner)
- [Finer Ord (flare_finer_ord)](https://huggingface.co/datasets/ChanceFocus/flare-finer-ord)

> Comprensión numérica

- [FinQA (flare_finqa)](https://huggingface.co/datasets/ChanceFocus/flare-finqa)
- [ConvFinQA (flare_finqa)](https://huggingface.co/datasets/ChanceFocus/flare-convfinqa)

> Resumen de texto

- [ECTSUM (flare_ectsum)](https://huggingface.co/datasets/ChanceFocus/flare-ectsum)
- [EDTSUM (flare_edtsum)](https://huggingface.co/datasets/ChanceFocus/flare-edtsum)

> Calificación crediticia

- [Alemán (flare_german)](https://huggingface.co/datasets/ChanceFocus/flare-german)
- [Australiano (flare_german)](https://huggingface.co/datasets/ChanceFocus/flare-australian)

> Pronóstico

- [BigData22 para movimiento de acciones (flare_sm_bigdata)](https://huggingface.co/datasets/ChanceFocus/flare-sm-bigdata)
- [ACL18 para movimiento de acciones (flare_sm_acl)](https://huggingface.co/datasets/ChanceFocus/flare-sm-acl)
- [CIKM18 para movimiento de acciones (flare_sm_cikm)](https://huggingface.co/datasets/ChanceFocus/flare-sm-cikm)

## Descripción general

¡Bienvenido al proyecto **PIXIU**! Este proyecto está diseñado para respaldar el desarrollo, el ajuste fino y la evaluación de modelos de lenguaje grandes (LLM) en el dominio financiero. PIXIU es un paso significativo hacia la comprensión y aprovechamiento del poder de los LLM en el dominio financiero.

### Estructura del repositorio

El repositorio está organizado en varios componentes clave, cada uno con un propósito único en la canalización de NLP financiero:

- **FLARE**: Nuestro conjunto de evaluación de comprensión y predicción del lenguaje financiero. FLARE sirve como conjunto de evaluación para los LLM financieros, con un enfoque en tareas de comprensión y predicción en varios contextos financieros.
- **FIT**: Nuestro conjunto de datos de instrucción financiera. FIT es un conjunto de datos de instrucción multitarea y multimodal diseñado específicamente para tareas financieras. Sirve como terreno de entrenamiento para el ajuste fino de LLM para estas tareas.

- **FinMA**: Nuestro modelo de lenguaje grande financiero (LLM). FinMA es el núcleo de nuestro proyecto, proporcionando el poder de aprendizaje y predicción para nuestras tareas financieras.

### Características clave

- **Recursos abiertos**: PIXIU proporciona abiertamente el LLM financiero, los datos de instrucción de ajuste fino y los conjuntos de datos incluidos en el conjunto de evaluación de referencia para fomentar la investigación abierta y la transparencia.
  
- **Multitarea**: Los datos de instrucción y el conjunto de referencia en PIXIU cubren un diverso conjunto de tareas financieras, que incluyen cuatro tareas de NLP financiero y una tarea de predicción financiera.
- **Multimodalidad**: Los datos de instrucción y el conjunto de referencia de PIXIU consisten en datos financieros multimodales, que incluyen datos de series de tiempo de la tarea de predicción de movimientos de acciones. Cubre varios tipos de textos financieros, que incluyen informes, artículos de noticias, tweets y presentaciones regulatorias.
- **Diversidad**: A diferencia de conjuntos de referencia anteriores que se centran principalmente en tareas de NLP financiero, el conjunto de evaluación de referencia de PIXIU incluye tareas críticas de predicción financiera alineadas con escenarios del mundo real, lo que lo hace más desafiante.

---

## FLARE 2.0: Conjunto de evaluación de comprensión y predicción del lenguaje financiero

En esta sección, proporcionamos un análisis detallado del rendimiento de FinMA en comparación con otros modelos líderes, que incluyen ChatGPT, GPT-4 entre otros. Para este análisis, hemos elegido una variedad de tareas y métricas que abarcan varios aspectos del procesamiento de lenguaje natural financiero y la predicción financiera.

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

3. Evaluación autoalojada

Para ejecutar la interfaz de inferencia:

```bash
bash scripts/run_interface.sh
```

Ajuste run_interface.sh de acuerdo con los requisitos de su entorno.

Para evaluar:

```bash
python data/*/evaluate.py
```

### Crear nuevas tareas

Crear una nueva tarea para FLARE implica crear un conjunto de datos Huggingface e implementar la tarea en un archivo Python. Esta guía lo guía a través de cada paso para configurar una nueva tarea utilizando el marco FLARE.

#### Creando su conjunto de datos en Huggingface

Su conjunto de datos debe crearse en el siguiente formato:

```python
{
    "query": "...",
    "answer": "...",
    "text": "..."
}
```

En este formato:

- `query`: Combinación de su indicación y texto
- `answer`: Su etiqueta

Para tareas multi-turno (como ConvFinQA):

Para tareas de clasificación (como [FPB (flare_fpb)](https://huggingface.co/datasets/ChanceFocus/flare-fpb)), se deben definir claves adicionales:

- `choices`: Conjunto de etiquetas
- `gold`: Índice de la etiqueta correcta en choices (comenzando desde 0)

Para tareas de etiquetado secuencial (como [Finer Ord (flare_finer_ord)](https://huggingface.co/datasets/ChanceFocus/flare-finer-ord)), se deben definir claves adicionales:

- `label`: Lista de etiquetas de token

- `token`: Lista de tokens

Para tareas de resumen extractivo (como [ECTSUM (flare_ectsum)](https://huggingface.co/datasets/ChanceFocus/flare-ectsum)), se deben definir claves adicionales:

- `label`: Lista de etiquetas de oración

Para tareas de **resumen abstractivo** y **respuesta a preguntas** (como [EDTSUM (flare_edtsum)](https://huggingface.co/datasets/ChanceFocus/flare-edtsum)), no se deben definir claves adicionales.

#### Implementando la tarea

Una vez que su conjunto de datos esté listo, puede comenzar a implementar su tarea. Su tarea debe definirse dentro de una nueva clase en flare.py o cualquier otro archivo Python ubicado dentro del directorio tasks.

Para adaptarse a una variedad de tareas, ofrecemos varias clases base especializadas, que incluyen `Classification`, `SequentialLabeling`, `RelationExtraction`, `ExtractiveSummarization`, `AbstractiveSummarization` y `QA`.

Por ejemplo, si está embarcándose en una tarea de clasificación, puede aprovechar directamente nuestra clase base `Classification`. Esta clase permite una creación de tareas eficiente e intuitiva. Para demostrar mejor esto, profundicemos en un ejemplo de creación de una tarea llamada FLARE-FPB utilizando la clase base `Classification`:

```python
class FlareFPB(Classification):
    DATASET_PATH = "flare-fpb"
```

¡Y eso es todo! Una vez que haya creado la clase de tarea, el siguiente paso es registrarla en el archivo `src/tasks/__init__.py` . Para hacer esto, agregue una nueva línea con el formato `"task_name": module.ClassName`. Así es como se hace:

```python
TASK_REGISTRY = {
    "flare_fpb": flare.FPB,
    "your_new_task": your_module.YourTask,  # Aquí es donde agrega su tarea  
}
```

#### Métricas de tareas predefinidas

| Tarea                                   | Métrica                                | Ilustración                                                  |
| --------------------------------------- | -------------------------------------- | ------------------------------------------------------------ |
| Clasificación                           | Precisión                              | Esta métrica representa la proporción de observaciones predichas correctamente al total de observaciones. Se calcula como (Verdaderos Positivos + Verdaderos Negativos) / Observaciones Totales. |
| Clasificación                           | Puntuación F1                          | La Puntuación F1 representa la media armónica de precisión y sensibilidad, creando un equilibrio entre estos dos factores. Es particularmente útil en escenarios donde un factor es más significativo que el otro. La puntuación varía de 0 a 1, siendo 1 la perfecta precisión y sensibilidad, y 0 el peor caso. Además, proporcionamos versiones 'ponderadas' y 'macro' de la Puntuación F1. |
| Clasificación                           | Proporción Faltante                    | Esta métrica calcula la proporción de respuestas donde no se devuelve ninguna opción de las dadas en la tarea. |
| Clasificación                           | Coeficiente de Correlación de Matthews (MCC) | El MCC es una métrica que evalúa la calidad de las clasificaciones binarias, produciendo una puntuación que varía de -1 a +1. Una puntuación de +1 significa una predicción perfecta, 0 indica una predicción no mejor que una elección al azar, y -1 indica una predicción completamente inversa. |
| Etiquetado Secuencial                   | Puntuación F1                          | En el contexto de tareas de Etiquetado Secuencial, utilizamos la Puntuación F1 calculada por la biblioteca `seqeval`, una métrica robusta de evaluación a nivel de entidad. Esta métrica exige una coincidencia exacta tanto en el alcance como en el tipo de entidad entre las entidades predichas y las reales para una evaluación correcta. Verdaderos Positivos (VP) representan entidades correctamente predichas, Falsos Positivos (FP) denotan entidades incorrectamente predichas o entidades con rangos/tipos no coincidentes, y Falsos Negativos (FN) señalan entidades omitidas en la verdad terrenal. La precisión, sensibilidad y Puntuación F1 se calculan usando estas cantidades, con la Puntuación F1 representando la media armónica de precisión y sensibilidad. |
| Etiquetado Secuencial                   | Puntuación F1 de Etiqueta              | Esta métrica evalúa el rendimiento del modelo basado únicamente en la corrección de las etiquetas predichas, sin considerar los rangos de entidad. |
| Extracción de Relaciones                | Precisión                              | La precisión mide la proporción de relaciones predichas correctamente de todas las relaciones predichas. Se calcula como el número de Verdaderos Positivos (VP) dividido por la suma de Verdaderos Positivos y Falsos Positivos (FP). |
| Extracción de Relaciones                | Sensibilidad                           | La sensibilidad mide la proporción de relaciones predichas correctamente de todas las relaciones reales. Se calcula como el número de Verdaderos Positivos (VP) dividido por la suma de Verdaderos Positivos y Falsos Negativos (FN). |
| Extracción de Relaciones                | Puntuación F1                          | La Puntuación F1 es la media armónica de precisión y sensibilidad, y proporciona un equilibrio entre estas dos métricas. La Puntuación F1 es la mejor en 1 (precisión y sensibilidad perfectas) y la peor en 0. |
| Sumarización Extractiva y Abstractiva   | Rouge-N                                | Esto mide la superposición de N-gramas (una secuencia contigua de N elementos de una muestra de texto dada) entre el resumen generado por el sistema y el resumen de referencia. 'N' puede ser 1, 2 o más, siendo ROUGE-1 y ROUGE-2 comúnmente usados para evaluar superposiciones de unigramas y bigramas respectivamente. |
| Sumarización Extractiva y Abstractiva   | Rouge-L                                | Esta métrica evalúa la subsecuencia común más larga (LCS) entre los resúmenes del sistema y de referencia. LCS tiene en cuenta la similitud de la estructura a nivel de frase de forma natural e identifica los n-gramas en secuencia co-ocurrentes más largos automáticamente. |
| Respuesta a Preguntas                   | EmACC                                  | EMACC evalúa la coincidencia exacta entre la respuesta generada por el modelo y la respuesta de referencia. En otras palabras, la respuesta generada por el modelo se considera correcta solo si coincide exactamente, palabra por palabra, con la respuesta de referencia. |

> Adicionalmente, puedes determinar si las etiquetas deben estar en minúsculas durante el proceso de coincidencia especificando `LOWER_CASE` en la definición de tu clase. Esto es pertinente ya que las etiquetas se coinciden basándose en su apariencia en la salida generada. Para tareas como exámenes donde las etiquetas son un conjunto específico de letras mayúsculas como 'A', 'B', 'C', esto normalmente debería establecerse en Falso.

---

## FIT: Conjunto de Instrucciones Financieras

Nuestro conjunto de instrucciones está especialmente adaptado para el LLM específico del dominio, FinMA. Este conjunto de datos ha sido meticulosamente ensamblado para afinar nuestro modelo en una amplia gama de tareas financieras. Presenta datos multitarea y multimodales públicamente disponibles derivados de múltiples conjuntos de datos financieros abiertos.

El conjunto de datos es multifacético, presentando tareas que incluyen análisis de sentimientos, clasificación de titulares de noticias, reconocimiento de entidades nombradas, respuesta a preguntas y predicción de movimientos bursátiles. Cubre modalidades de datos textuales y de series temporales, ofreciendo una rica variedad de datos financieros. Las instrucciones específicas de tarea para cada tarea han sido cuidadosamente diseñadas por expertos en el dominio.

### Modalidad y Prompts

La tabla a continuación resume las diferentes tareas, sus modalidades correspondientes, tipos de texto y ejemplos de las instrucciones utilizadas para cada tarea:

| **Tarea**                              | **Modalidades** | **Tipos de Texto**       | **Ejemplos de Instrucciones**                                    |
| -------------------------------------- | --------------- | ------------------------ | --------------------------------------------------------------- |
| Análisis de Sentimientos               | Texto           | titulares de noticias, tweets | "Analiza el sentimiento de esta declaración extraída de un artículo de noticias financieras. Proporciona tu respuesta como negativo, positivo o neutral. Por ejemplo, 'Las acciones de la empresa cayeron tras el escándalo.' se clasificaría como negativo." |
| Clasificación de Titulares de Noticias | Texto           | Titulares de Noticias     | "Considera si el titular menciona el precio del oro. ¿Se indica un Precio o No en el mercado de commodities del oro en el titular de la noticia? Por favor, responde Sí o No." |
| Reconocimiento de Entidades Nombradas  | Texto           | acuerdos financieros      | "En las oraciones extraídas de los acuerdos financieros en los archivos de la SEC de EE. UU., identifica las entidades nombradas que representan una persona ('PER'), una organización ('ORG') o una ubicación ('LOC'). El formato de respuesta requerido es: 'nombre de la entidad, tipo de entidad'. Por ejemplo, en 'Elon Musk, CEO de SpaceX, anunció el lanzamiento desde Cabo Cañaveral.', las entidades serían: 'Elon Musk, PER; SpaceX, ORG; Cabo Cañaveral, LOC'." |
| Respuesta a Preguntas                  | Texto           | informes de ganancias     | "En el contexto de esta serie de consultas relacionadas con las finanzas y la información adicional proporcionada por el pretexto, datos de la tabla y texto posterior de los informes financieros de una empresa, por favor, proporciona una respuesta a la pregunta final. Esto puede requerir extraer información del contexto y realizar cálculos matemáticos. Por favor, ten en cuenta la información proporcionada en las preguntas anteriores y sus respuestas al formular tu respuesta:" |
| Predicción de Movimientos Bursátiles   | Texto, Series Temporales | tweets, Precios de Acciones | "Analiza la información y las publicaciones en redes sociales para determinar si el precio de cierre de *\{tid\}* ascenderá o descenderá en *\{point\}*. Por favor, responde con Ascenso o Descenso." |

### Estadísticas del Conjunto de Datos

El conjunto de datos contiene una vasta cantidad de muestras de datos de instrucciones, lo que permite a FinMA capturar las sutilezas de las diversas tareas financieras. La tabla a continuación proporciona los detalles estadísticos del conjunto de datos de instrucciones:

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

### Generando Conjuntos de Datos para FIT

Cuando trabajas con el Conjunto de Datos de Instrucción Financiera (FIT, por sus siglas en inglés), es crucial seguir el formato prescrito para entrenar y probar modelos.

El formato debe verse así:

```json
{
    "id": "id único",
    "conversations": [
        {
            "from": "humano",
            "value": "Tu mensaje y texto"
        },
        {
            "from": "agente",
            "value": "Tu respuesta"
        }
    ],
    "text": "Texto a ser clasificado",
    "label": "Tu etiqueta"
}
```

Aquí está lo que significa cada campo::

- "id": un identificador único para cada ejemplo en tu conjunto de datos.
- "conversations": una lista de turnos de conversación. Cada turno se representa como un diccionario, donde "from" representa al hablante, y "value" representa el texto hablado en el turno.
- "text": el texto a clasificar.
- "label": la etiqueta de verdad fundamental para el texto..


El primer turno en la lista "conversations" siempre debe ser de "humano", y contener tu mensaje y el texto. El segundo turno debe ser de "agente", y contener tu respuesta.

---

## FinMA v0.1: Modelo de lenguaje grande financiero

Nos complace presentar la primera versión de FinMA, que incluye tres modelos FinMA-7B, FinMA-7B-full, FinMA-30B, ajustados en LLaMA 7B y LLaMA-30B. FinMA-7B y FinMA-30B están entrenados con los datos de instrucción de NLP, mientras que FinMA-7B-full está entrenado con todos los datos de instrucción de FIT que cubren tanto tareas de NLP como de predicción.

FinMA v0.1 está disponible ahora en [Huggingface](https://huggingface.co/ChanceFocus/finma-7b-nlp) para uso público. Esperamos las valiosas contribuciones que esta versión inicial hará al campo de NLP financiero y alentamos a los usuarios a aplicarlo en diversas tareas y escenarios financieros. También invitamos a recibir comentarios y experiencias compartidas para ayudar a mejorar futuras versiones.

### Cómo ajustar fino un nuevo modelo de lenguaje grande usando PIXIU basado en FIT

Próximamente.

---


## Cita

Si utiliza PIXIU en su trabajo, cite nuestro artículo.

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

PIXIU tiene licencia [MIT]. Para más detalles, consulte el archivo [MIT](LICENSE).

## Historial de estrellas

[![Star History Chart](https://api.star-history.com/svg?repos=chancefocus/pixiu&type=Date)](https://star-history.com/#chancefocus/pixiu&Date)

