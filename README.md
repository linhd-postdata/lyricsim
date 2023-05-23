# LYRICSIM: A novel dataset and framework for lyric similarity detection in Spanish

This repository contains the dataset and code for the paper [LYRICSIM: A novel dataset and framework for paragraph-level similarity detection in Spanish song lyrics](https://arxiv.org/abs/).

## Table of Contents

- [Introduction](#introduction)
- [Project Directory Structure](#project-directory-structure)
- [Dataset](#dataset)
  - [Annotation Task](#annotation-task)
    - [Description of the dataset](#description-of-the-dataset)
    - [Semantic Differential Scale](#semantic-differential-scale)
  - [Data Refinement](#data-refinement)
- [Fine-tuning and Evaluation](#fine-tuning-and-evaluation)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## Introduction

We propose an evaluation framework, inspired by recent work by other researchers in the field of Spanish NLP, that effectively measures the performance of semantic similarity models in the context of song lyrics, considering various aspects such as theme, message, emotions, and cultural context. We conduct extensive experiments to demonstrate the effectiveness of our dataset and evaluation framework in promoting the development of more accurate and applicable semantic similarity models for song lyrics. Finally, we provide a detailed analysis of the performance of various state-of-the-art models on our dataset, highlighting the strengths and weaknesses of each approach and identifying future research directions in this domain. By developing a dataset specifically tailored to song lyric similarity and a corresponding evaluation framework, this paper aims to bridge the gap between general-purpose semantic similarity tasks and domain-specific applications, ultimately contributing to the advancement of NLP research in the context of music and lyrics research and analysis.

## Project Directory Structure

In this section, we present the directory structure of the project, which provides an overview of the organization of the files and folders. This will help you navigate through the project more efficiently and understand the different components involved. The tree view of the project's directories and files is displayed within a code block for better readability:

```tree
lyricsim/
┣ data/
┃ ┣ processed/
┃ ┃ ┣ avg_ratings.csv
┃ ┃ ┣ clean_detailed_results.json
┃ ┃ ┗ filtered_detailed_results.csv
┃ ┗ raw/
┃   ┣ sts/
┃ ┃ ┃ ┣ answers/
┃ ┃ ┃ ┃ ┣ answers-belief.tsv
┃ ┃ ┃ ┃ ┣ answers-forums.tsv
┃ ┃ ┃ ┃ ┣ answers-headlines.tsv
┃ ┃ ┃ ┃ ┣ answers-images.tsv
┃ ┃ ┃ ┃ ┗ answers-students.tsv
┃ ┃ ┃ ┗ input/
┃ ┃ ┃   ┣ STS.input.answers-forums.txt
┃ ┃ ┃   ┣ STS.input.answers-students.txt
┃ ┃ ┃   ┣ STS.input.belief.txt
┃ ┃ ┃   ┣ STS.input.headlines.txt
┃ ┃ ┃   ┗ STS.input.images.txt
┃   ┣ DetailedResults.xlsx
┃   ┣ annotation_results.csv
┃   ┣ detailed_results.json
┃   ┣ full_lyrics.tsv
┃   ┗ lyrics_pairs.csv
┣ notebooks/
┃ ┣ data_processing.ipynb
┃ ┣ evaluation_process.ipynb
┃ ┣ exploration.ipynb
┃ ┣ sts_analysis.ipynb
┃ ┗ utils.py
┣ sts/
┃ ┣ alberti_base_sts_batch8_lr0.00001_decay0.1.sh
┃ ┣ generate_scripts.py
┃ ┗ sts_dataset.py
┣ .gitignore
┣ README.md
┣ bsc_run_glue.py
┗ requirements.txt
```

## Dataset

### Annotation Task

#### Description of the dataset

The dataset we prepared for the annotation task contains 75 song lyrics in Spanish that were selected for their diversity and popularity and for representing a wide range of music genres and themes. Also, we included song lyrics of varying lengths, to check whether this variable had any influence on the participant's perception of similarity.

#### Semantic Differential Scale

Participants in the study were recruited through a crowdsourcing platform and asked to rate the similarity between pairs of song lyrics in Spanish using a six-point semantic differential scale ranging from 0, for completely different items, to 5 for outstandingly similar items. They were instructed to evaluate the similarity of pairs of song lyrics based on various criteria, such as the primary theme or context of the lyrics, the message conveyed, the emotions or feelings expressed, the literal meaning, the vocabulary employed, the relationship between the sender and receiver, the language style, and the sociocultural context of the song.

The experimental procedure involved presenting subjects with pairs of song lyrics and asking them to estimate the similarity of the lyrics using a 6-point Likert scale ranging from "No similarity" to "Striking similarity." Each pair was assigned to three participants that were randomly chosen from a pool of 63. In this manner, we obtained 2775 comparisons that are included in the dataset located at `data/raw/annotation_results.csv`. A list of the lyrics pairs that were presented to the participants is also included in the same folder at `data/raw/lyrics_pairs.csv`.

### Data Refinement

The Data Refinement section addresses the mitigation of potential biases in the dataset arising from unbalanced annotation authoring. To guarantee the inclusion of high-quality annotations, only pairs exhibiting a strong consensus among the three annotators were retained.

The researchers employed various criteria to filter the collected data, aiming to decrease rating variability. They differentiated between similarity annotations (scores from 1 to 5) and dissimilarity annotations (scores 1 to 5), premised on the belief that similarity relationships are optimally represented as free-scale networks. This phenomenon is apparent in the Kernel Density Estimations (KDEs) of rating distributions in the STS and LyricSIM datasets, which demonstrate a bias towards lower ratings.

By clearly distinguishing between dissimilar and similar items, the researchers sought to further characterize the tail of the distribution.

To evaluate the reliability of the dataset, we computed the inter-annotator agreement using Krippendorff's alpha, which yielded a value of 0.90. Krippendorff's alpha is a versatile metric that generalizes other metrics responsible for quantifying the inter-rater reliability. It applies to both ordinal and nominal annotations and can accommodate any number of annotators. K-alpha produces a value ranging from 0 to 1, with 1 representing complete agreement. However, the criteria for sufficient agreement between annotators vary. According to the general consensus, an acceptable threshold is a K-alpha value greater than 0.8. The resulting inter-annotator agreement for the high-quality annotations dataset was found to be substantial, with a coefficient alpha of 0.90, indicating the dataset's reliability for future research in the field. Table X provides a comparison between the STS gold standard datasets used in the 2013 SemEval task and our dataset, detailing dataset size (in sentence pairs), the number of annotators per pair, and computed Krippendorff's alpha scores, when available. Regrettably, we could not locate the results of the STS 2012/2013 Spanish task.

## Fine-tuning and Evaluation

We follow a similar 85-5-10 split as in other studies, resulting in 638 song pairs for training, 38 pairs for development, and 68 pairs for testing. We use stratified sampling during the splitting process to ensure a balanced representation of each class. The evaluation metric is the combined score, computed as the arithmetic mean of two widely accepted correlation coefficients: Spearman's Rank Correlation and Pearson's Correlation.

We selected several language models, including their base and large versions, which we believe are suitable candidates for training and evaluation processes related to Spanish song lyrics. These models are BERTIN, RoBERTa-base-bne (MarIA base), RoBERTa-large-bne (MarIA large), Sentence Transformer, ALBERTI, DeBERTa, XML-RoBERTa base, and XML-RoBERTa large. Our selection aimed to include a significant representation of monolingual Spanish models and multilingual models that have demonstrated superior performance in Spanish tasks.

The fine-tuning process was conducted using the same practices found in similar studies. We employed scripts based on the HuggingFace Transformers library, with minor modifications to adapt them to our dataset structure. To maintain consistency across the models, each was initialized with a random head, and a fixed seed was used for reproducibility. We conducted a grid search over the following search space:

- Weight decay: 0.1, 0.01
- Learning rate: 1e-5, 2e-5, 3e-5, 5e-5
- Batch size: 8, 16, 32

Memory constraints, especially for larger models, led to using gradient accumulation when the batch size exceeded capacity, achieving the same effective batch size while keeping other hyperparameters at their default values. The maximum sequence length was set to 512 tokens for all models to accommodate all sentence pairs in the dataset. To prevent overfitting, each model was trained for a maximum of 5 epochs using the Adam optimizer and a linear decaying learning rate. We selected the checkpoint with the highest score according to the development set. Finally, we performed an evaluation of the test set for each model, using the best checkpoint from the previous step. A comprehensive display of the optimal configuration of hyperparameters for each model is presented in a separate table.

## Results

The results are shown in the Table for each model, fine-tuned as described. Among the eight models investigated in this study, the BERTIN base demonstrated the highest performance in terms of the Combined Score metric. Maria's base model closely followed BERTIN, exhibiting competitive results. Interestingly, both Maria models also showed slightly higher Spearman's Rank correlation coefficients than BERTIN. The XLM-RoBERTa model ranked fourth, while the ALBERTI base lagged noticeably behind the other models.

These findings suggest that the BERTIN model is the most effective in capturing semantic textual similarity on songs, with Maria as a strong contender; being two of the three monolingual models tested, they ranked higher than multilingual models such as XLM-RoBERTa, mDeBERTa3, and ALBERTI. XLM-RoBERTa base achieved the highest Spearman's Rank correlation score, while XLM-RoBERTa large, despite being a larger model, did not show a significant improvement over its base counterpart. The ALBERTI model appears to be less successful in this context, contrary to expectation, as this model was trained on data with a comparable structure to songs.
The results of the evaluation of the development set, which were used as criteria for selecting the best checkpoint from the fine-tuning process and hyperparameter selection, can be found in Table. A comparative analysis of the development and test set results may provide further insights into the generalization capabilities of the models and the effectiveness of the fine-tuning process.

|      **model name**       | **STS combined** | **combined** | **spearmanr** | **pearson** |
| :----------------------- | ---------------: | -----------: | ------------: | ----------: |
| BERTIN                    |            79.45 | **82.09**    |         79.84 | **84.35**   |
| MarIA large               |            84.11 | 81.63        |         80.09 | 83.16       |
| MarIA base                | **85.33**        | 80.76        |         80.3  | 81.23       |
| XLM-RoBERTa base          |            83.47 | 80.11        | **82.09**     | 78.14       |
| Sentence Transformer XLM-R|            -     | 79.79        |         78.99 | 80.59       |
| XLM-RoBERTa large         |            84.04 | 78.24        |         78.77 | 77.72       |
| mDeBERTa3                 |            83.61 | 77.79        |         78.01 | 77.57       |
| ALBERTI                   |            -     | 76.11        |         76.09 | 76.14       |

Table: Test combined scores for all the models considered. STS dataset results from other studies [1] have been added for illustration purposes.

| **model name** | **Batch size** | **Weight decay** | **Learning rate** | **Eval** | **Test** |
| :------------- | --------------:| ----------------:| -----------------:| --------:| --------:|
| BERTIN         |             16 |              0.01|           0.00005 |    76.31 | **82.09**|
| MarIA large    |             32 |              0.01|           0.00005 |    77.45 |  81.63   |
| MarIA base     |              8 |              0.01|           0.00003 |    77.00 |  80.76   |
| XLM-R base     |              8 |              0.01|           0.00002 | **77.91**|  80.11   |
| ST XLM-R       |             32 |              0.01|           0.00003 |    77.09 |  79.79   |
| XLM-R large    |              8 |              0.1 |           0.00001 |    76.13 |  78.24   |
| mDeBERTa3      |             32 |              0.1 |           0.00003 |    74.97 |  77.79   |
| ALBERTI        |             32 |              0.1 |           0.00005 |    71.67 |  76.11   |

Table: Best configuration for each model with combined test and evaluation scores.

[1] Agerri, R., & García-Serrano, A. (2022). Lessons learned from the STS shared tasks. Natural Language Engineering, 28(2), 135-148.


## Citation

If you use the LyricSIM dataset or any part of this code in your research, please cite our paper:

## License

This repository and its contents are licensed under the [MIT License](https://opensource.org/license/MIT/) for the code and data are under the [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.en) license.
