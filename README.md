# LYRICSIM: A novel dataset and framework for lyric similarity detection in Spanish
This repository contains the dataset and code for the paper [LYRICSIM: A novel dataset and framework for paragraph-level similarity detection in Spanish song lyrics](https://arxiv.org/abs/).

## Annotation Dataset
A crowd-sourced annotation experiment was conducted to investigate how individuals perceive similarity at the paragraph level. To obtain a representative sample of popular themes in the paragraphs to be compared, we randomly picked a set of 75 songs from a large corpus of song lyrics in Spanish with some constraints: 

#TODO 1. Specify constraints 
 
The experimental procedure involved presenting subjects with pairs of song lyrics and asking them to estimate the similarity of the lyrics using a 6-point Likert scale ranging from "No similarity" to "Striking similarity." Each pair was assigned to three participants that were randomly chosen from a pool of 63 . In this manner, we obtained 2775 comparisons that are included in the dataset located at `data/raw/annotation_results.csv`. A list of the lyrics pairs that were presented to the participants is also included in the same folder at `data/raw/lyrics_pairs.csv`.



  