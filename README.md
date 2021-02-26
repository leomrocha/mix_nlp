# mix_nlp
Repo for my research on NLP

This repository currently contains two main parts

1. Gutenberg Project data Exploration
2. NLP Encodings research


## Gutenberg Project

This project is currently in the latest stages writing a report that will be similar to the one already available on [conllu v2.6 analysis](https://leomrocha.github.io/2020-06-22-ud_conllu_v2.6_analysis/)

The graphs and final report need to be merged and the code, notebooks and site will be ported to another repository that's only for the results.

## NLP Encodings Research

The point is here to go back to the basic ideas and see what can be done to do create a universal encoder.

Most ideas are already written in a [draft](https://github.com/leomrocha/mix_nlp/blob/master/utf8/notebooks/DRAFT-V2-reorder-paper.ipynb) although unordered and need refining.

The next stages here are writing reports/papers on the following order:

### 1. Encoding Discussions and proposals
Finish the report (and maybe add some more for the analysis) motivation and a few ideas on encoding the complete utf-8 domain, show this makes input layers smaller (from 1 to 3 orders of magnitude depending on the encoding way)

### 2. Pre-training autoencoders for embedding vector 

Here we show how encodings work

  * Single character encodings overfitting compression?? (some evaluation needs to be done here)
  * Universal word/token level encodings 

Nevertheless there is the need to decode later, for this will depend on the application but a few things can be named (maybe a couple of examples too)
  - one-hot for categories

### 3. Practical Applications


#### Dataset ConLLu files

Check latest available conllu dataset, see if computational resources are enough for tagging ALL languages, if not, try to select a character subset that uses as many languages as possible

Start using it for NER, PoS, Coref ... Tagging
Show applications not for a single language but for all of them together (which is the bennefit of )


I've already done some tests on this, but basically needs redoing and more work


#### Language Translation

Encode and decode in different languages, see how this encoding can make encoding more universal


## (Far) Future

The idea behind all this is to set the basis for another NLP work wich I call NeuralDB where many of these encodings are saved in a DB (I've started looking at RocksDB as a backend)

