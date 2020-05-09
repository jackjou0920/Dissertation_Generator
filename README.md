![image](https://github.com/jackjou0920/Dissertation_Generator/blob/master/img/logo.png)
# Dissertation_Generator
It is a dissertation research for my Master degree in UK. The project implemented a Natural Language Generation (NLG) task: generating an meaningful and relevant abstract of dissertation given any topic in the certain domain, through the state-of-the-art neural network model, Transformer, by Tensorflow package.

## Requirements
* Python 3.6 with NLTK, Numpy, pdfminer, and mainly Tensorflow 2.0 environment
* Multiple GPU with over 32G memories
* Theories of machine learning, especially, neural network applied on NLP for understanding encoder-decoder, language modeling, and attention and self-attention mechanisms

## Architechture
### Scaled dot product attention
<img src="https://github.com/jackjou0920/Dissertation_Generator/blob/master/img/dot_product.png" width="300" />

### Multi-head attention
<img src="https://github.com/jackjou0920/Dissertation_Generator/blob/master/img/multi-attention.png" width="300" />

### Encoder and decoder
The core idea behind the Transformer model is self-attention whose entire architechture as below.
<img src="https://github.com/jackjou0920/Dissertation_Generator/blob/master/img/self-attention.png" width="600" />

## Features
* Crawled and pre-processed the previous dissertation in the department for being the dataset
* Utilised NVIDIA GPU for training to express the parallelisation of the Transformer architechture
* Achieved the high relevant output with the maximum 700 words
* Analysed and criticised variety of innovative models and techniques proposed by leading companies or organisations including GPT-2, BERT and Transformer-XL
* The BLEU score in certain case can obtain between 0.24 and 0.33

## Results
#### Example 1
![image](https://github.com/jackjou0920/Dissertation_Generator/blob/master/img/example1.png)

#### Example 2
![image](https://github.com/jackjou0920/Dissertation_Generator/blob/master/img/example2.png)

#### Example 3
![image](https://github.com/jackjou0920/Dissertation_Generator/blob/master/img/example3.png)

