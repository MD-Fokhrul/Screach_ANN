
Introduction
============

[ <span>Link to
code</span>](https://github.com/ajamjoom/Image-Captions)\

Recent developments in Image Captioning have been inspired by
advancements in object detection and machine translation in the past few
years. The task of image captioning involves two main aspects: (1)
resolving the object detection problem in computer vision and (2)
creating a language model that can accurately generate a sentence
describing the detected objects.\

Seeing the success of encoder-decoder models with soft attention in the
task of image caption (Xu et al., 2016), we use soft alignment (Bahdanau
et al., 2014) and modern approaches to object detection (Ba et al.,
2014) as our baseline model. To extend this work, we investigate the
effect of pre-trained embeddings on the task of image captioning by
integrating GloVe embeddings (Pennington et al., 2014) and BERT context
vectors (Vaswani et al., 2017) to enhance the models performance and
reduce training time.\

The contributions of this paper are the following:

-   We provide an enhanced pyTorch implementation of the “soft”
    deterministic attention mechanism with an encoder-decoder
    architecture for image captioning as described in Xu et al, (2016)

-   We integrate BERT context vectors and GloVe embeddings into our
    baseline model and enhance it’s performance.

-   Finally, we visualize our results and quantitatively validate our
    models with the MS COCO validation dataset. We show that our BERT
    model integration outperforms the baseline model described in Xu et
    al., (2016) while taking less time to train.\
    \
    \

Problem Description
===================

Given a single raw image, our goal is to generate a caption $\textbf{y}$
encoded as a one-hot vector corresponding to our vocabulary.

$$\textbf{y} = \{y_{1} ,..., y_{C} \} , y_{i}\in R^{V}$$

Where V is the size of the vocabulary and C is the length of the
caption.

Data
====

There are several easily accessible datasets for training and validating
models on the task of image captioning such as MS COCO, Flickr8k, and
Flickr30k. In this paper, we use the MS COCO 2014 dataset for both
training and validation. We utilized readily available MS COCO cleaning
and structuring scripts @pytorchTutorial1 @pytorchTutorial2 to parse the
captions, extract the vocabulary, and batch the images to optimize the
training process for our models.\

After re-sizing and normalizing all the images to 224x224 pixels, we
extract the captions and tokenized them with the NLTK tokenizer.
Following that, we built a vocabulary with all the training dataset
words, which came to be a list of 8,856 words.

Models and Algorithms
=====================

We use an encoder-decoder architecture to generate captions. The encoder
is a Convolutional Neural Network (CNN) that takes in a single image and
generates a vector describing the detected objects. This vector of
objects is then passed to the decoder, which is a Long Short-Term Memory
Network (LSTM) that attends to the image and outputs a descriptive
caption one word at each time step.\

In this section, we describe the encoder used in all our models and
three variants of the attention-based decoder from Xu et al, (2016). The
first decoder is an exact replica of the soft attention model described
in Xu et al, (2016) with optimized hyper-parameters. This initial model
will act as our baseline. The second and third decoders are extensions
on the baseline model that integrate GloVe embeddings and BERT’s
pre-trained context vectors to the captions to enhance the model’s
performance and reduce its training time.\

Encoder
-------

Similar to the encoder described in Xu et al, (2016), we use a CNN to
extract feature vectors from images. The Encoder produces L vectors,
where each vector has D-dimensions that represent part of the image.

$$\textbf{a} = \{ a_{1},...a_{L}\} \in R^{D}$$

Although it’s possible to create and train our own CNN for this task, we
used the pretrained ResNet-101 CNN as our encoder to reduce our training
time and focus on enhancing the performance of the decoder. To use
ResNet-101, we discard the pooling and linear layers - the last two
layers - as we only need the image encoding, rather than the image
classification. Then, we pass the output of the modified ResNet onto an
adaptive pooling layer to create a fixed size output vector - fixed L -
that can be easily passed to the decoder. We do not perform any
fine-tuning to ResNet-101.

Decoder: Baseline Attention Model
---------------------------------

We use a Long Short-Term Memory network to generate caption words one
step at a time by conditioning on the previous step’s hidden state, the
context vector, and the previously generated words. Our implementation
directly follows that of Xu et al. (2016) and Zaremba et al. (2014).

$$\begin{pmatrix}{i}\\{f}\\{o}\\{g}\\ \end{pmatrix} = \begin{pmatrix}{\sigma}\\{\sigma}\\{\sigma}\\{tanh}\\ \end{pmatrix} T_{D+m+n,n} 
\begin{pmatrix}{\textbf{Ey}_{t-1}}\\h_{t-1} \\{ z_{t}}  \end{pmatrix}$$
$$c_t = f \odot c_{t-1} + i \odot g$$ $$h_t = o \odot tanh(c_t)$$

i, f, o, and g are the input, forget, memory, and output states of the
LSTM. h is the hidden state, c is the cell state (that keeps long term
memory), $h_t$ denotes the hidden state at timestep $t$. Additionally,
$T_{n,m}$ defines an affine transformation from dimension n to m.
$\odot$ is an element-wise multiplication.\

$\textbf{E}$ represents an embedding matrix that is used and $z_t$
denotes the context vectors of the relevant part of the image at time
step $t$ that is generated through soft-attention. To perform
soft-attention in generating $z_t$, we use the “soft” attention
mechanism detailed in Xu et al. (2016). In this baseline model, the
caption embedding, $\textbf{E}$, is learned alongside training the model
to generate captions.\

In the next two decoder descriptions, we will explain how we optimize
$\textbf{E}$ to enhance the models performance.

Decoder: GloVe Attention Model
------------------------------

Recent methods for extracting and learning vector space representations
for words have proven successful in capturing fine-grained semantic and
syntactic regularities in words. In particular, GloVe word vectors
(Pennington et al., 2014) created a global log-bilinear regression model
that generates word vector representations that enable Machine learning
models to utalize these pre-trained embeddings. These embeddings are
helpful because they can be pre-trained on vast amounts of text data
instead of being trained alongside the task specific model, which
usually has a much smaller dataset.\

As an extension to the baseline decoder explained in the previous
section, we integrate GloVe embeddings into our decoder by apply them to
the images captions. However, we fine-tunned the embeddings along-side
training our model to increase it’s accuracy and make it better fit the
MS COCO dataset.\

We downloaded the gloev.6b 300-dimensions pre-trained embeddings
introduced in Pennington et al., (2014) and built a weights matrix that
has a GloVe embedding for every word in our vocabulary. We then
initialize the decoder embeddings with this weights matrix and
fine-tuned it as we trained our model by propagating back the gradients.

Decoder: BERT Attention Model
-----------------------------

In using the GloVe vector representations, each word is represented by a
single unique vector no-matter what context the word is used in. This
raised concerns for several researchers (Devlin et al., 2018) as they
realized that each word could have multiple meanings depending on where
the word is used. Rather than having one representation for each word,
BERT (Devlin et al., 2018) uses a Transformer to generate a
bi-directional contexualized word embeddings conditioned on the context
of the word in a sentence.\

BERT has two distinct models, BERT base and BERT large. The base version
has 12 encoder layers in it’s Trasnformer, 768 hidden unites in it’s
feedforward-network, and 12 attention heads. On the other hand, the
large version has 24 encoder layers in it’s Trasnformer, 1024 hidden
unites in it’s feedforward-network, and 16 attention heads. Throughout
our implementation, we use BERT base to generate the caption’s
contexualized word vectors due to the increase in training time the
large model introduces.\

In the decoder, we take a batch of captions as our input
$\textbf{c} = \{c_{1},...,c_{B}\}$, where $B$ is the size of the batch
and $c_{i}$ is a full text representation of the caption. Then we
iteratively take each caption $c_{i}$ and perform the following steps to
it:

1.  Tokenize each caption with BERT’s wordPiece tokenizer to enable BERT
    to digest the caption and add the special ’[CLS]’ BERT token to the
    beginning of the caption

2.  Pass the wordPieces into BERT base

3.  Retrieve the output of the 12th layer (last layer) and discard the
    embedding of the special ’[CLS]’ token

4.  Detokenize the embeddings by summing the BERT context vectors of
    wordPieces that belong to the same original word.

After doing the steps above to each caption in the batch we will have
caption embeddings $\textbf{b} = \{b_{1},...,b_{B}\}$, where $b_{i}$ is
a tensor of size (caption size x 768) as each word has a vector of size
768 as it’s contexualized embedding. $\textbf{b}$ can then directly
replace the GloVe embeddings and the trained embeddings used in the
baseline model and the GloVe model respectively.

Experiments and Results
=======================

The Baseline, GloVe, and BERT models were trained and validated using
the MS COCO 2014 dataset with the following optimized hyper-parameters
obtained from Xu et al, (2016) : (1) gradient clip = 5 (avoid gradient
explosion), (2) number of epochs = 4 (limited to 4 epochs due to GPU
accessibility), (3) batch size = 32, (4) decoder learning rate = 0.0004,
(5) dropout rate = 0.5, (6) vocab size = 8856, (7) encoder dimension =
2,048 (based on RESNET-101’s output size), (8) attention dimension =
512, and (9) All weights were initialize using a uniform distribution
with range = [-0.1,0.1].\

When it came down to implementing the embedding extensions, we used
embedding dimension of 512 for the baseline model, 300 for GloVe, and
768 for BERT. All the models were trained and validated on the same
dataset splits with the same vocabulary to enable an accurate
performance comparison.\

Each epoch in the baseline and GloVe model took around 3.5 hours to
train on a GTX 1070 Ti GPU, while the BERT model’s epoch took around 4.2
hours.

Baseline Attention Model
------------------------

Figure 1 shows sample results obtained from the baseline model.
Qualitatively analyzing the results, we see that the left image of
Figure 1 has an accurate, grammatically correct hypotheses although the
animal classifications are incorrect. Additionally, we see that the
model used the word “herd” where is should have been “flock” due to it’s
limited language model. The right image of Figure 1 shows a hypothese
that is more similar to the average hypotheses generated by the baseline
model where many word repetitions occur. This means that the model
correctly learned some representations, but didn’t finish training yet.

![image](baseline_amazing.png) ![(left) successful hypothesis from the
baseline model. (right) incorrect hypothesis from the baseline
model](surf.png "fig:")

We have additionally noticed that this model is unable to generate
sentences that have the same meaning as the reference sentences while
using different words, it seemed that the model is attempting to copy
the reference sentence word by word. This can be explained by the fact
that no pre-trained embeddings were used in this model. Therefore, it
was difficult for the model to learn accurate word representations that
would allow it to switch similar words.

GloVe Attention Model
---------------------

Although the Glove model has similar quantitative results to the
baseline model in validation loss and BLEU scores, the GloVe model
proved to be able to generate captions that use a different style of
writing than the reference captions by using different words that are
similar in meaning. This change can be explained by the fact that Glove
embeddings offer the model the ability to pick and choose the best
possible word from a cluster of similar words. The left image in Figure
2 shows can example where the words ’Asian people’ was translated to
’children’ in the generated caption. However, this model has similar
repetition problems as the baseline model due to the limited training we
did.\

![image](eating.png) ![(left) decent hypothesis from the GloVe model.
(right) incorrect hypothesis from the GloVe
model](glove_not_great.png "fig:")

BERT Attention Model
--------------------

Although we expected the BERT model to outperform both the baseline and
GloVe models because of it’s reliance on contexualized word embbedings,
we were surprised by the extent of increase in BLEU score it obtained.
The validation loss (Cross Entropy) decreased much faster in the BERT
model, which shows that the BERT embeddings are very accurate in
representing contexualized words. Additionally, this shows that context
is very important in generating image captions, which makes a lot of
sense because captions are supposed to relate objects in an image
together and make sense of them.

![image](BERT_change.png) ![Accurate captions by the BERT
model](BERT_solid.png "fig:")

Figure 3 shows two example of the BERT model predicting captions, both
captions make sense and are grammatically correct. Interestingly, the
predicted captions (hypotheses) predict similar sentences to the
reference caption but use different word variants to explain similar
things. In Figure 3, we see that ’back window’ was translated to ’back’
and ’holidays’ was translated to ’Christmas’. This is interesting as it
shows that the model offers correct captions that are not exactly the
same as the reference captions, which is the goal of this task. BERT
captions had few repetitions and were generally very well written.

Summary of findings
-------------------

Table 1 shows a full summary of our results. We validated our dataset by
running a full single epoch on the MS COCO 2014 validation data and
compared each hypotheses to 5 reference captions. Table 1 reports the
validation loss, BLEU-1, BLEU-2, BLEU-3, and BLEU-4 which gives an
accurate indication of how well each model performs.

[H]

<span>lrrrrrr</span> Model & Val (loss) & BLEU-1 & BLEU-2 & BLEU-3 &
BLEU-4\
Baseline Model & 3.452 & 48.51 & 18.84 & 7.406 & 3.097\
GloVe Model & 3.325 & 49.70 & 20.07 & 8.214 & 3.552\
BERT Model & **1.901** & **78.27** & **59.53** & **46.22** & **36.53**\

Baseline and GloVe yielded very similar results to each other, having
only a slight improvement with Glove due to the introduction of
pre-trained embeddings that were trained on vast amounts of data. BERT,
on the other hand, had much better results in all aspects as shown in
Table 1. Our BERT model results outperformed the results obtained by Xu
et al (2016) while being trained on fewer epochs @pytorchTutorial1.

![image](duck.png) ![Direct comparison of the three main models
proposed](plate.png "fig:")

Figure 4 shows a direct a comparison between the three models we
implemented. The results clearly show that Baseline and GloVe yield
similar results, while BERT outperforms them by large margins.

![Failed attempt to visualizing attention](failed_viz.png)

Figure 5 shows a failed attempt of implementing a program that
visualizes the attention and maps it onto the image so that we can see
what the model is attending to while predicting a specific word. The
issue we have is in how we map the attention values onto the image.

Conclusion
==========

We propose two extentions to the attention based approach to image
captioining introduced in xu et al, (2016) that enhanced the performance
of the model and reduced training time. Our BERT approach surpasses the
MS COCO validation scores obtained by Xu et al, (2016) while being
trained on fewer epochs with the same hyper-parameters. Our experiments
outline the important of word embeddings in nature language processing
and offer a new method in integrating BERT with already developed models
to enhance their performance.\

Possible extensions to our work would be to train a new model with BERT
large as apposed to BERT base, utilize beam search in validation, and
train the models until the training loss converges.

<span>9</span> Kelvin Xu, Jimmy Lei Ba, Ryan Kiros, Kyunghyun Cho ,
Aaron Courville, Ruslan Salakhutdinov, Richard S. Zemel, Yoshua Bengio.
Show, Attend and Tell: Neural Image Caption Generation with Visual
Attention. arXiv:1502.03044v3, April 2016 Bahdanau, Dzmitry, Cho,
Kyunghyun, and Bengio, Yoshua. Neural machine translation by jointly
learning to align and translate. arXiv:1409.0473, September 2014. Ba,
Jimmy Lei, Mnih, Volodymyr, and Kavukcuoglu, Koray. Multiple object
recognition with visual attention. arXiv:1412.7755, December 2014. Jacob
Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova BERT:
Pre-training of Deep Bidirectional Transformers for Language
Understanding arXiv:1810.04805, Oct 2018 Jeffrey Pennington, Richard
Socher, Christopher D. Manning GloVe: Global Vectors for Word
Representation Proceedings of the 2014 Conference on Empirical Methods
in Natural Language Processing (EMNLP), 2014 Show, Attend, and Tell. A
PyTorch Tutorial to Image Captioning
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
https://github.com/parksunwoo/show\_attend\_and\_tell\_pytorch
