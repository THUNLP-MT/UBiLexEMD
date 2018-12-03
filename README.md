# UBiLexEMD: An Unsupervised Bilingual Lexicon Inducer From Non-Parallel Data by Earth Mover's Distance Minimization #

This software can produce a bilingual lexicon from non-parallel data without any cross-lingual supervision. It does so by learning a transformation between source word embeddings and target ones by earth mover's distance minimization. The technique is described in the following paper:

> Meng Zhang, Yang Liu, Huanbo Luan, and Maosong Sun. Earth Mover's Distance Minimization for Unsupervised Bilingual Lexicon Induction. In Proceedings of EMNLP, 2017.

## Runtime Environment ##

This software has been tested in the following environment, but should work in a compatible one.

- 64-bit Linux
- Python 2.7 (for WGAN code in the `src` folder)
	- Theano
	- Lasagne ([bleeding-edge version](http://lasagne.readthedocs.io/en/latest/user/installation.html#bleeding-edge-version) needed as of April, 2017)
	- scikit-learn
- Python 3.4 (for the code in the `scripts` folder)
- Matlab R2010b (for EMDOT code)

## Usage ##

### Preparation ###

1\. Specify the variables in the `config` file. For example, if `config` contains the following lines:

	config=zh-en
	lang1=zh
	lang2=en

then the data should be located in `data/zh-en` with file extensions `zh` and `en`. Prepare the `matlab.config` file accordingly.

2\. Prepare the following data in the folder specified in Step 1:

- word2vec.zh/en: Word embeddings, which can be obtained by running word2vec on monolingual data.
- vocab-freq.zh/en: Space-separated word-frequency pairs.

Besides, prepare vocab.zh/en, vec.zh/en, count.zh/en from the above data.

### WGAN ###

Execute `./runWGAN.sh` to obtain a file named `W` that stores the transformation matrix.

### EMDOT ###

1\. Copy the `W` file produced by WGAN into `data/zh-en` (the folder specified in `config`). One such file is provided in this release.

2\. Launch Matlab. In the console, execute:

	loadData
	SinkhornWithTransformationInitFromData(X_s, X_t, weight_s, weight_t, length(weight_s), length(weight_t));
	exit

3\. Process the transport scheme to obtain the translations.

`./processFlow.sh 10`

4\. In `data/zh-en`, result.10 will contain translations of vocab.zh. For each source word, there will be multiple translations after the tab character, separated by space.

## Known Issue ##

It is recommended that the bleeding-edge version of Lasagne works with the latest development version of Theano. It has been tested on Lasagne version 0.2.dev1 and Theano version 0.9.0dev4.dev-RELEASE, but NaN may appear on Theano version 0.8.2.