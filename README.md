You will have access to TED talk parallel data for 3 low-resource languages: Galician (gl), Azerbaijani (az), and Belarusian (be), derived from ted_talks.tar.gz from the following repository:
> https://github.com/neulab/word-embeddings-for-nmt
Using, and improving upon, your implementation from the first assignment, please attempt to get the best accuracy possible on the low-resource language pairs. This can be done both through architectural improvements, and through semi-supervised learning or cross-lingual transfer from the data sources listed below.

The first data source is TED data in other languages. The TED talks corpus contains data for a bunch of other languages (~60). We have prepared this data in high-resource languages, Portuguese (pt), Turkish (tr), and Russian (ru), that pair with each of the low resource languages (gl with pt, az eith tr, and ge with ru). These languages are helper languages that can improve the results on the low-resource ones as they are highly related. You can use this data for training, or use any of the other training data in ted_talks.tar.gz.

In addition, we provide monolingual data (tokenized Wikipedia) that you can use in semi-supervised learning methods.

Out baseline model gives the following BLEU scores when trained on either the TED data in the target language, or the TED data in the target+related language, which you should aim to signficantly improve upon:

az-en: 3.0
az-en (with tr): 7.1

be-en: 5.4
be-en (with ru): 11.6

gl-en: 16.5
gl-en (with pt): 22.0

Note that we have provided a script, get_wikipedia.sh, that can be used to download wikipedia and process more languages which might be useful. To use, just clone the following repos in your working directory:

> https://github.com/moses-smt/mosesdecoder
> https://github.com/attardi/wikiextractor

And use the following command: sh get_wikipedia.sh <language> where <language> is a two-letter language code. For example: sh get_wikipedia.sh gl will download and process the Galician wikipedia.

We also provide the script we used to extract the parallel data from the files in ted_talks.tar.gz, extract_ted_talks.py, which could be useful in extracting data for other languages.
