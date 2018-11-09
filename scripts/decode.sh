python nmt.py decode model bpe_az_data/test.az.bpe.txt bpe_az_data/test.en.bpe.txt decode_words.txt
perl multi-bleu.perl bpe_az_data/test.en-az.en.txt < decode_words.txt
