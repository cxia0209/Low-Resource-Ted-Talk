#!/usr/bin/env bash
python vocab.py --train-src=data/train.en-tr.tr.txt --train-tgt=data/train.en-tr.en.txt data/vocab.tr-en.bin

python nmt.py train --train-src=data/train.en-tr.tr.txt --train-tgt=data/train.en-tr.en.txt \
--dev-src=data/dev.en-tr.tr.txt --dev-tgt=data/dev.en-tr.en.txt \
--embed-src=embeddings/wiki.multi.tr.vec --embed-tgt=embeddings/wiki.multi.en.vec --vocab=data/vocab.tr-en.bin --batch-size=64

python -u nmt.py train --train-src=bpe_data/train.az-and-tr.bpe.txt --train-tgt=bpe_data/train.en.txt \
--dev-src=bpe_data/dev.az.bpe.txt --dev-tgt=bpe_data/dev.en-az.en.txt --vocab=bpe_data/vocab.bin --batch-size=32 &> log.txt &

python -u nmt.py train --train-src=concat_data/train.src.txt --train-tgt=concat_data/train.tgt.txt --dev-src=concat_data/dev.en-az.az.txt --dev-tgt=concat_data/dev.en-az.en.txt --vocab=concat_data/vocab.bin --batch-size=16 &> log_orig.txt &

