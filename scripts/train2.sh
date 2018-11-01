#!/usr/bin/env bash
python vocab.py --train-src=data/train.en-tr.tr.txt --train-tgt=data/train.en-tr.en.txt data/vocab.tr-en.bin

python nmt.py train --train-src=data/train.en-tr.tr.txt --train-tgt=data/train.en-tr.en.txt \
--dev-src=data/dev.en-tr.tr.txt --dev-tgt=data/dev.en-tr.en.txt \
--embed-src=embeddings/wiki.multi.tr.vec --embed-tgt=embeddings/wiki.multi.en.vec --vocab=data/vocab.tr-en.bin --batch-size=64
