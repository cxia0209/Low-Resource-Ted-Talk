#!/usr/bin/env bash
python nmt.py \
    decode \
    --cuda \
    model\
    concat_data/test.en-az.az.txt \
    concat_data/test.en-az.en.txt \
    decode.txt