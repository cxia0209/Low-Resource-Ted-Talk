import sentencepiece as spm
import argparse


parser = argparse.ArgumentParser(description='Script for parsing...')
parser.add_argument('model_path', type=str)
parser.add_argument('decode', type=str)
args = parser.parse_args()

sp = spm.SentencePieceProcessor()
sp.Load(args.model_path)
hypo = []
with open(args.decode, 'r') as f:
    for line in f.readlines():
        hypo.append(sp.DecodePieces(line.split()))

with open('decode_words.txt', 'a') as f:
    for sent in hypo:
        f.write(sent + '\n')
