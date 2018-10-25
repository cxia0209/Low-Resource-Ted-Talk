import torch

vectors_az = torch.load('MUSE/vectors-az.pth')
vectors_en = torch.load('MUSE/vectors-en.pth')

vectors_az['w2i'] = dict()
vectors_en['w2i'] = dict()

print(vectors_en['dico'].id2word[0])
print(vectors_en['dico'].id2word[1])
print(vectors_en['dico'].id2word[2])
#
# print(vectors_en['vectors'][vectors_en['dico'].id2word[0]])
# print(vectors_en['vectors'][vectors_en['dico'].id2word[1]])
# print(vectors_en['vectors'][vectors_en['dico'].id2word[2]])
