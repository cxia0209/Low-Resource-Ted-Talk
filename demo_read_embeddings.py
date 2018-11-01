import torch

vectors_az = torch.load('embeddings/vectors-az.pth')
vectors_en = torch.load('embeddings/vectors-en.pth')

vectors_az['w2i'] = dict()
vectors_en['w2i'] = dict()
#
# print(vectors_en['dico'].id2word[0])
# print(vectors_en['dico'].id2word[1])
# print(vectors_en['dico'].id2word[2])
print(type(vectors_en['vectors']) == torch.Tensor)

# print(vectors_en['vectors'][vectors_en['dico'].id2word[0]])
# print(vectors_en['vectors'][vectors_en['dico'].id2word[1]])
# print(vectors_en['vectors'][vectors_en['dico'].id2word[2]])
