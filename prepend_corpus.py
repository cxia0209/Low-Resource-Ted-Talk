import sys
if __name__ == '__main__':
    input, output, prefix = sys.argv[1:]
    writer = open(output, 'w')
    for line in open(input):
        sent = line.strip().split(' ')
        sent_out = [prefix + '_' + token for token in sent]
        writer.write(' '.join(sent_out) + '\n')


