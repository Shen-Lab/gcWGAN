import sys

FILE = sys.argv[1]
FOLD = sys.argv[2]
LOWER_BOUND = int(sys.argv[3])
UPPER_BOUND = int(sys.argv[4])

with open(FILE,'r') as file_r:
    lines = file_r.readlines()

l = len(lines)

for i in xrange(l):
    line = lines[i]
    if '>' in line:
        if (i != 0) and ( fold == FOLD and len(seq) >= LOWER_BOUND and len(seq) <= UPPER_BOUND ):
            print title
            print len(seq)
            print seq
            print ''
        title = line.strip('\n')
        fold = line.split(' ')[1].split('.')
        fold = fold[0] + '.' + fold[1]
        seq = ''
    else:
        seq += line.strip('\n')

if ( fold == FOLD and len(seq) >= LOWER_BOUND and len(seq) <= UPPER_BOUND ):
    print title
    print len(seq)
    print seq
    print ''
