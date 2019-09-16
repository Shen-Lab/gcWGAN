import collections
import numpy as np
import re
from itertools import izip

def tokenize_string(sample):
    return tuple(sample.lower().split(' '))

class NgramLanguageModel(object):
    def __init__(self, n, samples, tokenize=False):
        if tokenize:
            tokenized_samples = []
            for sample in samples:
                tokenized_samples.append(tokenize_string(sample))
            samples = tokenized_samples

        self._n = n
        self._samples = samples
        self._ngram_counts = collections.defaultdict(int)
        self._total_ngrams = 0
        for ngram in self.ngrams():
            self._ngram_counts[ngram] += 1
            self._total_ngrams += 1

    def ngrams(self):
        n = self._n
        for sample in self._samples:
            for i in xrange(len(sample)-n+1):
                yield sample[i:i+n]

    def unique_ngrams(self):
        return set(self._ngram_counts.keys())

    def log_likelihood(self, ngram):
        if ngram not in self._ngram_counts:
            return -np.inf
        else:
            return np.log(self._ngram_counts[ngram]) - np.log(self._total_ngrams)

    def kl_to(self, p):
        # p is another NgramLanguageModel
        log_likelihood_ratios = []
        for ngram in p.ngrams():
            log_likelihood_ratios.append(p.log_likelihood(ngram) - self.log_likelihood(ngram))
        return np.mean(log_likelihood_ratios)

    def cosine_sim_with(self, p):
        # p is another NgramLanguageModel
        p_dot_q = 0.
        p_norm = 0.
        q_norm = 0.
        for ngram in p.unique_ngrams():
            p_i = np.exp(p.log_likelihood(ngram))
            q_i = np.exp(self.log_likelihood(ngram))
            p_dot_q += p_i * q_i
            p_norm += p_i**2
        for ngram in self.unique_ngrams():
            q_i = np.exp(self.log_likelihood(ngram))
            q_norm += q_i**2
        return p_dot_q / (np.sqrt(p_norm) * np.sqrt(q_norm))

    def precision_wrt(self, p):
        # p is another NgramLanguageModel
        num = 0.
        denom = 0
        p_ngrams = p.unique_ngrams()
        for ngram in self.unique_ngrams():
            if ngram in p_ngrams:
                num += self._ngram_counts[ngram]
            denom += self._ngram_counts[ngram]
        return float(num) / denom

    def recall_wrt(self, p):
        return p.precision_wrt(self)

    def js_with(self, p):
        log_p = np.array([p.log_likelihood(ngram) for ngram in p.unique_ngrams()])
        log_q = np.array([self.log_likelihood(ngram) for ngram in p.unique_ngrams()])
        log_m = np.logaddexp(log_p - np.log(2), log_q - np.log(2))
        kl_p_m = np.sum(np.exp(log_p) * (log_p - log_m))

        log_p = np.array([p.log_likelihood(ngram) for ngram in self.unique_ngrams()])
        log_q = np.array([self.log_likelihood(ngram) for ngram in self.unique_ngrams()])
        log_m = np.logaddexp(log_p - np.log(2), log_q - np.log(2))
        kl_q_m = np.sum(np.exp(log_q) * (log_q - log_m))

        return 0.5*(kl_p_m + kl_q_m) / np.log(2)

def load_dataset(max_length, max_n_examples, tokenize=False, max_vocab_size=2048, data_dir='/home/ishaan/data/1-billion-word-language-modeling-benchmark-r13output'):
    print "loading dataset..."

    lines = []

    finished = False

    for i in xrange(99):
        path = data_dir+("/training-monolingual.tokenized.shuffled/news.en-{}-of-00100".format(str(i+1).zfill(5)))
        with open(path, 'r') as f:
            for line in f:
                line = line[:-1]
                if tokenize:
                    line = tokenize_string(line)
                else:
                    line = tuple(line)

                if len(line) > max_length:
                    line = line[:max_length]

                lines.append(line + ( ("`",)*(max_length-len(line)) ) )

                if len(lines) == max_n_examples:
                    finished = True
                    break
        if finished:
            break

    np.random.shuffle(lines)

    counts = collections.Counter(char for line in lines for char in line)

    charmap = {'unk':0}
    inv_charmap = ['unk']

    for char,count in counts.most_common(max_vocab_size-1):
        if char not in charmap:
            charmap[char] = len(inv_charmap)
            inv_charmap.append(char)

    filtered_lines = []
    for line in lines:
        filtered_line = []
        for char in line:
            if char in charmap:
                filtered_line.append(char)
            else:
                filtered_line.append('unk')
        filtered_lines.append(tuple(filtered_line))

    for i in xrange(100):
        print filtered_lines[i]

    print "loaded {} lines in dataset".format(len(lines))
    return filtered_lines, charmap, inv_charmap


def tokenize_seq(sample): # should be checked, has not been checked yet!!!
    return tuple(sample.lower().split(''))



def load_dataset_protein(max_length, max_n_examples, tokenize=False, max_vocab_size=21, data_dir='../Data/Datasets/Final_Data/'):
    print "loading dataset..."


    pad = "!" # use ! for padding
    folds_dict = {}
    path = data_dir+"folds_coordinate_new"
    with open(path, 'r') as f:
            for line in f:
                line = line.strip().split()
                key = line[0]
                value = line[1:]
                value = [float(v) for v in value]
                folds_dict[key] = value

    seqs = []
    folds = []
    finished = False
    with open(data_dir+"seq_train") as file1, open(data_dir+"fold_train") as file2: 
    #with open(data_dir+seq_file) as file1, open(data_dir+fold_file) as file2:   #SZ change      
            for s, f in izip(file1, file2):
                s = s.strip()
                f = f.strip()

                if tokenize:
                    s = tokenize_seq(s)
                else:
                    s = tuple(s)
                if len(s) > max_length:
                    continue
                
                s = s + ( (pad,)*(max_length-len(s)) ) # padding
                seqs.append(s)  
                
                f = folds_dict[f]
                folds.append(f)
                if len(seqs) == max_n_examples:
                    finished = True
                    break

    indices = np.arange(len(seqs),dtype=np.int)
    np.random.shuffle(indices)
    seqs =  [ seqs[i] for i in indices]
    folds =  [ folds[i] for i in indices] 
   
    charmap = {'!':0,'a':1,'r':2,'n':3,'d':4,'c':5,'q':6,'e':7,'g':8,
               'h':9,'i':10,'l':11,'k':12,'m':13,'f':14,'p':15,'s':16,
               't':17,'w':18,'y':19,'v':20}
    inv_charmap = ['!','a','r','n','d','c','q','e','g','h','i',
                   'l','k','m','f','p','s','t','w','y','v']
    """
    charmap = {'!':0,'A':1,'R':2,'N':3,'D':4,'C':5,'Q':6,'E':7,'G':8,
               'H':9,'I':10,'L':11,'K':12,'M':13,'F':14,'P':15,'S':16,
               'T':17,'W':18,'Y':19,'v':20}
    inv_charmap = ['!','A','R','N','D','C','Q','E','G','H','I',
                   'L','K','M','F','P','S','T','W','Y','V']
    # SZ change
    """
    print "loaded {} lines in dataset".format(len(seqs))
    return seqs, folds, folds_dict, charmap, inv_charmap

def load_dataset_protein_pretrain(max_length, max_n_examples, tokenize=False, max_vocab_size=21, data_dir='../Data/Datasets/Final_Data/'):
    print "loading dataset..."

    pad = "!" # use ! for padding
    seqs = []
    finished = False

    with open(data_dir+"seq_50_uniref-0.5") as file1:
    #with open(data_dir+seq_file) as file1, open(data_dir+fold_file) as file2:   #SZ change      
            for s in file1:
                s = s.strip()

                if tokenize:
                    s = tokenize_seq(s)
                else:
                    s = tuple(s)
                if len(s) > max_length:
                    continue

                s = s + ( (pad,)*(max_length-len(s)) ) # padding
                seqs.append(s)

                if len(seqs) == max_n_examples:
                    finished = True
                    break
    indices = np.arange(len(seqs),dtype=np.int)
    np.random.shuffle(indices)
    seqs =  [ seqs[i] for i in indices]
  
    print "loaded {} lines in dataset".format(len(seqs))
    return seqs
