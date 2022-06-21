from random import randint
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
from timeit import default_timer as timer

plt.ion()

#np.set_printoptions(threshold=np.inf)

class BloomFilter(object):

    '''
    Class for Bloom filter, using murmur3 hash function
    '''

    def __init__(self, items_count = 5868, fp_prob = 0.0001):
        '''
        items_count : int
            Number of items expected to be stored in bloom filter
        fp_prob : float
            False Positive probability in decimal
        '''

        self.prime = 20011
        self.max_val = 20000

        # False posible probability in decimal
        self.fp_prob = fp_prob

        # Size of bit array to use
        self.size = self.get_size(items_count, fp_prob)
        #self.size = array_size

        # number of hash functions to use
        #self.hash_count = hash_functions.size
        self.hash_count = self.get_hash_count(self.size, items_count)

        self.hash_functions = self.get_hash_functions(self.max_val, self.hash_count)

        # Bit array of given size
        self.count_array = [0 for i in range(self.size)]

    def add(self, item):
        '''
        Add an item in the filter
        '''
        #digests = []
        # for i in range(self.hash_count):

        #     # create digest for given item.
        #     # i work as seed to mmh3.hash() function
        #     # With different seed, digest created is different
        #     digest = mmh3.hash(item,i) % self.size
        #     digests.append(digest)

        #     # set the bit True in bit_array
        #     self.bit_array[digest] = True

        for hash_id, hash_vals in enumerate(self.hash_functions):
            a, b = hash_vals

            # pass `val` through the `ith` permutation function
            output = int((a * item + b) % self.prime % self.size)
            #print(output)

            # conditionally update the `ith` value of vec
            self.count_array[output] += 1

    # def check(self, item):
    #     '''
    #     Check for existence of an item in filter
    #     '''
    #     for i in range(self.hash_count):
    #         digest = mmh3.hash(item, i) % self.size
    #         if self.bit_array[digest] == False:
    #
    #             # if any of bit is False then,its not present
    #             # in filter
    #             # else there is probability that it exist
    #             return False
    #     return True

    def get_count(self, item):
        count = 30000000

        for hash_id, hash_vals in enumerate(self.hash_functions):
            a, b = hash_vals

            # pass `val` through the `ith` permutation function
            output = int((a * item + b) % self.prime % self.size)
            #print(output)

            if (self.count_array[output] < count):
                count = self.count_array[output]

            # conditionally update the `ith` value of vec
        return count

    @classmethod
    def get_size(self, n, p):
        '''
        Return the size of bit array(m) to used using
        following formula
        m = -(n * lg(p)) / (lg(2)^2)
        n : int
            number of items expected to be stored in filter
        p : float
            False Positive probability in decimal
        '''
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return int(m)

    @classmethod
    def get_hash_count(self, m, n):
        '''
        Return the hash function(k) to be used using
        following formula
        k = (m/n) * lg(2)

        m : int
            size of bit array
        n : int
            number of items expected to be stored in filter
        '''
        k = (m / n) * math.log(2)
        return int(k)

    def get_hash_functions(self, max_val, k):
        '''
        Generate K universal hash functions to be used in the bloom filter
        m : int
            size of bit array
        n : int
            number of items expected to be stored in filter
        '''
        perms = [(randint(0, max_val), randint(0, max_val)) for i in range(k)]
        return perms

# specify the length of each minhash vector
#N = 100
#max_val = (2**32)-1
max_val = 20000

# initialize a sample minhash vector of length N
# each record will be represented by its own vec
#vec = [float('inf') for i in range(N)]

#def minhash(s, prime=4294967311):
def minhash(s, perms, prime=20011, N = 100):
    '''
    Given a set `s`, pass each member of the set through all permutation
    functions, and set the `ith` position of `vec` to the `ith` permutation
    function's output if that output is smaller than `vec[i]`.
    '''
    # initialize a minhash of length N with positive infinity values
    vec = [300000000 for i in range(N)]

    for id_val, val in enumerate(s):

        # ensure s is composed of integers
        if val == 1:

            # loop over each "permutation function"
            for perm_idx, perm_vals in enumerate(perms):
                a, b = perm_vals

                # pass `val` through the `ith` permutation function
                output = (a * id_val + b) % prime

                # conditionally update the `ith` value of vec
                if vec[perm_idx] > output:
                    vec[perm_idx] = output

    # the returned vector represents the minimum hash of the set s
    return vec

# doc1 = [1,1,1,0,0,1,1,0,0,1,1,1,1,0,0,1,1,0,0,1]
# doc2 = [1,1,0,0,0,1,1,0,0,1,1,1,1,0,0,1,1,0,0,1]

# minhash1 = minhash(doc1)
# minhash2 = minhash(doc2)

# print (minhash1)
# print (minhash2)

################################################

def getStats(pred_y, test_y_outlier, test_y_not_outlier):
    overlap_TP = set(pred_y) & set(test_y_outlier)
    TP = len(overlap_TP)
    #print (TP)

    overlap_TN = set(test_y_not_outlier) - set(pred_y)
    TN = len(overlap_TN)
    #print (TN)

    overlap_FP = set(pred_y) & set(test_y_not_outlier)
    FP = len(overlap_FP)
    #print (FP)

    overlap_FN = set(test_y_outlier) - set(pred_y)
    FN = len(overlap_FN)
    #print (FN)

    return (TP, TN, FP, FN)

def getPrecisionRecall(TP, TN, FP, FN):
    if TP == 0:
        return (0, 0)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    return (float(format(precision, '.4f')), float(format(recall, '.4f')))

def getROC(TP, TN, FP, FN):
    if TP == 0 or FP == 0:
        return (0, 0)

    tpr = TP / (TP + FN)
    fpr = FP / (FP + TN)

    return (float(format(tpr, '.4f')), float(format(fpr, '.4f')))

f=open('reuters_shuffled_y.txt', 'r')
lines=f.readlines()
test_y_outlier=[]
test_y_not_outlier=[]
for x in lines:
    x_spl = x.split(',')[1]
    if float(x_spl.rstrip()) == 1:
        test_y_outlier.append(int(x.split(',')[0]))
    if float(x_spl.rstrip()) == 0:
        test_y_not_outlier.append(int(x.split(',')[0]))
f.close()

#################################################

auc_ver = []
roc_ver = []

for num_hash in np.arange(10, 401, 10):
    for num_bin in np.arange(10, 401, 10):
        build_timer_A = timer()
        output_timer_A = timer()

        graph_title = 'Num Hashes (' + str(num_hash) + ') Bin Size (' + str(num_bin) + ')'
        fig_name = 'Num Hashes_' + str(num_hash) + '_Bin Size_' + str(num_bin)  + '.png'

        # create N tuples that will serve as permutation functions
        # these permutation values are used to hash all input sets
        perms = [(randint(0, max_val), randint(0, max_val)) for i in range(num_hash)]

        doc_list = []
        input = np.loadtxt("reuters_shuffled.txt", delimiter=',', dtype=int)
        input_docs = input[:, 1:]
        ids = input[:, 0]
        #ids = list(range(5868))
        #print(ids)

        for i in input_docs:
            minhashed_doc = minhash(i, perms, N = num_hash)
            doc_list.append(minhashed_doc)

        #scaler = MinMaxScaler()
        #scaler.fit_transform(doc_list)

        with open('reuters_signatures_' + graph_title + '.txt', 'w') as f:
            for item in doc_list:
                string = ','.join(map(str, item))
                for item in string:
                    f.write(item)
                f.write('\n')

        #################################################

        bin_members = []
        bin_list = []

        with open('reuters_signatures_' + graph_title + '.txt') as f:
            with open('reuters_binned_' + graph_title + '.txt', 'w') as f1:
                for line in f:
                    #num_bin = 5000

                    for p in line.split(','):
                        curr_bin = int(p) // (max_val // num_bin)
                        bin_members.append(curr_bin)
                        #print(str(curr_bin) + ',', end='')
                    bin_name = ''.join(str(e) for e in bin_members)
                    f1.write(str(bin_name) + '\n')
                    bin_list.append(bin_name)
                    bin_members = []
                    #if i > 0:
                        #f1.write(str(dim_sum) + "\n")

        #print(bin_list)

        bin_list = list(map(int, bin_list))

        bf = BloomFilter()

        print("Counter Started")

        mat = np.column_stack((ids, bin_list))

        for i in mat[:, 1]:
            bf.add(i)



        #dic = Counter(bin_list)
        bin_count = []

        for i in mat[:, 1]:
            bin_count.append(bf.get_count(i))
        # for bin in bin_list:
        #     bin_count.append(dic.get(bin))

        mat = np.column_stack((mat, bin_count))
        #print(mat)

        #mat_sorted = mat[mat[:,2].argsort()[::]]
        mat_sorted = sorted(mat, key=lambda x:(x[2],x[1]))

        mat_sorted = np.stack(mat_sorted, axis=0)

        np.savetxt('mat_sorted_' + graph_title + '.txt', mat_sorted, fmt="%s")

        build_timer_B = timer()

        print("mat_sorted is ready")

        ##############################################

        precision_ver = []
        recall_ver = []
        tpr_ver = []
        fpr_ver = []

        for i in np.arange(10, 5868, 10):
            out_bins = mat_sorted[:i, 0]
            #for bin in bin_list:
            #    if (bin in out_bins):
            #        pred_ids.append(1)
            #    else:
            #        pred_ids.append(0)

            TP, TN, FP, FN = getStats(list(map(int, out_bins)), test_y_outlier, test_y_not_outlier)
            precision, recall = getPrecisionRecall(TP, TN, FP, FN)
            precision_ver.append(precision)
            recall_ver.append(recall)

            tpr, fpr = getROC(TP, TN, FP, FN)
            tpr_ver.append(tpr)
            fpr_ver.append(fpr)
            #print (out_bins)
            #print(tpr_ver)
            #print(fpr_ver)
            #print ("Finished iteration %d" % i)
            
            if i == 10:
                output_timer_B = timer()

        output_timer = output_timer_B - output_timer_A
        build_timer = build_timer_B - build_timer_A

        curr_auc = [format(build_timer, '.2f'), format(output_timer, '.2f'), format(num_hash, '.1f'), format(num_bin, '.1f'), float(format(np.trapz(precision_ver, recall_ver), '.5f'))]
        auc_ver.append(curr_auc)

        roc_auc = [format(build_timer, '.2f'), format(output_timer, '.2f'), format(num_hash, '.1f'), format(num_bin, '.1f'), float(format(np.trapz(tpr_ver, fpr_ver), '.5f'))]
        roc_ver.append(roc_auc)

        plt.title("Precision x Recall_" + graph_title)
        plt.plot(recall_ver, precision_ver, linestyle='--', marker='o', color='b', label='Reuters')
        plt.legend(loc='best', prop={'size': 8})
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.savefig("Precision x Recall_" + fig_name)
        plt.show()
        plt.close('all')

        plt.title("ROC_" + graph_title)
        plt.plot(fpr_ver, tpr_ver, linestyle='--', marker='o', color='b', label='Reuters')
        plt.legend(loc='best', prop={'size': 8})
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.savefig("ROC_" + fig_name)
        plt.show()
        plt.close('all')

auc_ver = np.array(auc_ver)
roc_ver = np.array(roc_ver)

sorted_auc_ver = auc_ver[auc_ver[:,4].argsort()[::-1]]
sorted_roc_ver = roc_ver[roc_ver[:,4].argsort()[::-1]]

np.savetxt(("./Best_Configurations/" + "best_pxr.txt"), sorted_auc_ver, fmt='%s')
np.savetxt(("./Best_Configurations/" + "best_roc.txt"), sorted_roc_ver, fmt='%s')
