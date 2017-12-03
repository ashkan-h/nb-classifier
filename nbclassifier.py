import sys
import re
from collections import Counter
import operator
import math
from math import fsum

def extract_words(text):
    punctuations = '''!()-[]{};:"\,<>./?@#$%^&*_~'''
    new_string = text.lower()
    new_string = "".join(c for c in new_string if c not in punctuations)
    new_string = new_string.split(" ")
    return [x for x in new_string if x]


class NbClassifier(object):

    def __init__(self, training_filename, stopword_file = None):
        self.attribute_types = set()
        self.label_prior = {}    
        self.word_given_label = {}   


        self.collect_attribute_types(training_filename, 2)
        self.train(training_filename)          

    def collect_attribute_types(self, training_filename, k):
        print ("Using the stopwords.txt for better features.")

        print ("Current Value of K is : " + str(k))
        stop_set = set()

        with open("stopwords_mini.txt", "r") as f:
            for line in f:
                line = line.strip()
                stop_set.add(line)

        words = ' '.join(open(training_filename).read().split())     
        freqs = Counter(extract_words(words))

        del freqs['spam']
        del freqs['ham']

        for key, val in freqs.items():
            if val >= k and key not in stop_set:
                self.attribute_types.add(key)


    def train(self, training_filename):
        
        c = 0.05
        spam_list, ham_list = [], []
        spam_dict, ham_dict = {}, {}
        condition_spam, condition_ham = [], []
        total_num_label_spam, total_num_label_ham = 0.0, 0.0
        n_ham , n_spam, total_sms = 0, 0, 0
        vocab_size = len(self.attribute_types)
        
        print ("Current Value of C is : " + str(c))
        with open(training_filename, "r") as f:
            for line in f:
                if line.startswith('ham'):
                    n_ham += 1
                    ham_list += re.split(r'[\n\r\t]+', line)
                elif line.startswith('spam'):
                    n_spam += 1
                    spam_list += re.split(r'[\n\r\t]+', line)

        ham_dict = Counter(extract_words(' '.join(ham_list)))
        spam_dict = Counter(extract_words(' '.join(spam_list))) 
        total_sms = n_spam + n_ham

        self.label_prior['spam'] = n_spam / total_sms
        self.label_prior['ham'] = n_ham / total_sms

        print ("[Sanity Check] -- Sum of priors : " + str(fsum(self.label_prior.values())))


        del spam_dict['spam']
        del ham_dict['ham']

        
        for k,v in ham_dict.items():
            if k in self.attribute_types:
                total_num_label_ham += v

        for k,v in spam_dict.items():
            if k in self.attribute_types:
                total_num_label_spam += v

        for k,v in ham_dict.items():
            if k in self.attribute_types:
                if k not in spam_dict:
                    self.word_given_label[(k,'spam')] = (0 + c) / (total_num_label_spam + (c * vocab_size))
                self.word_given_label[(k,'ham')] = (v + c) / (total_num_label_ham + (c * vocab_size))

        for k,v in spam_dict.items():
            if k in self.attribute_types:
                if k not in ham_dict:
                    self.word_given_label[(k,'ham')] = (0 + c) / (total_num_label_ham + (c * vocab_size))
                self.word_given_label[(k,'spam')] = (v + c) / (total_num_label_spam + (c * vocab_size))
    

        condition_spam = [v for k,v in self.word_given_label.items() if k[1] == 'spam']
        condition_ham = [v for k,v in self.word_given_label.items() if k[1] == 'ham']
        

        print("[Sanity Check] -- Sum of conditionals for label spam : "  + str(fsum(condition_spam)))
        print("[Sanity Check] -- Sum of conditionals for label ham : " + str(fsum(condition_ham)))

     
    def predict(self, text):
        
        all_ham = math.log(self.label_prior['ham'])
        all_spam = math.log(self.label_prior['spam'])

        current_text = extract_words(text)

        for word in current_text:
            if word in self.attribute_types:
                if (word, 'spam') in self.word_given_label:
                    all_spam += math.log(self.word_given_label[(word, 'spam')])
                if (word, 'ham') in self.word_given_label: 
                    all_ham += math.log(self.word_given_label[(word, 'ham')])

        return {'ham': all_ham, 'spam': all_spam}


    def evaluate(self, test_filename):
        tp, fp, fn, tn = 0.0, 0.0, 0.0, 0.0
        with open(test_filename, "r") as f:
            for line in f:
                line = re.split(r'[\n\r\t]+', line)
                real_category = line[0]
                text_to_predict = line[1]

                prediction = max(self.predict(text_to_predict).items(), key=operator.itemgetter(1))[0]

                if prediction == 'spam' and real_category == 'spam':
                    tp += 1
                elif prediction == 'spam' and real_category == 'ham':
                    fp += 1
                elif prediction == 'ham' and real_category == 'ham':
                    tn += 1
                elif prediction == 'ham' and real_category == 'spam':
                    fn += 1

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fscore =  (2 * precision * recall)/(precision + recall)
        accuracy = (tp + tn)/(tp + tn + fp + fn)

        return precision, recall, fscore, accuracy


def print_result(result):
    print("Precision:{} Recall:{} F-Score:{} Accuracy:{}".format(*result))


if __name__ == "__main__":
    
    classifier = NbClassifier(sys.argv[1])
    result = classifier.evaluate(sys.argv[2])
    print_result(result)
