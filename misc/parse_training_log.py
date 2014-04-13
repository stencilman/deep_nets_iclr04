# A simple script to parse through the training log and extract per-epoch
# test and training set errors.  This is very crude.

import sys
import re

# This function is a little hacky, but I think I got most of the cases.
# It just returns the first number in a string (integer or decimal)
def get_first_num(str):
    for idx, char in enumerate(str):
        if char.isdigit() or (idx < len(str) - 1 and char == '-' and str[idx+1].isdigit()):
            right_decimal = False
            for idy in range(idx+1,len(str)):
                if (not str[idy].isdigit()) and str[idy] != '.':
                    return str[idx:idy]
                elif str[idy] == '.' and not right_decimal:
                    right_decimal = True
                elif str[idy] == '.' and right_decimal:
                    return str[idx:idy]
            return str[idx:]
    return None
            

def main():
    if len(sys.argv) < 2:
        print 'Usage: python parse_training_log.py <log_filename>'
        return
    
    print 'Parsing log file: ' + sys.argv[1]
    
    # Get all the lines in the file (we need to be able to peak lines later)
    lines = [line.strip() for line in open(sys.argv[1])]
    
    # We're looking for blocks of the log file that look like this:
    # epoch 10, minibatch 62/62
    #      mean test error 4.487690
    #      mean train error 4.615923
    epochs = []
    test_errs = []
    train_errs = []
    test_errs_do_on = []
    pos_test = []
    pos_train = []
    neg_test = []
    neg_train = []
    for idx, line in enumerate(lines):
        if re.match('training @ iter = (.*)', line):
            # Now we have to find the training and test set errors
            for idy in range (idx, idx+13):
                if idy < len(lines):
                    if re.match("(.*)epoch (.*), minibatch (.*)", lines[idy]):
                        epochs.append(int(get_first_num(lines[idy])))
                    if re.match("(.*)mean test error dropout_on (.*)", lines[idy]):
                        test_errs_do_on.append(float(get_first_num(lines[idy])))
                    elif re.match("(.*)mean test error (.*)", lines[idy]):
                        test_errs.append(float(get_first_num(lines[idy])))
                    elif re.match("(.*)mean train error (.*)", lines[idy]):
                        train_errs.append(float(get_first_num(lines[idy])))
                    elif re.match("train positives (.*)", lines[idy]):
                        pos_train.append(float(get_first_num(lines[idy])))
                    elif re.match("train negatives (.*)", lines[idy]):
                        neg_train.append(float(get_first_num(lines[idy])))
                    elif re.match("test positives (.*)", lines[idy]):
                        pos_test.append(float(get_first_num(lines[idy])))
                    elif re.match("test negatives (.*)", lines[idy]):
                        neg_test.append(float(get_first_num(lines[idy])))
                    
    if len(epochs) != len(test_errs) or len(epochs) != len(train_errs) or \
        len(epochs) != len(pos_test) or \
        len(epochs) != len(pos_train) or \
        len(epochs) != len(neg_train) or \
        len(epochs) != len(neg_test):
        raise Exception("Something went wrong parsing log file!")
            
    # Now print out the epochs, test and training errors nicely
    if len(test_errs_do_on) > 0:
        if len(test_errs_do_on) != len(epochs):
            raise Exception("Something went wrong parsing log file!")
        print "epoch test_err train_err test_err_do_on"
        for epoch, test_err, train_err, test_err_do in zip(epochs, test_errs, train_errs, test_errs_do_on):
            print "%06d %.6e %.6e %.6e" % (epoch, test_err, train_err, test_err_do)
    else:
        print "epoch test_err   train_err  pos_test   neg_test   pos_train  neg_train"
        for epoch, test_err, train_err, ptest, ntest, ptrain, ntrain in zip(epochs, test_errs, train_errs, pos_test, neg_test, pos_train, neg_train):
            print "%05d %.4e %.4e %.4e %.4e %.4e %.4e" % (epoch, test_err, train_err, ptest, ntest, ptrain, ntrain)
    
    
if __name__ == '__main__':    
    main()
