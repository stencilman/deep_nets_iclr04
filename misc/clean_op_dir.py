import sys, os, re

def main():
    if len(sys.argv) < 3:
        print 'Usage python clean_op_dir.py <op_dir> <epoch0>, [<epoch1>, ...]'
        print '    epochN is an epoch number to save.'
        return

    exceptions = ['weak_neg', 'losses.png', 'conf']  # Filenames to ignore

    op_dir = sys.argv[1]
    epochs = []
    for i in range(2, len(sys.argv)):
        epochs.append(sys.argv[i])
   
    if not os.path.exists(op_dir):
        print 'op_dir does not exist: ' + op_dir
        return
    
    print 'Cleaning op_dir: "' + sys.argv[1] + '", and keeping epochs: ' + \
        str(epochs)
    print 'Also keeping filenames containing: ' + str(exceptions)

    files_deleted = 0
    for dirname, dirnames, filenames in os.walk(op_dir):
        for filename in sorted(filenames):
            delete = True
            for epoch in epochs:
                if re.match('(.*)ep' + epoch + '.npy', filename):
                    delete = False
                if re.match('(.*)epoch-%04d' % int(epoch) + '(.*)', filename):
                    delete = False
                if re.match('(.*)epoch%04d' % int(epoch) + '(.*)', filename):
                    delete = False
                # This last case is a hack: since the convout and filters are
                # incorrectly labeled.
                if re.match('(.*)iter%04d' % int(epoch) + '(.*)', filename):
                    delete = False
            for exception in exceptions:
                if re.match('(.*)' + exception + '(.*)', filename):
                    delete = False


            if delete:
                print 'deleting ' + filename
                os.remove(os.path.join(dirname, filename)) 
                files_deleted += 1
    
    print str(files_deleted) + " files were deleted. Clean is finished..."

if __name__ == '__main__':
    main()
