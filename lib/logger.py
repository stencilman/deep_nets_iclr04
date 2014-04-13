# A VERY simple singleton to encapsulate file i/o for logging.  It's basic
# functionality is to redirect stdout/err so that all print statements also 
# stream to a file.

# This class is definitely not commercial quality (ie, issues arise if someone
# inherits from Logger, or if someone redefines stderr or stdout!)

import sys

# Singleton design pattern comes from:
# http://www.youtube.com/watch?v=0vJJlVBVTFg#t=14m38s

class Logger(object):
    def __new__(cls, *a, **k):
        if not hasattr(cls, '_inst'):
            cls._inst = super(Logger, cls).__new__(cls, *a, **k)
            cls._inst.is_open = False
        return cls._inst
        
    def open(self, filename):
        if self.is_open:
            raise Exception("Logger is already open!")
        self.filename = filename
        self.log_file = open(self.filename, 'w')
        self.open = True
        self.stderror_writer = StderrWriter(self)
        # Save the old stderr & stdout write functions and then override them
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self.stderror_writer
        
    def close(self):
        if (self.open):
            self.log_file.close()
            self.open = False
            sys.stdout = self.old_stdout
            sys.stderr = self.old_stderr
    
    def write(self, str):
        if not self.open:
            raise Exception("logger has been closed")
        # Write to real stdout
        self.old_stdout.write(str)
        self.old_stdout.flush()
        # Write to file
        self.log_file.write(str)
        self.log_file.flush()
        
    def flush(self):
        self.old_stdout.flush()
    
# This is a bit of a hack: we need to also override stderr, however this means
# we need 2 classes with different write methods but which both log to the
# same output file.  Instead, just define a simple class that is spawned by our
# singleton above to handle this.

class StderrWriter(object):
    def __init__(self, logger_inst):
        self.logger_inst = logger_inst
    
    def write(self, str):
        if not self.logger_inst.open:
            raise Exception("logger has been closed")
        # Write to real stderr
        self.logger_inst.old_stderr.write(str)
        self.logger_inst.old_stderr.flush()
        # Write to file
        self.logger_inst.log_file.write(str)
        self.logger_inst.log_file.flush()
        
    def flush(self):
        self.logger_inst.old_stderr.flush() 
    