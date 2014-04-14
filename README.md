deep_nets
=========
Download and process FLIC dataset
1. cd data/FLIC; python process.py

2. Train detector:
THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python train_detector.py conf.xml

3. Test detector:
THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python test_detector.py conf.xml
