deep_nets
=========

1. Download and process FLIC dataset<br/>
cd data/FLIC; python process.py<br/>

2. Train detector:<br/>
THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python train_detector.py conf.xml<br/>

3. Test detector:<br/>
THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python test_detector.py conf.xml
