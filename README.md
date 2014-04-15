deep_nets
=========

1. Download and process FLIC dataset<br/>
cd data/FLIC; python process.py<br/>

2. Train detector:<br/>
THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python train_detector.py conf.xml<br/>

3. Test detector:<br/>
Change the epoch_no value in conf.xml. While training, models after every epoch are written to disk. You can use any one of those.
THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python test_detector.py conf.xml
