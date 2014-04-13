import os, sys

if len(sys.argv) < 4:
    print 'Usage: python misc/merge_stitches_files.py <model_dir> <epoch> <format>'
    print '<format> is 0 or 1 and is the dir format (FLIC or stitched)'

model_dir = sys.argv[1]
ep = sys.argv[2]
dir_format = int(sys.argv[3])
if not os.path.exists(model_dir):
    print 'model_dir does not exist: ' + model_dir
    sys.exit(1)

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../lib'))
if not path in sys.path:
    sys.path.append(path)

import alfileformat as al
if dir_format == 0:
    base = model_dir + '/FLIC-sapp-all_sc_'
elif dir_format == 1:
    base = model_dir + '/stitched_sc_'
else:
    raise Exception("<format> must be 0 or 1")

for scale in ['0.44', '0.54', '0.64', '0.74', '0.84', '0.94']:
    if dir_format == 0:
        base_dir = base + scale + '_ep_' + ep + '/'
    else:
        base_dir = base + scale + '/'
    print base_dir
    if dir_format == 0:
        al.merge_files(base_dir, base_dir + '/FLIC-sapp-all_'+scale+'_ep_' + ep + '.al', 64, 1.0)
    else:
        al.merge_files(base_dir, base_dir + '/FLIC-sapp-all_'+scale+'_ep_' + ep + '.al', 64, float(scale))
