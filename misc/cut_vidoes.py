import os
import subprocess as sp
import re
import sys

from collections import defaultdict
behave_videos = defaultdict(list)

behave_videos['1-11200.avi'].append((40, 8))  # (start, duration) in seconds
behave_videos['1-11200.avi'].append((50, 8))
 
behave_videos['11500-17450.avi'].append((7, 7))
 
behave_videos['24300-35200.avi'].append((234, 8))
behave_videos['24300-35200.avi'].append((284, 8))
behave_videos['24300-35200.avi'].append((334, 8))
behave_videos['24300-35200.avi'].append((370, 8)) 

behave_videos['35450-47160.avi'].append((0, 8)) 
behave_videos['35450-47160.avi'].append((8, 8))
behave_videos['35450-47160.avi'].append((16, 7))
behave_videos['35450-47160.avi'].append((255, 8)) 
behave_videos['35450-47160.avi'].append((289, 6))
behave_videos['35450-47160.avi'].append((304, 6))
behave_videos['35450-47160.avi'].append((311, 6)) 
behave_videos['35450-47160.avi'].append((383, 8))
behave_videos['35450-47160.avi'].append((391, 8))

behave_videos['47300-58400.avi'].append((1, 10)) 
behave_videos['47300-58400.avi'].append((123, 6))
behave_videos['47300-58400.avi'].append((129, 6))

behave_videos['59800-66750.avi'].append((10, 10))  
behave_videos['59800-66750.avi'].append((20, 10))  
behave_videos['59800-66750.avi'].append((30, 10))  
behave_videos['59800-66750.avi'].append((40, 10))  
behave_videos['59800-66750.avi'].append((58, 10))  
behave_videos['59800-66750.avi'].append((113, 10))
behave_videos['59800-66750.avi'].append((123, 10))  
behave_videos['59800-66750.avi'].append((133, 10))
behave_videos['59800-66750.avi'].append((162, 10))  
behave_videos['59800-66750.avi'].append((260, 10))  
  

if __name__ == '__main__':
    for vidname in behave_videos.keys():
        no_ext = os.path.splitext(vidname)[0]
        for idx, clip in enumerate(behave_videos[vidname]):
            s, d = clip
            print "ffmpeg -i {0:s} -ss {1:d}s -t {2:d}s -vcodec libx264 -vpre slow -b 2000k processed_videos/{3:s}_{1:d}_{2:d}.mp4".format(vidname, s, d, no_ext)
            # p3 = sp.check_output("", shell=True) 

    
