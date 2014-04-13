import xml.etree.ElementTree as ET 
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#this one is a hack, fix it after CVPR 2014 deadline
def write_many(jnt_pos_2d, filepath):
    annotationlist = ET.Element('annotationlist')
    imgname = jnt_pos_2d.keys()[0]
    annotation = ET.SubElement(annotationlist, 'annotation')
    image = ET.SubElement(annotation, 'image')    
    ET.SubElement(image, 'name').text = imgname.split('/')[-1]
        
    points = jnt_pos_2d[imgname]
    for x, y, score in points:
        annorect = ET.SubElement(annotation, 'annorect')
        ET.SubElement(annorect, 'x1').text = '0'
        ET.SubElement(annorect, 'y1').text = '0'
        ET.SubElement(annorect, 'x2').text = '0'
        ET.SubElement(annorect, 'y2').text = '0'
        ET.SubElement(annorect, 'score').text = str(score)
        annopoints = ET.SubElement(annorect, 'annopoints')
        point = ET.SubElement(annopoints, 'point')
        ET.SubElement(point, 'x').text = str(x)
        ET.SubElement(point, 'y').text = str(y)
    
    output_file = open( filepath, 'w+' )
    output_file.write( ET.tostring( annotationlist ) )
    output_file.close()
                

def write(jnt_pos_2d, filepath):
    annotationlist = ET.Element('annotationlist')
    for idx, (imgname, jntpos) in enumerate(jnt_pos_2d.iteritems()):
        annotation = ET.SubElement(annotationlist, 'annotation')
        image = ET.SubElement(annotation, 'image')
        ET.SubElement(image, 'name').text = imgname.split('/')[-1]
        ET.SubElement(annotation, 'imgnum').text = str(idx)
        annorect = ET.SubElement(annotation, 'annorect')
        ET.SubElement(annorect, 'x1').text = '0'
        ET.SubElement(annorect, 'y1').text = '0'
        ET.SubElement(annorect, 'x2').text = '0'
        ET.SubElement(annorect, 'y2').text = '0'
        ET.SubElement(annorect, 'score').text = '-1'
        annopoints = ET.SubElement(annorect, 'annopoints')
        for i0, jnt in enumerate(jntpos):
            point = ET.SubElement(annopoints, 'point')
            ET.SubElement(point, 'x').text = str(jnt[0])
            ET.SubElement(point, 'y').text = str(jnt[1])
            ET.SubElement(point, 'id').text = str(i0)            
            ET.SubElement(point, 'is_visible').text = '-1'
        
    output_file = open( filepath, 'w+' )
    output_file.write( ET.tostring( annotationlist ) )
    output_file.close()
    
def read(filepath):
        xml = ET.parse(filepath)
        annotations = xml.findall('annotation')
        jnt_pos_2d = dict()
        for anno in annotations: 
            imgname = anno.find('image/name').text
            x1 = (anno.find('annorect/x1').text)
            y1 = (anno.find('annorect/y1').text)
            x2 = (anno.find('annorect/x2').text)
            y2 = (anno.find('annorect/y2').text)
            score = (anno.find('annorect/score').text)
            points = anno.findall('annorect/annopoints/point')
            jnt_pos_2d[imgname] = []
            for pt in points:
                # id = pt.find('id').text
                # print 'point id:' + id
                x = int(pt.find('x').text)
                y = int(pt.find('y').text)
                jnt_pos_2d[imgname].append((x, y))
        return jnt_pos_2d
    
def read_conv_anno(filepath, smoothI = None):
    jnt_pos_2d = dict()
    imgname, xs, ys, ss = read_conv_anno_full(filepath, smoothI)
    idx = ss.index(max(ss))
    jnt_pos_2d[imgname] = [(xs[idx], ys[idx])]
    #write(jnt_pos_2d, filepath.split('/')[0]+'/glob_prior/'+filepath.split('/')[-1])
    return jnt_pos_2d
    
def read_dpm_anno(filepath):
    xml = ET.parse(filepath)
    annotations = xml.findall('annotation')
    for anno in annotations: 
        imgname = anno.find('image/name').text
        x1 = float(anno.find('annorect/x1').text)
        y1 = float(anno.find('annorect/y1').text)
        x2 = float(anno.find('annorect/x2').text)
        y2 = float(anno.find('annorect/y2').text)
        score = float(anno.find('annorect/score').text)
        x = (x1 + x2) / 2.0
        y = (y1 + y2) / 2.0
        break
    return imgname, x, y, score

def read_conv_anno_full(filepath, smoothI = None):
    print 'Reading: '+filepath
    xml = ET.parse(filepath)
    annotation = xml.find('annotation')
    jnt_pos_2d = dict()
    allannorects = []
    xs = []
    ys = []
    ss = []
    imgname = annotation.find('image/name').text
    for annorect in annotation.findall('annorect'):
        score = float(annorect.find('score').text)
        points = annorect.find('annopoints/point')
        x = float(points.find('x').text)
        y = float(points.find('y').text)
        xs.append(x)
        ys.append(y)
        if smoothI is not None:
            if y == 240:
                y -= 1
            if x == 320:
                x -= 1
            score  *= smoothI[y,x]
        ss.append(score)
    return imgname, xs, ys, ss

def merge_files(dir, filepath, windowsize, scale = 1.0):
    windowsize /= scale
    alfiles = glob.glob(dir+'/*.al')
    alfiles = sorted(alfiles)
    alfiledata = []
    for alfile in alfiles:
        imgname, xs, ys, ss = read_conv_anno_full(alfile)
        alfiledata.append((imgname, xs, ys, ss))
        """
        im = plt.imread(dir + '/' + imgname)
        plt.imshow(im)
        plt.scatter(xs, ys, c=ss, cmap=cm.coolwarm)
        plt.show()
        """
    #now write the al file with this data
    annotationlist = ET.Element('annotationlist')
    for imgname, xs, ys, ss in alfiledata:
        annotation = ET.SubElement(annotationlist, 'annotation')
        image = ET.SubElement(annotation, 'image')
        ET.SubElement(image, 'name').text = imgname
        idx = imgname.split('.')[0].split('_')[-1]
        ET.SubElement(annotation, 'imgnum').text = idx
        for x, y, score in zip(xs, ys, ss):
            # x = x/float(scale) 
            # y = y/float(scale) 
            annorect = ET.SubElement(annotation, 'annorect')
            ET.SubElement(annorect, 'x1').text = str(x-windowsize/2)
            ET.SubElement(annorect, 'y1').text = str(y-windowsize/2)
            ET.SubElement(annorect, 'x2').text = str(x+windowsize/2)
            ET.SubElement(annorect, 'y2').text = str(y+windowsize/2)
            ET.SubElement(annorect, 'score').text = str(score)
            #add a pt to center
            annopoints = ET.SubElement(annorect, 'annopoints')
            point = ET.SubElement(annopoints, 'point')
            ET.SubElement(point, 'x').text = str(x)
            ET.SubElement(point, 'y').text = str(y)
            ET.SubElement(point, 'id').text = str(0)            


    output_file = open( filepath, 'w+' )
    output_file.write( ET.tostring( annotationlist ) )
    output_file.close()

def merge_dpm_files(dir, filepath):
    alfiles = glob.glob(dir+'/*.al')
    alfiles = sorted(alfiles)
    alfiledata = []
    for idx, alfile in enumerate(alfiles):
        filename = 'imgidx-{0:d}-{1:d}.al'.format(idx, idx)
        imgname, xs, ys, ss = read_dpm_anno(alfile)
        alfiledata.append((imgname, xs, ys, ss))
        """
        im = plt.imread(dir + '/' + imgname)
        plt.imshow(im)
        plt.scatter(xs, ys, c=ss, cmap=cm.coolwarm)
        plt.show()
        """
    windowsize = 60
    #now write the al file with this data
    annotationlist = ET.Element('annotationlist')
    for imgname, x, y, s in alfiledata:
        imgname = imgname.split('/')[-1]
        annotation = ET.SubElement(annotationlist, 'annotation')
        image = ET.SubElement(annotation, 'image')
        ET.SubElement(image, 'name').text = imgname
        idx = imgname.split('.')[0].split('_')[-1]
        ET.SubElement(annotation, 'imgnum').text = idx
        annorect = ET.SubElement(annotation, 'annorect')
        ET.SubElement(annorect, 'x1').text = str(x-windowsize/2)
        ET.SubElement(annorect, 'y1').text = str(y-windowsize/2)
        ET.SubElement(annorect, 'x2').text = str(x+windowsize/2)
        ET.SubElement(annorect, 'y2').text = str(y+windowsize/2)
        ET.SubElement(annorect, 'score').text = str(s)
            #add a pt to center
        annopoints = ET.SubElement(annorect, 'annopoints')
        point = ET.SubElement(annopoints, 'point')
        ET.SubElement(point, 'x').text = str(x)
        ET.SubElement(point, 'y').text = str(y)
        ET.SubElement(point, 'id').text = str(0)            


    output_file = open( filepath, 'w+' )
    output_file.write( ET.tostring( annotationlist ) )
    output_file.close()   
           
        
#merge_dpm_files('/Users/ajain/Projects/pose_estimate/final_experiments/vsDPM/alfiles/', '/Users/ajain/Projects/pose_estimate/final_experiments/vsDPM/flic-351-dpm.al')
