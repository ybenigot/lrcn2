#!/usr/bin/env python
# adaped from coco_caption, coco_to_hdf5_data.py

# this creates a text file 'coco-caption.txt' with structure :
# - fields separated by blanks
# - field 1 is absolute path to an image
# - all subsequents fields are words of a sentence describing the image

# we need coco PythonAPI installed
# we need PYTHONPATH, example value :
#echo $PYTHONPATH
#/Users/yves/Documents/coco/coco-master/PythonAPI/pycocotools/:/Users/yves/Documents/coco/coco-master/PythonAPI

import re
import os
from coco import COCO

pc=True
if pc:
	IMAGE_ROOT='/media/yves/sandisk3/datasets/coco/images/train2014'
	COCO_PATH = 'media/yves/sandisk3/datasets/coco'
else:
	IMAGE_ROOT='/Users/yves/Documents/caffe2/data/coco/images/train2014'
	COCO_PATH = '/Users/yves/Documents//caffe2/data/coco/coco'

COCO_ANNO_PATH = '%s/annotations/captions_%%s2014.json' % COCO_PATH

SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')

def split_sentence(sentence):
  # break sentence into a list of words and punctuation
  sentence = [s.lower() for s in SENTENCE_SPLIT_REGEX.split(sentence.strip()) if len(s.strip()) > 0]
  # remove the '.' from the end of the sentence
  if sentence[-1] != '.':
    # print "Warning: sentence doesn't end with '.'; ends with: %s" % sentence[-1]
    return sentence
  return sentence[:-1]

def read(coco):

	split_ids = coco.imgs.keys()
	image_path_to_id = {}
	known_images = {}
	num_total = 0
	num_missing = 0
	num_captions = 0
	outfile=open('coco-caption.txt','w')
	for image_id in split_ids:
		image_info = coco.imgs[image_id]
		image_path = '%s/%s' % (IMAGE_ROOT, image_info['file_name'])
		image_path_to_id[image_path] = image_id
		if os.path.isfile(image_path):
			assert image_id not in known_images  # no duplicates allowed
			known_images[image_id] = {}
			known_images[image_id]['path'] = image_path
			print('-----> found '+image_path)
			known_images[image_id]['sentences'] = [split_sentence(anno['caption'])
					for anno in coco.imgToAnns[image_id]]
			num_captions += len(known_images[image_id]['sentences'])
			first=True
			for x in known_images[image_id]['sentences']:
				line = ' '.join([y.encode('ascii') for y in x ])
				print ("   caption : " + line)
				if first :
					outfile.write(image_path+' '+line+'\n')
					first = False # write only first sentence along with image file name

		else:
			num_missing += 1
			print 'Warning (#%d): image not found: %s' % (num_missing, image_path)
		num_total += 1
	print('total images scanned : ',num_total)
	outfile.close()

if __name__ == "__main__":
	coco_split_name='train'
	coco = COCO(COCO_ANNO_PATH % coco_split_name)
	read(coco)
