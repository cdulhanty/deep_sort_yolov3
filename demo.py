#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import csv
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')

def main(yolo, files, nthFrames):

    for file in files:
      
        for nthFrame in nthFrames:
        
            if nthFrame < 1:
                continue

            # Definition of the parameters
            max_cosine_distance = 0.3
            nn_budget = None
            nms_max_overlap = 1.0

            # deep_sort 
            model_filename = 'model_data/mars-small128.pb'
            encoder = gdet.create_box_encoder(model_filename,batch_size=1)
            metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
            tracker = Tracker(metric)

            # open video file
            video_capture = cv2.VideoCapture(file)
            width = int(video_capture.get(3))
            height = int(video_capture.get(4))
            vid_fps = int(video_capture.get(5))

            #setup output video
            outname = 'output_{}'.format(file)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            videoWriter = cv2.VideoWriter(outname, fourcc, vid_fps, (width, height))
            
            print('file @' + str(vid_fps) + ' fps')

            # setup ouput csv 
            f = open('output_{}.csv'.format(file),'w')
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['frame_id', 'track_id' , 'x', 'y', 'w', 'h'])
            f.flush()

            printIndex = 50
            while nthFrame != 1 and printIndex / nthFrame == 0 :
                printIndex += 1
            
            frame_index = 0
            fps = 0.0

            while True:

                ret, frame = video_capture.read()  # frame shape 640*480*3

                if ret != True:
                    break;

                frame_index = frame_index + 1

                if nthFrame > 1 and frame_index % nthFrame == 0:
                    continue

                t1 = time.time()    

                image = Image.fromarray(frame)
                boxs = yolo.detect_image(image)

                features = encoder(frame,boxs)

                # score to 1.0 here
                detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

                # Run non-maxima suppression.
                boxes = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
                detections = [detections[i] for i in indices]

                # Call the tracker
                tracker.predict()
                tracker.update(detections)

                for track in tracker.tracks:

                    if track.is_confirmed() and track.time_since_update >1 :
                        continue

                    bbox = track.to_tlbr()

                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                    cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

                for det in detections:
                    bbox = det.to_tlbr()
                    cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)

                # write processed frame to video
                videoWriter.write(frame)
                                
                if frame_index % printIndex == 0:
                    print(frame_index)
                    
            videoWriter.release()
            video_capture.release()
            f.close()

if __name__ == '__main__':
    main(YOLO(), sys.argv[1], sys.argv[2])
