#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import csv
import argparse
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

def main(yolo):
    
    args = parse_args()
    
    if args.nFrames < 1:
        sys.exit(0)

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
    video_capture = cv2.VideoCapture(args.video)
    width = int(video_capture.get(3))
    height = int(video_capture.get(4))
    vid_fps = int(video_capture.get(5)/float(args.nFrames))

    #setup output video
    outname = 'output_{}'.format(args.video)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoWriter = cv2.VideoWriter(outname, fourcc, vid_fps, (width, height))

    print(args.video +' @ ' + str(vid_fps) + ' fps')

    # setup ouput csv 
    f = open('output_{}.csv'.format(args.video),'w')
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['frame_id', 'track_id' , 'x', 'y', 'w', 'h'])
    f.flush()

    printIndex = 50
    while args.nFrames != 1 and printIndex % args.nFrames == 0 :
        printIndex += 1

    frame_index = 0
    fps = 0.0

    while True:

        ret, frame = video_capture.read()  # frame shape 640*480*3
        
        if ret != True:
            break;

        frame_index = frame_index + 1
        
        if args.nFrames > 1 and frame_index % args.nFrames == 0:
            continue

        t1 = time.time()    

        image = Image.fromarray(frame)
        boxs = yolo.detect_image(image)
        
        # only keep box if box in hit box area
        out_indicies = []
        
        x_1 = 560
        x_2 = 640
        y_1 = 140
        y_2 = 380
        
        for index, value in enumerate(boxs):
            if value[0] > x_1 and value[0] < x_2 and value[1] > y_1 and value[1] < y_2:
                out_indicies.append(index)
        
        boxs = [boxs[i] for i in out_indicies]
                
        # write white box to frame displaying hit box
        cv2.rectangle(frame, (x_1, y_1), (x_2, y_2), (0, 0, 0), 2)
        cv2.putText(frame, "Front", (x_1, y_1), 0, 5e-3 * 200, (0, 0, 0), 2)

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
            
            #

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(track.c1,track.c2,track.c3), 2)
            #cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

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
            
def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Arg. parser for yolov3 x deep_sort demo")
    
    parser.add_argument(
        "--video",
        help="Path to video")
    
    parser.add_argument(
        "--nFrames", help="Fraction of frames to run on",
        default=1,
        required=False)
    
    return parser.parse_args()

if __name__ == '__main__':
    main(YOLO())
