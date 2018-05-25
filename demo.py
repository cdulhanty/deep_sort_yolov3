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
import collections
import math

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')

def main(yolo):

    args = parse_args()

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
    vid_fps = int(video_capture.get(5))
    n_frames = int(video_capture.get(7))

    #setup output video
    outname = 'output_{}'.format(args.video)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoWriter = cv2.VideoWriter(outname, fourcc, vid_fps, (width, height))

    print(args.video +' @ ' + str(vid_fps) + ' fps')

    # setup ouput csv
    f = open('output_{}.csv'.format(args.video),'w')
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['frame_id', 'track_id'])
    f.flush()

    printIndex = 50

    frame_index = 0
    fps = 0.0

    n_people_queue = collections.deque(maxlen=24) # TODO: confirm value here
    n_people_sum = 0.0
    n_people_avg = 0.0

    wait_times_sum = 0.0
    wait_times_avg = 0.0

    while True:

        ret, frame = video_capture.read()  # frame shape 640*480*3

        if ret != True:
            break;

        frame_index = frame_index + 1

        t1 = time.time()

        image = Image.fromarray(frame)
        boxs = yolo.detect_image(image)

        n_people_queue.append(len(boxs))

        if frame_index % 24 == 0:

            for value in n_people_queue:
                n_people_sum += value

            n_people_avg = math.ceil(n_people_sum/len(n_people_queue))
            n_people_sum = 0

        cv2.putText(frame, "# People Detected: " + str(n_people_avg), (10, 30), 0, 1, (0, 0, 0), 2)

        # define hit box (front of the line)
        x_1 = 580
        x_2 = 640
        y_1 = 140
        y_2 = 380

        # write white box to frame displaying hit box
        cv2.rectangle(frame, (x_1, y_1), (x_2, y_2), (0, 0, 0), 2)
        cv2.putText(frame, "FRONT", (x_1-60, y_1), 0, 1, (0, 0, 0), 2) #TODO -fix font size

        # only keep box if box in hit box box
        out_indicies = []
        for index, value in enumerate(boxs):
            if value[0] > x_1 and value[0] < x_2 and value[1] > y_1 and value[1] < y_2:
                out_indicies.append(index)

        boxs = [boxs[i] for i in out_indicies]

        features = encoder(frame, boxs)

        # score to 1.0 here
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections, frame_index) # pass the frame_index

        # Write wait time to frame
        for value in tracker.wait_times:
            wait_times_sum += value

        if len(tracker.wait_times) > 0:
            wait_times_avg = math.ceil(wait_times_sum/len(tracker.wait_times))
        wait_times_sum = 0

        cv2.putText(frame, "Avg. Move-Up Time: %.2f s" %(wait_times_avg/12.), (10, 70), 0, 1, (0, 0, 0), 2)
        cv2.putText(frame, "Current Wait Time: %.2f s" %(wait_times_avg/12.*n_people_avg), (10, 110), 0, 1, (0, 0, 0), 2)

        for track in tracker.tracks:

            if track.is_confirmed() and track.time_since_update >1 :
                continue

            bbox = track.to_tlbr()

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(track.c1,track.c2,track.c3), 2)
            #cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)
            writer.writerow([frame_index, track.track_id])

        #for det in detections:
            #bbox = det.to_tlbr()
            #cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)

        if frame_index + 100 > n_frames:
            cv2.putText(frame, "Total Number of People Processed: " + tracker.num_valid_tracks, (240, 220), 0, 1, (0, 0, 0), 2)

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

    return parser.parse_args()

if __name__ == '__main__':
    main(YOLO())
