import argparse
import cv2 as cv
import random
import pickle
import os
import numpy as np
from fractions import Fraction

import gate_detector
import data_labelling


"""
Main Underwater Object Detection Program
"""


def gate_detector_video_test(video_name, im_resize=1.0, record=False, debug=False):
    """
    Test GateDetector on a video, records the output

    @param im_resize: The resized image the detector is run on
    @param record: True if you want to record and save the output of the detector
    @param debug: True if you want debug information displayed on each frame

    """
    detector = gate_detector.GateDetector(im_resize=im_resize, debug=debug)
    cap = cv.VideoCapture('../videos/' + video_name )
    frame_width = int(cap.get(3)*im_resize)#*2
    frame_height = int(cap.get(4)*im_resize)#*2
    if record:
        r = random.randint(0,1000)
        out = cv.VideoWriter('../videos/output' + str(r) + '.avi',cv.VideoWriter_fourcc(*'XVID') , 20.0, (frame_width,frame_height))
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    while(cap.isOpened()):
        ret, src = cap.read()
        if ret == True:
            pre = detector.preprocess(src)
            seg = detector.morphological(detector.segment(pre)) 
            # seg, grad, mask = detector.segment(pre)
            hulls = detector.create_convex_hulls(seg)
            gate = detector.bound_gate_using_poles(hulls, src)
            pose = detector.estimate_gate_pose(gate)

            # 4 Frame output
            # seg = cv.cvtColor(seg, cv.COLOR_GRAY2BGR)
            # grad = cv.cvtColor(grad, cv.COLOR_GRAY2BGR)
            # mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
            # top_row = np.hstack((grad, mask))
            # bot_row = np.hstack((basis, pose))
            # pose = np.vstack((top_row, bot_row))

            cv.imshow('Gate',pose)
            if debug:
                cv.imshow('Processed', pre)
                cv.imshow('Segmented', seg)
            if record :
                out.write(pose)
            if cv.waitKey(25) & 0xFF == ord('q'):
                break
        else: 
            break
    cap.release()
    if record:
        out.release()
    cv.destroyAllWindows()


def gate_detector_image_test(image_name, im_resize=1.0, debug=False):
    """
    Test GateDetector on an image, shows the images at various stages of detection

    @param im_resize: The resized image the detector is run on
    @param debug: True if you want debug information displayed on the image

    """
    src = cv.imread('../images/gate/' + image_name,1)

    detector = gate_detector.GateDetector(im_resize=im_resize, debug=debug)

    pre = detector.preprocess(src)
    seg = detector.morphological(detector.segment(pre))
    hulls = detector.create_convex_hulls(seg)
    gate = detector.bound_gate_using_poles(hulls, src)
    pose = detector.estimate_gate_pose(gate)

    cv.imshow('Gate', pose)
    if debug:
        cv.imshow('Processed', pre)
        cv.imshow('Segmented', seg)
    cv.waitKey(0)


def gate_detector_label_poles():
    """
    Run Pole label program
    """
    labeller = data_labelling.PoleHullLabeller()
    labels = labeller.create_labelled_dataset()

    # Dump labels data to pickle
    r = random.randint(0,1000)
    d = os.path.dirname(os.getcwd())
    with open(os.path.join(d, 'pickle/pole_data' + str(r) + '.pkl'), 'wb') as file:
        pickle.dump(labels, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Underwater Object Detection")
    parser.add_argument('-g','--gate', required = True)
    parser.add_argument('-r', '--resize', default=1.0)
    parser.add_argument('-d', '--debug', action='store_true', default=False)
    parser.add_argument('-R', '--record', action='store_true', default=False)
    io_args = parser.parse_args()

    im_resize = float(Fraction(io_args.resize))
    gate = io_args.gate
    debug = io_args.debug
    record = io_args.record

    if gate == "im":
        # Run detector on set of 20 images of various distances to gate
        for i in range(20):
            im_name = str(i) + '.jpg'
            gate_detector_image_test(im_name, im_resize=im_resize, debug=debug)
    elif gate == "vid":
        # Run detector on video
        gate_detector_video_test('gate.mp4',im_resize=im_resize, record=record, debug=debug)
    elif gate == "label":
        # Run data label program
        gate_detector_label_poles()






