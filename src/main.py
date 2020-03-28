import argparse
import cv2 as cv
import random
import pickle
import os
import numpy as np
from fractions import Fraction

from object_detectors import gate_detector, path_marker_detector
from machine_learning import data_labelling, pole_classifier


"""
Main Underwater Object Detection Program
"""


def detector_video_test(video_name, detector, record=False):
    """
    Test GateDetector on a video, records the output if record is true

    @param detector: Detector object which must implement base object detector class
    @param record: True if you want to record and save the output of the detector
    """
    print("-------------------------------------------------------------------")
    print("            Video Test: Press any key to stop video                ")
    print("-------------------------------------------------------------------")
    directory = os.path.dirname(os.getcwd())
    infile = os.path.join(directory, 'videos/' + video_name )
    cap = cv.VideoCapture(infile)
    im_resize = detector.im_resize
    frame_width = int(cap.get(3)*im_resize)
    frame_height = int(cap.get(4)*im_resize)
    if record:
        r = random.randint(0,1000)
        outfile = os.path.join(directory, 'videos/output' + str(r) + '.avi' )
        out = cv.VideoWriter(outfile ,cv.VideoWriter_fourcc(*'XVID') , 20.0, (frame_width,frame_height))
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    while(cap.isOpened()):
        ret, src = cap.read()
        if ret == True:
            im_1,im_2,im_out = detector.detect(src)
            cv.imshow('Output',im_out)
            if debug:
                cv.imshow('Image 1', im_1)
                cv.imshow('Image 2', im_2)
            if record :
                out.write(im_out)
            if cv.waitKey(25) & 0xFF != 255:
                break
        else: 
            break
    cap.release()
    if record:
        out.release()
    cv.destroyAllWindows()


def detector_image_test(folder, detector):
    """
    Tests detector on a set of 20 image

    @param detector: Detector object which must implement base object detector class

    """
    print("-------------------------------------------------------------------")
    print("      Image Test: Press any key to advance to the next image       ")
    print("-------------------------------------------------------------------")
    directory = os.path.dirname(os.getcwd())
    for i in range(20):
        im_name = str(i) + '.jpg'
        im_file = os.path.join(directory, 'images', folder, im_name)
        src = cv.imread(im_file,1)
        im1,im2,out = detector.detect(src)
        cv.imshow('Output',out)
        if debug:
            cv.imshow('Image 1', im1)
            cv.imshow('Image 2', im2)
        cv.waitKey(0)
        # Reset gate contour 
        detector.gate_cntr = None


def label_data(datatype):
    """
    Run data label program given by datatype

    @param datatype: The type of data that is being labelled
    """
    labels = []
    if datatype is "pole":
        labeller = data_labelling.PoleHullLabeller()
        labels = labeller.create_labelled_dataset()
    elif datatype is "pathmarker":
        labeller = data_labelling.PathMarkerHullLabeller()
        labels = labeller.create_labelled_dataset()
    r = random.randint(0,1000)
    directory = os.path.dirname(os.getcwd())
    filename = datatype + "_data" + str(r) + '.pkl'
    with open(os.path.join(directory, 'pickle', filename), 'wb') as file:
        pickle.dump(labels, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Underwater Object Detection")

    parser.add_argument('-g','--gate')
    parser.add_argument('-pm', '--pathmarker')
    parser.add_argument('-r', '--resize', default=1.0)
    parser.add_argument('-d', '--debug', action='store_true', default=False)
    parser.add_argument('-R', '--record', action='store_true', default=False)

    parser.add_argument('-c', '--classify')
    parser.add_argument('-df', '--datafile', default='pole_data1.pkl')
    io_args = parser.parse_args()

    # Programs
    gate = io_args.gate
    path_marker = io_args.pathmarker
    classify = io_args.classify

    # Detector args
    im_resize = float(Fraction(io_args.resize))
    debug = io_args.debug
    record = io_args.record

    # Classify args
    datafile = io_args.datafile
    
    programs = [gate,classify,path_marker]
    if (sum(p is not None for p in programs) != 1):
        print("Please only use exactly one of [--gate, --pathmarker, --classify]")
        exit()

    if gate is not None:
        detector = gate_detector.GateDetector(im_resize=im_resize, debug=debug)
        if gate == "im": 
            detector_image_test('gate', detector)
        elif gate == "vid":
            detector_video_test('gate.mp4', detector, record=record)
        elif gate == "label":
            label_data("pole")

    if path_marker is not None:
        detector = path_marker_detector.PathMarkerDetector(im_resize=im_resize, debug=debug)
        if path_marker == "im":
            detector_image_test('pathmarker', detector)
        elif path_marker == "vid":
            detector_video_test('pathmarker.mp4', detector, record=record)
        elif path_marker == "label":
            label_data("pathmarker")

    if classify is not None:
        if classify == "pole":
            pole_classifier.PoleClassifier(datafile=datafile).run()








