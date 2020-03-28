import cv2 as cv
import os
import numpy as np

from object_detectors import gate_detector, path_marker_detector


"""
Classes for labelling data
"""


class ObjectHullLabeller:
    """
    Base class for labelling hulls associated to an object or not
    """

    
    def __init__(self, folder, detector, filter_fn):
        """
        Initializes an object hull labeller

        @param folder: The image folder that contains images of the object
        @param detector: The detector used to create the hulls
        @param filter_fn: Function to filter hulls clearly not the desired object, 
                returns True if it could be the object, otherwise returns False
        """
        self.folder = folder
        self.detector = detector
        self.filter_fn = filter_fn


    def create_labelled_dataset(self):
        """
        Opens the image folder for the object and for each image, allows a user to label the hulls from
        that image, then retuns a dataset of the hulls to their labels
                
        @returns: A dataset mapping the hulls of every image to their given labels
        """

        print("-------------------------------------------------------------------")
        print("              How to Use the Object Hull Label Tool                ")
        print("-------------------------------------------------------------------")
        print("- If a hull is NOT associated to the object: press the 1 button    ")
        print("- If a hull IS associated to the object: press the 2 button        ")
        print("\n- If any other key is pressed, the program EXITS                 ")
        print("-------------------------------------------------------------------")

        imgs = []
        labels = []
        directory = os.path.dirname(os.getcwd())
        
        # Get absolute path of all images in the images folder
        for dirpath,_,filenames in os.walk(os.path.join(directory, 'images', self.folder)):
            for f in filenames:
                imgs.append(os.path.abspath(os.path.join(dirpath, f)))

        # Get the hulls from the segmented image and run the display and label program for each image
        for img in imgs:
            src = cv.imread(img, 1)
            pre, seg, out = self.detector.detect(src)
            hulls = self.detector.convex_hulls(seg)
            labels += self.display_and_label_hulls(hulls, pre)
        return labels

        
    def display_and_label_hulls(self, hulls, src):
        """
        Displays each hull and allows a user to label the hull as being associated to the object or not and returns a
        data structure mapping hulls to the given labels

        @param hulls: The convex hulls 
        @param src: The source image from which the convex hulls have been created

        @returns: A list where each entry is a tuple mapping a hull to it's label
        """
        
        labels = []
        not_object, maybe_object = [], []  
        for hull in hulls:
            maybe_object.append(hull) if self.filter_fn(hull) else not_object.append(hull)      
        for hull in not_object:
            labels.append((hull, 0))
        for hull in maybe_object:
                cpy = src.copy()
                hull_img = cv.polylines(cpy, [hull], True, (0,0,255), 3)
                cv.imshow("Hull", hull_img)
                keycode = cv.waitKey(0)
                if keycode == 49:
                    labels.append((hull, 0))
                    print("Not the object")
                elif keycode == 50:
                    labels.append((hull, 1))
                    print("The object!")
                else:
                    raise Exception("Unexpected Key Pressed")
        cv.destroyAllWindows()
        return labels


class PoleHullLabeller(ObjectHullLabeller):
    """
    Class for labelling pole hulls 
    """


    def __init__(self, folder, detector):
        def filter_fn(hull):
            angle = 0
            MA = 1
            ma = 1
            try:
                _,(MA,ma),angle = cv.fitEllipse(hull)
            except:
                pass
            cosAngle = np.abs(np.cos(angle*np.pi/180))
            # Only human-classify hulls if it is reasonably a vertically oriented rectangle
            if  (cosAngle < 1.75) and (cosAngle > 0.85) and (MA/ma < 0.30):
                return True
            else: 
                return False
        super().__init__(folder, detector, filter_fn)


class PathMarkerHullLabeller(ObjectHullLabeller):
    """
    Class for labelling path markers
    """


    def __init__(self, folder, detector):
        def filter_fn(hull):
            return True
        super().__init__(folder, detector, filter_fn)


