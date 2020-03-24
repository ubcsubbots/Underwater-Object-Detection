import cv2 as cv
import os
import numpy as np

import gate_detector


"""
Classes for labelling data
"""

class PoleHullLabeller:


    def create_labelled_dataset(self):
        """
        From the images folder, tries to open the images and for each image, allows the user to label the hulls as being 
        associated to a pole or not, and returns a dictionary of hulls to labels
        
        @returns: A dataset mapping the hulls of each image to their labels
        """

        print("-------------------------------------------------------------------")
        print("                 How to Use the Hull Label Tool")
        print("-------------------------------------------------------------------")
        print("- If a hull is NOT associated to a pole: press the 1 button")
        print("- If a hull IS associated to a pole: press the 2 button")
        print("\n- If any other key is pressed, the program EXITS")
        print("-------------------------------------------------------------------")

        detector = gate_detector.GateDetector(im_resize=3.0/4)

        imgs = []
        labels = []
        directory = os.path.dirname(os.getcwd())
        
        # Get absolute path of all images in the images folder
        for dirpath,_,filenames in os.walk(os.path.join(directory, 'images', 'gate')):
            for f in filenames:
                imgs.append(os.path.abspath(os.path.join(dirpath, f)))

        # Get the hulls from the segmented image and run the display and label program for each image
        for img in imgs:
            src = cv.imread(img, 1)
            pre = detector.preprocess(src)
            seg = detector.segment(pre)
            mor = detector.morphological(seg)
            hulls = detector.create_convex_hulls(seg)
            labels += self.display_and_label_hulls(hulls, pre)
        return labels

        
    def display_and_label_hulls(self, hulls, src):
        """
        Displays each hull and allows someone to label the hull as being associated to a pole or not and returns a
        data structure mapping hulls to labels

        @param hulls: The convex hulls 
        @param src: The source image from which teh convex hulls have been created

        @returns: A list where each entry is a tuple mapping a hull to it's label
        """
        
        labels = []

        for hull in hulls:

            angle = 0
            MA = 1
            ma = 1
            try:
                _,(MA,ma),angle = cv.fitEllipse(hull)
            except:
                pass
            cosAngle = np.abs(np.cos(angle*np.pi/180))

            # Only human-classify hulls if it is reasonably a vertically oriented rectangle
            # This is a hueristic to not have to waste time clasifying hulls clearly not poles
            if  (cosAngle < 1.75) and (cosAngle > 0.85) and (MA/ma < 0.28):
                cpy = src.copy()
                hull_img = cv.polylines(cpy, [hull], True, (0,0,255), 3)
                cv.imshow("Hull", hull_img)
                keycode = cv.waitKey(0)
                if keycode == 49:
                    labels.append((hull, 0))
                    print("Not a Pole")
                elif keycode == 50:
                    labels.append((hull, 1))
                    print("A Pole!")
                else:
                    raise Exception("Unexpected Key Pressed")
            else:
                labels.append((hull, 0))
        cv.destroyAllWindows()
        return labels

