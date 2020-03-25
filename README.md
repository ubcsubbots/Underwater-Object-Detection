
# Underwater-Object-Detection
For formulating and testing underwater object detection algorithms.  Note that this is a repo for designing and testing, which is why it is in python, the actual implementation of the algorithms will be in C++ for better performance.

### Gate Detector
To run the gate detector algorithm run

`python main.py --gate <type> --resize <im_resize> [---debug][--record]`

Where `<type>` is one of  `im/vid/label`, the `--resize` flag is a fraction defining the scale of the images the algorithm is run on, and `--debug --record` are flags that dictate whether or not to add debug information to the algorithm output and whether or not to record (only works with `vid` type). An example command would be

`python main.py --gate vid --resize 1/2 --record`

Here is an gif showing some of the inner workings of the algorithm

![](/videos/gate_example.gif)

From top left going clockwise we have the thresholded saturation gradient combined with orange hue mask, convex hulls drawn around connected clusters of the segmented image, visualized basis alignment, and gate detection and pose estimation. The algorithm works by first segmentating a preprocessed image by performing a bitwise OR on the thresholded saturation gradient and orange hue mask. The saturation gradient works well at identifying the orange poles from far away, but fails closer to the gate which is where the orange hue mask is best at identifying the poles. Then, we calculate the convex hulls around the connected components of the segmented image and identify which hulls belong to a pole using an SVM classifier. The hulls are filtered before classification to ignore hulls whose area is much too small or large to be associated to a pole. Using these hulls, we can get the extrema points of the bounding box which wraps around the orange poles of either side of the gate, which is drawn on the image in the bottom right in red. Furthermore, since we know the true dimensions of the gate, we can calculate the nessary rotation and translation needed to bring the gate into the desired position (planar and centered in the image) based on the percieved pose of the gate which can be derived from the bounding box. This basis translation step is shown in the bottom left where the dark green represents the current basis of the gate and light green represents the desired basis.  
