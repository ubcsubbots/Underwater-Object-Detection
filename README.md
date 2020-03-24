
# Underwater-Object-Detection
For formulating and testing underwater object detection algorithms.  Note that this is a repo for designing and testing, which is why it is in python, the actual implementation of the algorithms will be in C++ for better performance.

### Gate Detector
To run the gate detector algorithm run

`python main.py --gate <type> --resize <im_resize> [---debug][--record]`

Where `<type>` is one of  `im/vid/label`, the `--resize` flag is a fraction defining the scale of the images the algorithm is run on, and `--debug --record` are flags that dictate whether or not to add debug information to the algorithm output and whether or not to record (only works with `vid` type). An example command would be

`python main.py --gate vid --resize 1/2 --record`

![](gate_example.gif)
