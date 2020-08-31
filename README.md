# FloorPlanObjectDetection

There are several methods to object detection:
Labeling the objects using  contour/ bounding box: 
Object Detection using Template Matching which can be done without machine learning, using template image.
Object detection using deep learning: 
There are models available for doing object detection recognition in an image. For example, the most prominent ones are RCNN, fast RCNN, faster RCNN, mask RCNN, Yolo and SSD. However, most of the them are developed for natural images/Real life images. 
In this case, we are working on document images ex floor plans, where we need to detect objects like Windows, flowerpots, dining table, Bed, Computer, sofa, sink, etc.
There are not many model available for the  documents type images.
In order to apply deep learning, we need training image labelled with classes (in our case, icons used by architecture software in a floorplan).
I tried to use existing  YOLO model ,and loaded  pretrained model and weight for document type, with darkflow implementation and  however facing import issue with Darflow (https://github.com/thtrieu/darkflow).
In the interest of time, have labeled the objects using contour only. 
