---
layout: post
title:  "Yolo"
date:   2018-03-30 21:43:43 -0600
categories: Machine Learning,
---

[Yolo](https://pjreddie.com/media/files/papers/YOLOv3.pdf) is a really
good object detector and pretty fast compared to other state of the art object detectors and
the author of Yolo is really really cool. Look at the author's [commit history](https://github.com/pjreddie/darknet/commits/master?after=508381b37fe75e0e1a01bcb2941cb0b31eb0e4c9+34)
and  [resume](https://pjreddie.com/static/Redmon%20Resume.pdf).

There are already a lot of blog posts explaining how yolo works, this post is **No** different.
I am writing this just to improving my writing skills. The author released three different versions
of Yolo. Yolov1 is really fast but doesn't do a great job at detection compared to state of the art
detectors. Yolov2 and v3 are fast and do a better job at detection.
<!-- I will talk about Yolo v2 & v3 in this post. -->

## Object Detection

Given an image, the goal is to detect all the objects in the image. Broadly, you could do it in two
ways:
1. get all the bounding boxes(bbox) possible i.e., proposals on an image and classify each of the bounding box to have an
object or not and the class of the object. If all the objects were of the same size, then there wouldn't
be many different possible bboxes, you fix the size of your bbox and roll it over the image to
get your bbox proposals, but if the objects are of variable size, you need many proposals of different
sizes. R-CNN, Fast R-CNN, Faster R-CNN all generate bbox proposals at some stage in the detection pipeline.
Since the network has to go through several bbox proposals, all these detectors run at <10 fps.


![variable-sizes]({{site.baseurl}}/images/large_vs_small.jpg){:class="img-responsive"}
source: http://www.cornel1801.com/animated/Gulliver-s-Travels-1939/part-5-welcome-to-lilliput.html

2. each bbox has 4 coordinates (center_x, center_y, obj_w, obj_h) , center of the object, width & height
of the object. you could directly regress to learn these four values, & classify to get obj/noobj, class
scores. Yolo uses this approach. since you have to run the image through the network just once, these type
of detectors are fast. Yolo runs at > 30 fps.

    **Note:**The R-CNN's also use regression to predict the offsets to the bbox proposals.

## How does Yolo work?

In Yolov2, an image is passed through a Convolutional neural network and the output of CNN is a tensor(feature map) of size
(num_channels,  grid_width , grid_height)

![how_does_it_work]({{site.baseurl}}/images/how_does_it_work3.png){:class="img-responsive"}

num_channels --> (num_classes + 4 + 1) * num_anchors, 4 is for the four bbox coordinates & 1 is
for objectness score (whether or not there is an object in that grid cell), num_classes corresponds to the class score
predictions. num_anchors is the number of predictions at each grid cell. If you look closely, the network makes `num_anchors`
bbox's, objectness scores, class scores predictions at each grid cell.

In Yolov1 width, height of the objects are predicted directly. In Yolov2, instead of predicting the width/height of the object directly
they are predicted w.r.t anchor boxes. For example, if you are working on a pedestrian dataset, you know most of your
objects are going to have thinner and taller bounding boxes, so instead of predicting w, h directly, they are predicted with respect to the
predefined anchor boxes and these anchor boxes are calculated using K-means clustering.

** Note: ** In yolov2, the network doesn't predict the bbox coordinates directly but instead uses the following parametrization:

if bx, by, bw, bh are the actual bbox coordinates & tx, ty, tw, th, to are network predictions, they are related as follows:

        bx = σ(tx) + cx
        by = σ(ty) + cy
        bw = p_w * exp(tw)
        bh = p_h * exp(th)
        Pr(object) ∗ IoU(b, object) = σ(to)

cx, cy are the width, height to the grid in consideration from the top left corner. The below figure may help clarify the above
equations if they are not clear.

![parametrization]({{site.baseurl}}/images/feature_map.png){:class="img-responsive"}





