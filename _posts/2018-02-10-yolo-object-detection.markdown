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
detectors. Yolov2 and v3 are fast and do a better job at detection. I will talk about Yolo v2 & v3 in
this post.

## Object Detection

Given an image, the goal is to detect all the objects in the image. Broadly, you could do it in two
ways:
1. get all the bounding boxes(bbox) possible i.e., proposals on an image and classify each of the bounding box to have an
object or not and the class of the object. If all the objects were of the same size, then there wouldn't
be many different possible bboxes, you fix the size of your bbox and roll it over the image to
get your bbox proposals, but if the objects are of variable size, you need many proposals of different
sizes. R-CNN, Fast R-CNN, Faster R-CNN all generate bbox proposals at some stage in the detection pipeline.
Since the network has to go through several bbox proposals, all these detectors run at <10 fps.

![variable-sizes]({{site.baseurl}}/images/330-tallest-and-shortest-man.jpg){:class="img-responsive"}

    source: http://www.101qs.com/330-tallest-and-shortest-man

2. each bbox has 4 coordinates (center_x, center_y, obj_w, obj_h) , center of the object, width & height
of the object. you could directly regress to learn these four values, & classify to get obj/noobj, class
scores. Yolo uses this approach. since you have to run the image through the network just once, these type
of detectors are fast. Yolo runs at > 30 fps.

    Note:The R-CNN's also use regression to predict the offsets to the bbox proposals.

## How does Yolo work?

In Yolo, a image is passed through a CNN and the output of CNN is a tensor of size
(num_channels,  grid_width , grid_height)

num_channels --> (num_anchors + 4 + 1) * num_classes

![how_does_it_work]({{site.baseurl}}/images/how_does_it_work.jpg){:class="img-responsive"}





