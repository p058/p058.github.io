---
layout: post
title:  "Yolo - Part 1"
date:   2018-01-30 21:43:43 -0600
categories: Machine Learning,
---

[Yolo](https://pjreddie.com/media/files/papers/YOLOv3.pdf) is a really
good object detector and pretty fast compared to other state of the art object detectors and
the author of Yolo is really really cool. Look at the author's [commit history](https://github.com/pjreddie/darknet/commits/master?after=508381b37fe75e0e1a01bcb2941cb0b31eb0e4c9+34)
and  [resume](https://pjreddie.com/static/Redmon%20Resume.pdf).

There are already a lot of blog posts explaining how yolo works, this post is **No** different.
I am writing this just to improving my writing skills. The author released ~~two~~ three different versions
of Yolo. Yolov1 is really fast but doesn't do a great job at detection compared to state of the art
detectors. Yolov2 and v3 are fast and do a better job at detection.
<!-- I will talk about Yolo v2 & v3 in this post. -->

## Object Detection

Given an image, the goal is to detect all the objects in the image. Broadly, you could do it in two
ways:
1. get all possible bounding boxes(bbox) i.e., proposals for an image and classify each of the bounding box to have an
object or not and the class of the object. If all the objects were of the same size, then there wouldn't
be many different possible bboxes, you fix the size of your bbox and roll it over the image to
get your bbox proposals, but if the objects are of variable size, you need many proposals of different
sizes. R-CNN, Fast R-CNN, Faster R-CNN all generate bbox proposals at some stage in the detection pipeline.
Since the network has to go through several bbox proposals, all these detectors run at <10 fps.

2. each bbox can be represented with 4 coordinates (center_x, center_y, obj_w, obj_h) , center of the object,
width & height of the object. you could directly regress to learn these four values, & classify to learn obj/noobj,
class scores. Yolo uses this approach. since you have to run the image through the network just once, these type
of detectors are fast. Yolo runs at > 30 fps.

<!-- ![variable-sizes]({{site.baseurl}}/images/large_vs_small.jpg){:class="img-responsive"} -->
<!-- source: http://www.cornel1801.com/animated/Gulliver-s-Travels-1939/part-5-welcome-to-lilliput.html -->

**Note:**The R-CNN's also use regression to predict the offsets to the bbox proposals.

## How does Yolo work?

In Yolov2, an image is passed through a Convolutional neural network(CNN) and the output of CNN is a feature map of size
(num_channels,  grid_width , grid_height)

![how_does_it_work]({{site.baseurl}}/images/how_does_it_work3.png){:class="img-responsive"}

where num_channels = (num_classes + 4 + 1) * num_anchors, 4 is for the four bbox coordinates & 1 is
for objectness score (whether or not there is an object in that grid cell), num_classes corresponds to the class score
predictions. num_anchors is the number of predictions at each grid cell. If you look closely, the network makes bbox's,
objectness scores, class scores predictions corresponding to various anchors at each grid cell.

What are the anchors doing here?

Consider an example where you are working with a pedestrian dataset, you know that most of your bounding boxes are going
to be thin and tall, so instead of predicting width, height of your bounding boxes directly, you can predict offsets to
some predefined bounding boxes called anchor boxes (or dimension clusters)

How are anchor boxes defined?

Anchor boxes are defined by their width, height. In Yolov2 the anchor boxes are choosed based on K-means clustering.
From your training dataset extract all your objects width, height in the following format:

0 w1 0 h1

0 w2 0 h2

...

where w1, h1... are the width & height scaled to 1. Once you have all your objects, cluster them using K-means for
various values of k with distance metric d(a, b)= 1 - IoU(a, b), where IoU is the intersection over union between two boxes.
boxes of similar sizes have low distance between them. choose an appropriate value for k & corresponding cluster centers.
cluster centers are your anchor boxes.

Running K-means On the VOC dataset with k=5 will give 5 anchor boxes & they look something like this:

![raw_anchors]({{site.baseurl}}/images/raw_anchors.png){:class="img-responsive"}

In the above, the taller and thinner anchor boxes could be for detecting objects like person, tree e.t.c. where as the
wider ones could be for detecting objects like car, buses e.t.c. So, in this example at each grid cell in the output
feature map, the network makes 5 predictions, each corresponding to a different anchor.

If we scale the predictions on the output feature map to that of image size and plot predictions for just one grid cell,
we see:

![anchors_at_a_location]({{site.baseurl}}/images/anchors_at_a_location.png){:class="img-responsive"}

In the above image, the red boxes are the anchor boxes, green boxes are the adjusted bounding boxes and scores are the objectness
scores. Obviously, not all predictions at a grid cell are valid, we filter them based on objectness scores.

Hopefully, the anchor boxes part is clear.

**Note:** In yolov2, the network doesn't predict the bbox coordinates directly but instead uses the following parametrization:

if bx, by, bw, bh are the actual bbox coordinates & tx, ty, tw, th, to are network predictions, they are related as follows:

        bx = σ(tx) + cx
        by = σ(ty) + cy
        bw = p_w * exp(tw)
        bh = p_h * exp(th)
        Pr(object) ∗ IoU(b, object) = σ(to)

cx, cy are the distances to the top left corner of the grid in consideration from the top left corner of the feature map. The below
figure may help clarify the above equations if they are not already clear.

![parametrization]({{site.baseurl}}/images/feature_map_example.png){:class="img-responsive"}

In the above image, an example 5 x 5 grid(output feature map) is overlayed on the input image and
an example anchor box & bounding box dimensions are shown at a grid cell location.

Lets go through an example:

**Step 1:** Raw Image. The image is resized into the shape that the networks expect, in this case,
we resize the image to (416, 416)

![raw_image]({{site.baseurl}}/images/raw_image.png){:class="img-responsive"}

**Step 2:** The resized image is passed through CNN (we will talk about the CNN later) and we get
an output tensor of size (num_channels, cell_width, cell_height) and after we convert the
output tensor to actual bbox coordinates and plot them, we get something like this:

![img_with_all_outputs]({{site.baseurl}}/images/img_with_all_outputs.png){:class="img-responsive"}

In the above image, there are 5 * 13 * 13 = 845 bounding boxes.

PyTorch code to convert the CNN output to actual bbox output would look something like this, this is
basically code for the above equations.

breakdown the CNN output

```python
batch_size, num_predictions, cell_width, cell_height = output.size()

# resize the output
output = output.view(batch_size, num_anchors, (5 + num_classes),
                     cell_width,
                     cell_height)

# break the output
tx, ty, tw, th, to, tcls_hat = [output[:, :, 0, :, :].unsqueeze(2),
                               output[:, :, 1, :, :].unsqueeze(2),
                               output[:, :, 2, :, :].unsqueeze(2),
                               output[:, :, 3, :, :].unsqueeze(2),
                               output[:, :, 4, :, :],
                               output.narrow(2, 5, num_classes).contiguous()]
```

create a meshgrid of cx,cy values

```python

cx = output.data.new(np.linspace(0, cell_width - 1, cell_width)).float().expand(batch_size,
                                                                                num_anchors,
                                                                                1,
                                                                                cell_width,
                                                                                cell_height)

cy = output.data.new(np.linspace(0, cell_height - 1, cell_height)).float().expand(batch_size,
                                                                                  num_anchors,
                                                                                  1,
                                                                                  cell_width,
                                                                                  cell_height). \
    transpose(3, 4)
```

create anchor widths, heights tensors and expand them to required sizes

```python
pw = output.data.new([_wh[0] for _wh in anchors]).float().expand(batch_size,
                                                                            cell_width,
                                                                            cell_height,
                                                                            num_anchors).transpose(1, 3)

ph = output.data.new([_wh[1] for _wh in anchors]).float().expand(batch_size,
                                                                             cell_width,
                                                                             cell_height,
                                                                             num_anchors).transpose(1, 3)
```


Finally, transform using the above equations:

```python
bx = (torch.sigmoid(tx).data + cx)
by = (torch.sigmoid(tx).data + cy)
bw = (torch.exp(tw).data * pw.unsqueeze(2))
bh = (torch.exp(th).data * ph.unsqueeze(2))
bbox = torch.cat([bx, by, bw, bh], 2)
```

**Step 3** : The above image has all the predictions, but we only need those predictions where there is an
object, so we filter based on the objectness score. We set object_threshold to 0.4 and filter the predictions.
Plotting the filtered predictions would look like this:

![img_with_filtered_outputs]({{site.baseurl}}/images/img_with_filtered_outputs.png){:class="img-responsive"}

**Step 4** : This looks much better, but there are still some overlapping bounding boxes. To clear those,
we drop overlapping bounding boxes using Non Maximum Supression. What we do here is sort all the
filtered bbox's based on objectness score and go through bbox's from top to bottom and drop all the bboxes with
an overlap (IoU) greater than a certain threshold. Final output would look like this:

![img_with_final_outputs2]({{site.baseurl}}/images/img_with_final_outputs2.png){:class="img-responsive"}

We have seen how to go from input image to objects using Yolov2, but we haven't seen the CNN architecture &
how the CNN is trained. We will go over those in the next post.
