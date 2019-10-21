#!/bin/bash

python scripts/extract_features_vg.py \
    -prototxt models/vg_faster_rcnn_end2end/test_rpn.prototxt \
    -caffemodel models/vg_faster_rcnn_end2end/resnet101_faster_rcnn_final.caffemodel \
    -images /datasets/nocaps/images/nd_valid \
    -output /outputs/nd_valid;

python scripts/extract_features_vg.py \
    -prototxt models/vg_faster_rcnn_end2end/test_rpn.prototxt \
    -caffemodel models/vg_faster_rcnn_end2end/resnet101_faster_rcnn_final.caffemodel \
    -images /datasets/nocaps/images/od_valid \
    -output /outputs/od_valid;
    
python scripts/extract_features_vg.py \
  -prototxt models/vg_faster_rcnn_end2end/test_rpn.prototxt \
  -caffemodel models/vg_faster_rcnn_end2end/resnet101_faster_rcnn_final.caffemodel \
  -images /datasets/nocaps/images/nd_test \
  -output /outputs/nd_test;

python scripts/extract_features_vg.py \
    -prototxt models/vg_faster_rcnn_end2end/test_rpn.prototxt \
    -caffemodel models/vg_faster_rcnn_end2end/resnet101_faster_rcnn_final.caffemodel \
    -images /datasets/nocaps/images/od_test \
    -output /outputs/od_test;


