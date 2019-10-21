#!/bin/bash

python scripts/extract_features_vg.py \
    -prototxt models/vg_faster_rcnn_end2end/test_rpn.prototxt \
    -caffemodel models/vg_faster_rcnn_end2end/resnet101_faster_rcnn_final.caffemodel \
    -images /datasets/nocaps/images/nd_valid \
    --from_image_path \
    -output /outputs/nd_valid;

python scripts/extract_features_vg.py \
    -prototxt models/vg_faster_rcnn_end2end/test_rpn.prototxt \
    -caffemodel models/vg_faster_rcnn_end2end/resnet101_faster_rcnn_final.caffemodel \
    -images /datasets/nocaps/images/od_valid \
    --from_image_path \
    -output /outputs/od_valid;
    
python scripts/extract_features_vg.py \
  -prototxt models/vg_faster_rcnn_end2end/test_rpn.prototxt \
  -caffemodel models/vg_faster_rcnn_end2end/resnet101_faster_rcnn_final.caffemodel \
  -images /datasets/nocaps/images/nd_test \
  --from_image_path \
  -output /outputs/nd_test;

python scripts/extract_features_vg.py \
    -prototxt models/vg_faster_rcnn_end2end/test_rpn.prototxt \
    -caffemodel models/vg_faster_rcnn_end2end/resnet101_faster_rcnn_final.caffemodel \
    -images /datasets/nocaps/images/od_test \
    --from_image_path \
    -output /outputs/od_test;


