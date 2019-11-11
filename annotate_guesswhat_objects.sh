#!/bin/bash

python scripts/extract_features_vg.py \
    -prototxt models/vg_faster_rcnn_end2end/test_rpn.prototxt \
    -caffemodel models/vg_faster_rcnn_end2end/resnet101_faster_rcnn_final.caffemodel \
    -images /datasets/guesswhat/img \
    --from_image_path \
    --force-boxes /datasets/guesswhat/object_annotations.json \
    -output /outputs/guesswhat/objects_fastrcnn/;