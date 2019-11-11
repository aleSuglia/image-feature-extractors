#!/bin/bash

nvidia-docker run -it --name vg_container -v $PWD/scripts:/workspace/scripts -v /scratch/asuglia/data/nocaps/img:/datasets/nocaps/images -v /scratch/asuglia/data/nocaps/annotations:/datasets/nocaps/annotations -v /scratch/asuglia/image-feature-extractors/models/:/workspace/models/ -p 8880:8880 vg_image /bin/bash
