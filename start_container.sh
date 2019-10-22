#!/bin/bash

nvidia-docker run -it --name vg_container -v $PWD/scripts:/workspace/scripts -v /scratch/ale_models/nocaps/img:/datasets/nocaps/images -v /scratch/ale_models/nocaps/annotations:/datasets/nocaps/annotations -p 8880:8880 vg_image /bin/bash
