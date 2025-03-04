FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime 
RUN mkdir /job
WORKDIR /job
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# You should install any dependencies you need here.

RUN pip install pandas 
RUN pip install alphabet-detector
# RUN pip install datasets # Don't need this since only used for training data