FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime 
RUN mkdir /job
WORKDIR /job
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# You should install any dependencies you need here.

RUN apt-get update && apt-get install -y curl && \
    curl https://sh.rustup.rs -sSf | sh -s -- -y && \ 
    export PATH="$HOME/.cargo/bin:$PATH"
RUN pip install pandas 
RUN pip install -U sentence-transformers
