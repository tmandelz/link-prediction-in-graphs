FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime
RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib/libcudart.so' >> ~/.bashrc
RUN apt-get update --allow-insecure-repositories && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN mkdir /work
WORKDIR /work
RUN pip install -U git+https://github.com/huggingface/accelerate.git

# install requirements
COPY additional/i4ds_cluster/requirements.txt requirements.txt
RUN pip install -r requirements.txt
# install sparse and scatter explicitly
RUN pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

WORKDIR /work/project
