FROM python:3.10

RUN apt update && \
    apt --no-install-recommends install -y \
        gfortran \
        libblas-dev \
        liblapack-dev && \
    pip install --no-cache-dir \
        cython \
        numpy \
        scipy \
        six && \
    rm -rf /var/lib/apt/lists/*

COPY . /workspace/ttpy

RUN cd /workspace/ttpy && \
    pip install -v --no-cache-dir . && \
    rm -rf /workspace/ttpy && \
    python -c "import tt"

WORKDIR /workspace
