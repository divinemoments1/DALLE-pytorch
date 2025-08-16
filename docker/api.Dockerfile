# syntax=docker/dockerfile:1

ARG PYTORCH_IMAGE=pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime
FROM ${PYTORCH_IMAGE} as base

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
	git \
	wget \
	ca-certificates \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install repo and app requirements
COPY app/requirements.txt /app/app-requirements.txt
RUN python -m pip install --upgrade pip && \
	pip install -r /app/app-requirements.txt

# Install current library (this repo)
COPY . /app/src
RUN pip install -e /app/src

# Runtime env
ENV OUTPUTS_DIR=/data/outputs
RUN mkdir -p ${OUTPUTS_DIR}
VOLUME ["/data"]

EXPOSE 8000

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]