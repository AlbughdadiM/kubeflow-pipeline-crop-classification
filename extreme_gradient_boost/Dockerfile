FROM python:3.8-slim
WORKDIR /pipelines/components/src
COPY requirements.txt /pipelines/components/src
RUN pip install -r requirements.txt
COPY extreme_gradient_boost.py /pipelines/components/src