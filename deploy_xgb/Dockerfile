FROM python:3.8-slim
WORKDIR /pipelines/components/src
COPY requirements.txt /pipelines/components/src
RUN pip install -r requirements.txt
COPY deploy_xgb.py /pipelines/components/src
EXPOSE 5000
RUN mkdir /model
COPY data /model
CMD exec python3 /pipelines/components/src/deploy_xgb.py