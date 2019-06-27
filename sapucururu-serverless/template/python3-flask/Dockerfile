FROM openfaas/of-watchdog:0.5.3 as watchdog
FROM python:3.6-stretch

COPY --from=watchdog /fwatchdog /usr/bin/fwatchdog
RUN chmod +x /usr/bin/fwatchdog

ARG ADDITIONAL_PACKAGE
# Alternatively use ADD https:// (which will not be cached by Docker builder)
RUN apt-get update \
    && apt-get install -y apt-utils curl unzip bash build-essential musl-dev gcc make ${ADDITIONAL_PACKAGE}

# Add non root user
RUN useradd -m app && echo "app:app" | chpasswd && adduser app sudo
RUN chown -R app /home/app

USER app

ENV PATH=$PATH:/home/app/.local/bin

WORKDIR /home/app/

COPY index.py           .
COPY requirements.txt   .
COPY trained-models/*   ./

USER root
RUN pip install -r requirements.txt

USER app
RUN mkdir -p function
RUN touch ./function/__init__.py
WORKDIR /home/app/function/
COPY function/requirements.txt	.
RUN pip install --user -r requirements.txt

WORKDIR /home/app/

USER root
COPY function           function
#RUN chown -R app:app ./
USER app

ENV imodel_architecture_path="/home/app/frog_identifier_cifar10-final_model_architecture.json"
ENV imodel_wights_path="/home/app/frog_identifier_cifar10-final_model_weights.h5"
ENV itarget_size="32,32"
ENV cmodel_architecture_path="/home/app/frog_classifier_imagenet128-final_model_architecture.json"
ENV cmodel_wights_path="/home/app/frog_classifier_imagenet128-final_model_weights.h5"
ENV ctarget_size="128,128"

ENV mode="http"
ENV cgi_headers="true"
ENV fprocess="python3 index.py"
ENV upstream_url="http://127.0.0.1:5000"

#HEALTHCHECK --interval=10s CMD [ -e /tmp/.lock ] || exit 1

CMD ["fwatchdog"]
