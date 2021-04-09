# USAGE
#
#   1. Build the docker container:
# $ docker build -t  . ratocer
#   2. Run the container with a bash
# $ docker run -it --rm -v "imgpath":/usr/src/app --user "$(id -u):$(id -g)"  ratocer main.py "imgpath"


FROM ubuntu:20.04
WORKDIR /usr/src/app
COPY . .
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin
RUN apt-get update && apt-get install --no-install-recommends -y software-properties-common python3 python3-pip wget \
    && apt-get update \
    && add-apt-repository ppa:alex-p/tesseract-ocr-devel \
    && apt-get update \
    && pip3 install --no-cache-dir -r requirements.txt \
    && apt-get install -y --no-install-recommends libleptonica-dev tesseract-ocr\
    && wget -O  /usr/share/tesseract-ocr/5/tessdata/frak2021.traineddata https://digi.bib.uni-mannheim.de/~jkamlah/tesseract/models/frak2021.traineddata \
    && apt-get clean all

ENTRYPOINT ["python3"]
