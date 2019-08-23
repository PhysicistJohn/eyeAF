FROM ubuntu

WORKDIR /app

ADD . /app

# Updating Ubuntu packages
RUN apt-get update && yes|apt-get upgrade
RUN apt-get install -y wget bzip2 cmake build-essential libgtk2.0-dev pkg-config
RUN wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
RUN bash Anaconda3-5.0.1-Linux-x86_64.sh -b
RUN rm Anaconda3-5.0.1-Linux-x86_64.sh
ENV PATH /root/anaconda3/bin:$PATH
#RUN conda update conda
#RUN conda update anaconda
#RUN conda update --all
RUN pip install imutils
RUN pip install dlib
RUN conda install --file requirements.txt


ENV NAME FaceDetect

CMD ["python","-i","detect_face_parts.py"]