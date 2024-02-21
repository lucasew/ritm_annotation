FROM python:3.10
RUN apt update && apt install -y libgl1-mesa-glx libglib2.0-0
COPY requirements.txt /
RUN pip install -r /requirements.txt --progress-bar off
WORKDIR /app
COPY . /app
RUN pip install . --progress-bar off
ENTRYPOINT ["ritm_annotation"]
