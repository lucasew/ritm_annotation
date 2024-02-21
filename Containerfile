FROM python:3.10
COPY . /app
WORKDIR /app
RUN apt update && apt install -y libgl1-mesa-glx libglib2.0-0
RUN pip install . --progress-bar off
CMD ["ritm_annotation"]
