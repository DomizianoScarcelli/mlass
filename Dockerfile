# For more information, please refer to https://aka.ms/vscode-docker-python
FROM --platform=linux/amd64 python:3.8.5

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN apt-get update
RUN apt-get -y install mpich
RUN apt-get -y install libsndfile1

# Install pip requirements
RUN pip install --upgrade pip
COPY lass_audio/requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /
COPY . .

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
# CMD ["python", "-m", "lass_audio.lass.separate"]
CMD ["python", "-m", "lass_audio.lass.separate.py"]

