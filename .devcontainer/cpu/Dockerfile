FROM ubuntu:22.04
RUN apt-get update\ 
    && apt-get install -y git  python3 python3-pip 
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN pip install pandas openpyxl matplotlib