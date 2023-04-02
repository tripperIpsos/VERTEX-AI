FROM us-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.0-23:latest

WORKDIR /

COPY trainer/* /trainer/

RUN pip3 install --no-cache-dir --upgrade pip

RUN pip3 install --no-cache-dir -r trainer/requirements.txt

# 5. Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python3", "trainer/train.py"]
