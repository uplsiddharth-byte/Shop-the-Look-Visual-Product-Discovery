# Hugging Face Space (Docker SDK), free CPU. Code only: the data (3 GB) is downloaded from a private dataset repo at start-up.
FROM python:3.12-slim
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 DEVICE=cpu HOST=0.0.0.0 PORT=7860 \
    HF_HOME=/home/user/.cache/huggingface OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
# CPU-only PyTorch wheels (the default Linux wheel drags in about 2 GB of CUDA libraries)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY --chown=user app ./app
COPY --chown=user web ./web
COPY --chown=user start.sh .
EXPOSE 7860
CMD ["bash", "start.sh"]
