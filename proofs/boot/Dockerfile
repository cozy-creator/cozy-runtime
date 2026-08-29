# Runtime-side stand-in for the base worker image. The base Python has the control
# Runtime and its small dependencies installed once; the package wheel remains only in the
# proof origin and must arrive through ArtifactGrant after boot.
ARG PYTHON_IMAGE=python:3.12-slim
FROM ${PYTHON_IMAGE}
RUN apt-get update \
 && apt-get install -y --no-install-recommends openssl ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY wheels/ /wheels/
# Exact base-family wheels, used only to author the stand-in manifest the real image builder
# bakes. They are installed in the base Python, never copied into a package environment.
RUN python -m pip download --only-binary=:all: -d /wheels/deps \
      msgspec protobuf grpcio
RUN python -m pip install --no-cache-dir /wheels/cozy_runtime-*.whl \
 && mkdir -p /opt/cozy/base-runtime /run/cozy /var/lib/cozy
COPY nvidia-smi /usr/bin/nvidia-smi
COPY pod.py hub.py proof_data.py /
RUN chmod 0755 /usr/bin/nvidia-smi
ENTRYPOINT ["python", "/pod.py"]
