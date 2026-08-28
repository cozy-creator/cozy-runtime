# The BOOT proof's substrate stand-in. Not the shipped substrate image (that recipe lives
# in tensorhub `build/substrate/` and is pinned to a cozy-creator bootstrap commit); this
# is the same three grants the substrate makes to the adapter — the control-Runtime wheel
# at its fixed path, /run/cozy, /var/lib/cozy — with nothing else in the image, so a boot
# that reaches ready-but-empty here reached it on the adapter's own code.
ARG PYTHON_IMAGE=python:3.12-slim
FROM ${PYTHON_IMAGE}
RUN apt-get update \
 && apt-get install -y --no-install-recommends openssl ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY wheels/ /wheels/
RUN python -m pip install --no-cache-dir /wheels/cozy_runtime-*.whl \
 && mkdir -p /opt/cozy/control-runtime /run/cozy /var/lib/cozy
RUN cp /wheels/cozy_runtime-*.whl /opt/cozy/control-runtime/cozy-runtime.whl
COPY nvidia-smi /usr/bin/nvidia-smi
COPY pod.py /pod.py
RUN chmod 0755 /usr/bin/nvidia-smi
ENTRYPOINT ["python", "/pod.py"]
