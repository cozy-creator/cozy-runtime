# The BOOT proof's substrate stand-in. Not the shipped substrate image (that recipe lives
# in tensorhub `build/substrate/` and is pinned to a cozy-creator bootstrap commit); this
# is the same grants the substrate makes to the adapter — the control-Runtime wheel and the
# one materializer's `uv` at their fixed paths, /run/cozy, /var/lib/cozy — with nothing else
# in the image, so a boot that reaches ready-but-empty here reached it on the adapter's own
# code, and a placement that converges here converged by fetching every byte.
ARG PYTHON_IMAGE=python:3.12-slim
FROM ${PYTHON_IMAGE}
RUN apt-get update \
 && apt-get install -y --no-install-recommends openssl ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY wheels/ /wheels/
# The wheelhouse the RELEASE names, downloaded once at image build. The materialized venv
# installs these offline from exact bytes; nothing is resolved at convergence time.
RUN python -m pip download --no-deps --only-binary=:all: -d /wheels/deps \
      msgspec protobuf grpcio
RUN python -m pip install --no-cache-dir /wheels/cozy_runtime-*.whl \
      /wheels/proof_cli_endpoint-*.whl \
 && mkdir -p /opt/cozy/control-runtime /run/cozy /var/lib/cozy
RUN cp /wheels/cozy_runtime-*.whl /opt/cozy/control-runtime/cozy-runtime.whl
COPY uv /opt/cozy/control-runtime/uv
COPY nvidia-smi /usr/bin/nvidia-smi
COPY pod.py hub.py proof_data.py /
RUN chmod 0755 /usr/bin/nvidia-smi /opt/cozy/control-runtime/uv
ENTRYPOINT ["python", "/pod.py"]
