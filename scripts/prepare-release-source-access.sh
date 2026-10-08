#!/usr/bin/env bash
# Hosted source builds only. TensorD is public; TensorFS may need existing read access.
set -euo pipefail
export GIT_TERMINAL_PROMPT=0
repository=https://github.com/cozy-creator/tensorfs.git
if git ls-remote "$repository" HEAD >/dev/null 2>&1; then
  echo 'TensorFS source is accessible without additional SSH configuration.'
  exit 0
fi
if [[ -z "${TENSORFS_KEY:-}" ]]; then
  echo '::error::Hosted source mode cannot read private TensorFS. Provide existing TensorFS read access via TENSORFS_READ_KEY, or build with local authenticated Git and select mode=prebuilt with approved hashes. TensorD is public and needs no COZY_MACHINE_READ_KEY.' >&2
  exit 1
fi
mkdir -m 700 -p ~/.ssh
(umask 077; printf '%s\n' "$TENSORFS_KEY" > ~/.ssh/tensorfs)
# Validate without printing key material, before any expensive wheel/kernel work.
ssh-keygen -y -P '' -f ~/.ssh/tensorfs >/dev/null
ssh-keyscan github.com >> ~/.ssh/known_hosts
printf 'Host tensorfs.github.com\n  HostName github.com\n  IdentityFile ~/.ssh/tensorfs\n  IdentitiesOnly yes\n' >> ~/.ssh/config
git config --global url.'git@tensorfs.github.com:cozy-creator/tensorfs'.insteadOf https://github.com/cozy-creator/tensorfs
git ls-remote "$repository" HEAD >/dev/null
