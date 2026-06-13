#!/usr/bin/env bash
# One-shot bootstrap for xdna-npu-toolkit.
#   curl -fsSL https://raw.githubusercontent.com/tibrezus/xdna-npu-toolkit/main/install.sh | bash
set -euo pipefail

PY="${PYTHON:-python3}"
echo "Using: $($PY --version)"

# Run straight from source -- zero install, no deps. Just grab the package dir.
DEST="${XNPU_HOME:-$HOME/.local/share/xdna-npu-toolkit}"
mkdir -p "$(dirname "$DEST")"

if [ -d "$DEST/.git" ]; then
  git -C "$DEST" pull -q 2>/dev/null || true
else
  if ! git clone -q https://github.com/tibrezus/xdna-npu-toolkit "$DEST" 2>/dev/null; then
    echo "git clone failed; falling back to tarball"
    curl -fsSL https://github.com/tibrezus/xdna-npu-toolkit/archive/refs/heads/main.tar.gz \
      | tar xz -C "$DEST" --strip-components=1
  fi
fi

# Make an executable shim on $PATH if ~/.local/bin exists.
if [ -d "$HOME/.local/bin" ]; then
  cat > "$HOME/.local/bin/xdna-npu" <<EOF
#!/usr/bin/env bash
exec $PY -m xdna_npu "\$@"
EOF
  chmod +x "$HOME/.local/bin/xdna-npu"
  echo
  echo "Installed. Run:  xdna-npu doctor"
else
  echo
  echo "Installed to $DEST. Run:  PYTHONPATH=$DEST $PY -m xdna_npu doctor"
fi
