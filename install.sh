#!/bin/sh
# CachePotato installer.
#
#   curl -fsSL https://raw.githubusercontent.com/theryeguy92/CachePotato/main/install.sh | sh
#
# Downloads the prebuilt `cachepotato` binary for this machine from the latest GitHub release,
# checks it against the release's SHA-256, and puts it in ~/.local/bin. Nothing is installed
# system-wide and no model is downloaded: `cachepotato run qwen3-30b-a3b` does that on first use.
#
# Environment:
#   CACHEPOTATO_VERSION      a release tag such as v0.1.0 (default: the latest release)
#   CACHEPOTATO_INSTALL_DIR  where the binary goes (default: ~/.local/bin)
#   CACHEPOTATO_DOWNLOAD_URL a directory holding the release assets, instead of GitHub
#                            (a mirror, or file:///path for testing a locally built package)
#
# Prefer not to pipe a script into a shell? Download the .tar.gz and its .sha256 from
# https://github.com/theryeguy92/CachePotato/releases, run `sha256sum -c`, and copy the binary
# onto your PATH. This script does exactly that.
set -eu

REPO="theryeguy92/CachePotato"
TARGET="x86_64-unknown-linux-gnu"
ASSET="cachepotato-$TARGET.tar.gz"
DEST="${CACHEPOTATO_INSTALL_DIR:-$HOME/.local/bin}"

say() { printf '%s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

# ---- can this machine run it? Checked before downloading anything. ----
[ "$(uname -s)" = "Linux" ] || die "CachePotato runs on Linux only for now (this is $(uname -s))."
[ "$(uname -m)" = "x86_64" ] || die "CachePotato needs an x86-64 CPU (this is $(uname -m))."
flags=$(grep -m1 '^flags' /proc/cpuinfo 2>/dev/null || true)
case " $flags " in *" avx2 "*) ;; *) die "this CPU lacks AVX2, which the engine's kernels require." ;; esac
case " $flags " in *" fma "*) ;; *) die "this CPU lacks FMA, which the engine's kernels require." ;; esac

if command -v curl >/dev/null 2>&1; then
    fetch() { curl -fsSL --retry 3 -o "$2" "$1"; }
elif command -v wget >/dev/null 2>&1; then
    fetch() { wget -q -O "$2" "$1"; }
else
    die "neither curl nor wget is installed."
fi
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required (coreutils)."
command -v tar >/dev/null 2>&1 || die "tar is required."

if [ -n "${CACHEPOTATO_DOWNLOAD_URL:-}" ]; then
    BASE="${CACHEPOTATO_DOWNLOAD_URL%/}"
elif [ -n "${CACHEPOTATO_VERSION:-}" ]; then
    BASE="https://github.com/$REPO/releases/download/$CACHEPOTATO_VERSION"
else
    BASE="https://github.com/$REPO/releases/latest/download"
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT INT TERM

say "Downloading $BASE/$ASSET ..."
fetch "$BASE/$ASSET" "$tmp/$ASSET" || die "download failed: $BASE/$ASSET"
fetch "$BASE/$ASSET.sha256" "$tmp/$ASSET.sha256" || die "download failed: $BASE/$ASSET.sha256"
(cd "$tmp" && sha256sum -c --status "$ASSET.sha256") || die "checksum mismatch: the download is corrupt or was tampered with."

tar -xzf "$tmp/$ASSET" -C "$tmp"
mkdir -p "$DEST"
install -m 755 "$tmp/cachepotato-$TARGET/cachepotato" "$DEST/cachepotato"

say ""
say "Installed $("$DEST/cachepotato" --version) to $DEST/cachepotato"
case ":$PATH:" in
    *":$DEST:"*) ;;
    *) say ""
       say "$DEST is not on your PATH. Add this to your shell's startup file:"
       say "  export PATH=\"$DEST:\$PATH\"        (bash/zsh)"
       say "  fish_add_path $DEST                 (fish)" ;;
esac
say ""
say "Next:"
say "  cachepotato doctor              check memory, disk and what this machine can run"
say "  cachepotato run qwen3-30b-a3b   download the model (17.5 GB) and start chatting"
