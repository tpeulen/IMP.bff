#!/usr/bin/env bash
# The headers the wasm build compiles against, into one prefix outside the
# checkout (default ~/opt/wasm-deps):
#
#   tools/pyodide/wasm_deps.sh [prefix]
#
# The IMP-free core needs three header-only libraries -- Boost (random, math,
# histogram, range), Eigen and cereal -- and nothing compiled, so there is
# nothing to cross-build: the headers are the same files a native build uses.
# Eigen is installed through its own CMake so find_package(Eigen3 CONFIG)
# finds it; Boost is the emscripten-ports header archive (the one
# `-sUSE_BOOST_HEADERS` would fetch), found by FindBoost; cereal is found by
# its header.
set -euo pipefail
PREFIX="${1:-$HOME/opt/wasm-deps}"
BOOST_VERSION="${BOOST_VERSION:-1.83.0}"
EIGEN_VERSION="${EIGEN_VERSION:-3.4.0}"
CEREAL_VERSION="${CEREAL_VERSION:-1.3.2}"

if [[ -f "$PREFIX/.done-boost$BOOST_VERSION-eigen$EIGEN_VERSION-cereal$CEREAL_VERSION" ]]; then
    exit 0
fi
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$PREFIX/include"

echo "==> Boost $BOOST_VERSION headers"
curl -fsSL -o "$WORK/boost.zip" \
    "https://github.com/emscripten-ports/boost/releases/download/boost-$BOOST_VERSION/boost-headers-$BOOST_VERSION.zip"
unzip -q "$WORK/boost.zip" -d "$WORK/boost"
rm -rf "$PREFIX/include/boost"
cp -R "$(dirname "$(find "$WORK/boost" -name version.hpp -path '*boost/version.hpp' | head -1)")" "$PREFIX/include/boost"

echo "==> Eigen $EIGEN_VERSION"
curl -fsSL "https://gitlab.com/libeigen/eigen/-/archive/$EIGEN_VERSION/eigen-$EIGEN_VERSION.tar.gz" | tar -xz -C "$WORK"
cmake -S "$WORK/eigen-$EIGEN_VERSION" -B "$WORK/eigen-build" -DCMAKE_INSTALL_PREFIX="$PREFIX" \
      -DBUILD_TESTING=OFF -DEIGEN_BUILD_DOC=OFF -DEIGEN_BUILD_PKGCONFIG=OFF >/dev/null
cmake --install "$WORK/eigen-build" >/dev/null

echo "==> cereal $CEREAL_VERSION"
curl -fsSL "https://github.com/USCiLab/cereal/archive/refs/tags/v$CEREAL_VERSION.tar.gz" | tar -xz -C "$WORK"
rm -rf "$PREFIX/include/cereal"
cp -R "$WORK/cereal-$CEREAL_VERSION/include/cereal" "$PREFIX/include/"

touch "$PREFIX/.done-boost$BOOST_VERSION-eigen$EIGEN_VERSION-cereal$CEREAL_VERSION"
echo "==> headers in $PREFIX"
