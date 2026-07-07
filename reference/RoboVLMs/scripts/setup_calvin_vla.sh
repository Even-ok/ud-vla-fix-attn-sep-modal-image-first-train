# 1. 自动将现有的 deb 源复制并转换为 deb-src 源，追加到 sources.list 中
# （如果你的环境是较老的 Ubuntu/Debian，使用标准 sources.list）
if [ -f /etc/apt/sources.list ]; then
    grep '^deb ' /etc/apt/sources.list | sed 's/^deb /deb-src /' >> /etc/apt/sources.list
fi

# （如果你的环境是最新的 Ubuntu 24.04+，它使用 deb822 格式）
if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then
    sed -i 's/Types: deb/Types: deb deb-src/g' /etc/apt/sources.list.d/ubuntu.sources
fi
apt-get update -yqq

# Install dependency for calvin
apt-get -yqq install libegl1 libgl1

# Install EGL mesa
apt-get install -yqq libegl1-mesa libegl1-mesa-dev
apt-get install -yqq mesa-utils libosmesa6-dev llvm
apt-get -yqq install meson
apt-get -yqq build-dep mesa
