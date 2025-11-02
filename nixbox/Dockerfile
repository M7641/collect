FROM nixos/nix

RUN nix-channel --update

# RUN nix-env -iA nixpkgs.gbm \
#     nixpkgs.hyprland-protocols \
#     nixpkgs.hyprlang \
#     nixpkgs.hyprutils \
#     nixpkgs.hyprwayland-scanner \
#     nixpkgs.libdrm \
#     nixpkgs.libpipewire-0.3 \
#     nixpkgs.libspa-0.2 \
#     nixpkgs.sdbus-cpp \
#     nixpkgs.wayland-client \
#     nixpkgs.wayland-protocols

# RUN git clone --recursive https://github.com/hyprwm/xdg-desktop-portal-hyprland \
#     &&  cd xdg-desktop-portal-hyprland/ \
#     && cmake -DCMAKE_INSTALL_LIBEXECDIR=/usr/lib -DCMAKE_INSTALL_PREFIX=/usr -B build \
#     && cmake --build build \
#     && sudo cmake --install build
