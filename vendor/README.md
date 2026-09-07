# Pinned build sources

These archives contain upstream Ramulator2, argparse, spdlog and yaml-cpp at the exact commits in Fugue-asplos-dependencies.json. They include original licenses. The build applies this repository's original pim_ramulator_src overlays and patches. CMake runs with FETCHCONTENT_FULLY_DISCONNECTED=ON; no sibling checkout, submodule, network fetch or prebuilt binary is used.

Upstreams: https://github.com/CMU-SAFARI/ramulator2, https://github.com/p-ranav/argparse, https://github.com/gabime/spdlog, https://github.com/jbeder/yaml-cpp.
