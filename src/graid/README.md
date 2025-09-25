# BaM — GRAID Driver Integration

BaM is a GPU-centric storage library that lets GPU applications issue NVMe I/O directly from GPU code (no CPU involvement). This driver contains an integration of **GRAID** — a GPU-accelerated NVMe RAID solution — which exposes a unified, protected RAID volume to BaM via a lightweight **GPU-Initiated I/O Queue (GIIOQ)** interface.

---

## Overview

* **GPU-driven NVMe I/O:** GPU threads submit storage I/O directly using BaM’s mechanisms, bypassing the CPU entirely.
* **GRAID RAID Volume:** Multiple NVMe devices are combined into a single RAID5 volume providing data protection and aggregated performance.
* **GIIOQ:** A lightweight GPU-Initiated I/O Queue designed for efficiency and future RDMA-friendly remote-GPU access.
* **High throughput:** On a simulated 32-device RAID5, BaM + GRAID achieved ~100M IOPS using a single NVIDIA H100 (GRAID used <10% of GPU resources).

> **Note:** Currently, only the `nvm-block-bench` benchmark has been modified to exercise GRAID’s RAID volume support.

---

## Benchmark (example)

This command was used to reproduce the ~100M IOPS result. Run it from the repository root with a `build` directory:

```bash
t=$(( 8 << 20 ))
sudo ./build/bin/nvm-block-bench --ssd 2 -g 0 \
  --page_size=4096 --threads=${t} --pages=${t} \
  --num_blks=$((1 << (40-12))) --random true \
  --reqs=1 --num_queues=1
```

* `--ssd` specifies the **SSD type**. `2` corresponds to the **GRAID RAID volume** in this setup.
* `-g 0` selects the CUDA GPU device ID.
* This benchmark invokes a very large number of GPU threads/pages to drive heavy I/O load against the RAID volume.

[![asciicast](https://asciinema.org/a/CzjhTGd2IhsBEzuNMsUfOMSAi.svg)](https://asciinema.org/a/CzjhTGd2IhsBEzuNMsUfOMSAi)

---

## Building

The project uses CMake and the build steps are the same as the original BaM project. The updated CMake configuration will detect an installed GRAID Linux DKMS driver under `/usr/src` and compile BaM with GRAID support automatically.

Basic build steps:

```bash
git submodule update --init --recursive
mkdir -p build && cd build
cmake ..
make libnvm        # builds BaM GPU library and user components
make benchmarks    # builds nvm-block-bench and other benchmarks
```

If you build kernel/user modules from `build/module`, follow the same steps as the upstream BaM project to build and install modules.

---

## GRAID Driver Integration

* The GRAID Linux driver exports a character device: `/dev/graid`.
* The BaM GRAID driver communicates **directly** with `/dev/graid`.
* **Important:** The GRAID BaM driver **does not use** BaM’s `libnvm` kernel module. The integration path uses the GRAID character device rather than the standard BaM kernel driver.

This design keeps the GRAID integration self-contained: GRAID owns RAID logic and device access, while BaM uses the GIIOQ interface to issue I/O to the RAID volume.

---

## Usage

* Use `sudo` where device access is required (e.g., running the benchmark or applications accessing `/dev/graid`).
* From the GPU/application perspective, the GRAID RAID volume behaves like a block device accessible via the BaM/GIIOQ path. Application-level code should continue to use BaM APIs for memory registration and I/O submission.

---

## Development Status & TODO

Current status: Prototype integration with working GIIOQ path and benchmark support.

Planned next work items:

1. Further optimize GRAID GPU kernels to reduce GPU usage.
2. Run benchmarks on real NVMe hardware (current results are from simulated devices).
3. Implement an IPC buffer allocation interface so applications can allocate GPU memory via the GRAID driver.
4. Implement full RAID5/6 write support (current focus is read-heavy/benchmark paths).
5. Support fine-grained 512-bytes I/O. Currently, GRAID RAID volume only supports 4K LBA.
