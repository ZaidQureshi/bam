gpudirect-nvme
===============================================================================
This codebase builds on top of a opensource codebase by Jonas Markussen
available [here](https://github.com/enfiskutensykkel/ssd-gpu-dma).
We take his codebase and make it more robust by adding more error checking
and fixing issues of memory alignment along with increasing the performance when large number of requests are available.

We add to the codebase functionality allowing any GPU thread independently
access any location on the NVMe device. To facilitate this we develop high-throughput
concurrent queues.

Furthermore, we add the support for an application to use multiple NVMe SSDs.

Finally, to lessen the programmer's burden we develop abstractions, like an array abstraction
and a data caching layer,
so that the programmer writes their GPU code like they are trained to and the library automatically checks
if accesses hit in the cache or not and if they miss to automatically fetch the needed data from the NVMe device.
All of these features are developed into a header-only library in the [`include`](./include/) directory.
These headers can be used in Cuda C/C++ application code.




Hardware/System Requirements
-------------------------------------------------------------------------------
This code base requires specific type of hardware and specific system configuration to be functional and performant.

### Hardware Requirements ###
* A x86 system supporting PCIe P2P
* A NVMe SSD. Any NVMe SSD will do.
  * Please make sure there isn't any needed data on this SSD  as the system can write data to the SSD if the application requests to.
* A NVIDIA Tesla grade GPU that is from the Volta or newer generation. A Tesla V100 fits both of these requirements
  * A Tesla grade GPU is needed as it can expose all of its memory for P2P accesses over PCIe. (NVIDIA Tesla T4 does not work as it only provides 256M of BAR space)
  * A Volta or newer generation of GPU is needed as we rely on memory synchronization primitives only supported since Volta.
* A system that can support `Above 4G Decoding` for PCIe devices.
  * This is needed to address more than 4GB of memory for PCIe devices, specifically GPU memory.
  * This is a feature that might need to be ENABLED in the BIOS of the system.

### System Configurations ###
* As mentioned above, `Above 4G Decoding` needs to be ENABLED in the BIOS
* The system's IOMMU should be disabled for ease of debugging.
  * In Intel Systems, this requires disabling `Vt-d` in the BIOS
  * In AMD Systems, this requires disableing `IOMMU` in the BIOS
* The `iommu` support in Linux must be disabled too, which can be checked and disabled following the instructions [below](#disable-iommu-in-linux).
* In the system's BIOS, `ACS` must be disabled if the option is available
* Relatively new Linux kernel (ie. 5.x).
* CMake 3.10 or newer and the _FindCUDA_ package for CMake
* GCC version 5.4.0 or newer. Compiler must support C++11 and POSIX threads.
* CUDA 10.2 or newer
* Nvidia driver (at least 440.33 or newer)
* Kernel module symbols and headers for the Nvidia driver. The instructions for how to compile these symbols are given [below](#compiling-nvidia-driver-kernel-symbols).

### Disable IOMMU in Linux ###
If you are using CUDA or implementing support for your own custom devices, 
you need to explicitly disable IOMMU as IOMMU support for peer-to-peer on 
Linux is a bit flaky at the moment. If you are not relying on peer-to-peer,
we would in fact recommend you leaving the IOMMU _on_ for protecting memory 
from rogue writes.

To check if the IOMMU is on, you can do the following:

```
$ cat /proc/cmdline | grep iommu
```

If either `iommu=on` or `intel_iommu=on` is found by `grep`, the IOMMU
is enabled.

You can disable it by removing `iommu=on` and `intel_iommu=on` from the 
`CMDLINE` variable in `/etc/default/grub` and then reconfiguring GRUB.
The next time you reboot, the IOMMU will be disabled.

### Compiling Nvidia Driver Kernel Symbols ###
Typically the Nvidia driver kernel sources are installed in the `/usr/src/` directory.
So if the Nvidia driver version is `450.51.06`, then they will be in the `/usr/src/nvidia-450.51.06` directory.
So assuming the driver version is `450.51.06`, to get the kernel symbols you need to do the following commands as the `root` user.

```
$ cd /usr/src/nvidia-450.51.06
$ sudo make
```

Building the Project
-------------------------------------------------------------------------------
From the project root directory, do the following:

```
$ git submodule update --init --recursive
$ mkdir -p build; cd build
$ cmake ..
$ make libnvm                         # builds library
$ make benchmarks                     # builds benchmark program
```

The CMake configuration is _supposed to_ autodetect the location of CUDA, 
Nvidia driver and project library. CUDA is located by the _FindCUDA_ package for
CMake, while the location of both the Nvidia driver can be manually
set by overriding the `NVIDIA` defines for CMake 
(`cmake .. -DNVIDIA=/usr/src/...`).

After this, you should also compile the custom `libnvm` kernel module for NVMe devices.
Assuming that you are still standing in the build
directory, do the following in the `build` directory:

```
$ cd module
$ make
```

Loading/Unloading the Kernel Module
-------------------------------------------------------------------------------
In order to be able to use the custom kernel module for the NVMe device, we need to first unbind
the NVMe device from the default Linux NVMe driver.
To do this, we need to find the PCI ID of the NVMe device.
To find this we can use the kernel log. For example, if the required NVMe device want is mapped to the `/dev/nvme0` block device, we can do the following to find the PCI ID.

```
$ dmesg | grep nvme0
[  126.497670] nvme nvme0: pci function 0000:65:00.0
[  126.715023] nvme nvme0: 40/0/0 default/read/poll queues
[  126.720783]  nvme0n1: p1
[  190.369341] EXT4-fs (nvme0n1p1): mounted filesystem with ordered data mode. Opts: (null)

```
The first line gives the PCI ID for the `/dev/nvme0` device as `0000:65:00.0`.

To unbind the NVMe driver for this device we need to do the following as the `root` user:

```
# echo -n "0000:65:00.0" > /sys/bus/pci/devices/0000\:65\:00.0/driver/unbind
```

Now we can load the custom kernel module from the `build` directory with the following:

```
$ cd module
$ sudo make load
```

This should create a `/dev/libnvm0` device file for the controller.

The module can be unloaded from the `build` directory with the following:

```
$ cd module
$ sudo make unload
```

Running the Example Benchmark
-------------------------------------------------------------------------------
The example benchmark application tests the random access read bandwidth from the GPU threads to the NVMe device.
It assumes an array of `uint64_t` on the backing NVMe device and makes random or sequential accesses to the array.
The application must be run with `sudo` as it needs direct access to the `/dev/libnvm0` file.
The applicaiton will exist as the `./bin/nvm-array-bench` binary after compilation.
The application arguments are as follows:

``` 
$ ./bin/nvm-array-bench --help
OPTION            TYPE            DEFAULT   DESCRIPTION                         
  page_size       count           4096      size of page in cache               
  blk_size        count           64        CUDA thread block size              
  queue_depth     count           16        queue depth per queue               
  num_elems       count           2147483648number of 64-bit elements in backing array
  gpu             number          0         specify CUDA device                 
  reqs            count           1         number of reqs per thread           
  pages           count           1024      number of pages in cache            
  num_queues      count           1         number of queues per controller     
  random          bool            true      if true the random access benchmark runs, if false the sequential access benchmark runs
  threads         count           1024      number of CUDA threads
```

The application prints many things during initalization as it helps in debugging, however near the end it prints some
statistics of the GPU kernel, as shown below:

```
Elapsed Time: 200.576	Number of Read Ops: 1024	Data Size (bytes): 8192
Read Ops/sec: 5.1053e+06	Effective Bandwidth(GB/S): 0.0380374
```

If I want to run a large GPU kernel on GPU 0 with many threads (262144 threads) each making 1 random request to an array with 4-billion 64-bit integers, a page cache with 262144 pages each with size 4096 bytes, 128 NVMe queues each 1024 elements deep, I would run the following command:

```
sudo ./bin/nvm-array-bench --threads=262144 --blk_size=1024 --reqs=1 --pages=262144 --queue_depth=1024 --num_queues=128 --page_size=4096 --gpu=0 --num_elems=4294967296
```

If I want to run the same benchmark but now with each thread accessing the array sequentially, I would run the following command:

```
sudo ./bin/nvm-array-bench --threads=262144 --blk_size=1024 --reqs=1 --pages=262144 --queue_depth=1024 --num_queues=128 --page_size=4096 --gpu=0 --num_elems=4294967296 --random=false
```

Disclaimer: The NVMe SSD I was using supports 128 queues each with 1024 depth. However, even if your SSD supports less number of queues and/or less depth the system will automatically use the numbers reported by your device.
