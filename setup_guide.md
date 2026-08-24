## 1. Driver (OS-dependent)

Install the proprietary NVIDIA driver matching the GPU and the CUDA/cupy version you intend to use

```bash

nvidia-smi

```

Building the Spectrum kernel module (`spcm4`) with GPUDirect support requires the matching NVIDIA driver source tree with its `Module.symvers` already generated, since `spcm4`'s `spcm_cuda.o` links directly against the driver's exported `nvidia_p2p_get_pages` / `nvidia_p2p_dma_map_pages` / etc. symbols

`nvidia-peermem` is not required or used by SCAPP

Then in `spcm4-*/m4i_krnl_linux/Makefile`, uncomment and point:

```makefile

NVIDIA_DRV_SRC := /usr/src/nvidia-<version>/

```

For Debian use the following:

```
# Debian 13 proprietary NVIDIA driver
NVIDIA_DRV_SRC := $(firstword $(wildcard /usr/src/nvidia-current-*))
NVIDIA_SYMVERS := $(firstword $(wildcard /var/lib/dkms/nvidia-current/*/$(shell uname -r)/x86_64/module/Module.symvers))

...
    ifneq ($(UNAME_P),aarch64)
        # if Module.symvers does not exist then user has to build the driver module to generate it
        ifeq (,$(wildcard $(NVIDIA_SYMVERS)))
		$(error ERROR: Module.symvers not found at $(NVIDIA_SYMVERS))
	endif

	KBUILD_EXTRA_SYMBOLS := $(NVIDIA_SYMVERS)
...
```

and rebuild/install via `spcm4-*/make_spcm4_linux_kerneldrv.sh` Confirm the symbols actually resolved:

```bash

nm /lib/modules/$(uname -r)/kernel/drivers/spcm4.ko | grep nvidia_p2p

```

## 2. IOMMU passthrough

If `dmesg` shows, while running:

```

AMD-Vi: Event Logged [IO_PAGE_FAULT domain=0x000f ...]

```

This means the IOMMU is intercepting the Spectrum card's P2P DMA write into GPU memory and rejecting it because that address isn't mapped in the card's strict per-device IOMMU domain

Fix (`/etc/default/grub`, append the existing `GRUB_CMDLINE_LINUX_DEFAULT` value):

```

GRUB_CMDLINE_LINUX_DEFAULT="quiet amd_iommu=on iommu=pt"

```

```bash

sudo update-grub # or grub2-mkconfig -o /boot/grub2/grub.cfg on RHEL-based

sudo reboot

```

Verify after reboot:

```bash

cat /proc/cmdline # confirm amd_iommu=on iommu=pt present

```

Then re-run a SCAPP script with `dmesg -w` in parallel and confirm no more `IO_PAGE_FAULT` events appear.
