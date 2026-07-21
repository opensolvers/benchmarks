# gpu — PowerVR BXE-2-32 on the SpaceMiT K1: why GPU compute is closed, and the open-stack route (deferred)

This directory documents the **GPU side-quest** on the SpaceMiT K1 (Orange Pi RV2):
whether the on-die **Imagination PowerVR BXE-2-32** GPU can be used for compute, why
the vendor-shipped path is a dead end, and the one remaining open-source route — which
is **R&D, currently deferred (not switching the board's kernel for weeks)**.

Unlike the other directories here, this is **not a speed A/B**. It is a
**characterization / negative-result** writeup, kept because "we proved this door is
locked, and here is exactly which lock" is itself a reusable result for anyone else
staring at a K1 GPU.

> **Bottom line up front:** GPU compute on this board is **not reachable** with any
> buildable-by-us userspace today. The hardware is compute-capable, but every shipped
> API is gated to the wrong GPU variant, and the only self-buildable stack (open Mesa
> `pvr` + `drm/imagination`) needs a kernel we are deliberately **not** installing yet.
> Even on full success the ceiling is **~20 GFLOPS FP32 — roughly 50–100× *below* the
> 8× X60 RVV int8 GEMM** for our LLM workload. **The CPU/RVV path stays the target.**

---

## 1. The hardware vs the wall

| Layer | State | Verdict |
|---|---|---|
| **Silicon** (BXE-2-32, BVNC `36.29.52.182`) | IMG IP spec: OpenCL 3.0 + Vulkan 1.3; `/sys/kernel/debug/pvr/status` shows a live **CDM (Compute Data Master)** row | ✅ **compute-capable** |
| **Firmware** (`rgx.fw.36.29.52.182`, proprietary format) | ships in the SpaceMiT deb, loads fine | ✅ present |
| **Vendor userspace** (`libVK_IMG`, `libPVROCL`, `libGLESv2_PVR_MESA`) | **all three blobs embed only `"B-Series BXM-4-64"`** — none contain BXE | ❌ **gated to the wrong SKU** |

The wall is **100% userspace**. The entire SpaceMiT-shipped DDK (Vulkan + OpenCL +
GLES-compute) is compiled/allowlisted for the **BXM-4-64** SKU and rejects our
**BXE-2-32** at platform/instance creation — even though the matching firmware loads
and the hardware exposes a compute data master.

### Evidence (on-board, reproducible)

- **OpenCL:** `libPVROCL` gates on BVNC → refuses `36.29.52.182`.
- **Vulkan:** `vkCreateInstance` against the IMG ICD (`/etc/vulkan/icd.d/powervr_icd.json`
  → `libVK_IMG.so`, api 1.3.277) → **`-9 VK_ERROR_INCOMPATIBLE_DRIVER`**.
  `strace` is decisive and mirrors the OpenCL failure exactly: the driver opens
  `/dev/dri/renderD128`, `DRM_IOCTL_VERSION` succeeds, it runs the same PVR GEM-alloc
  probe batch (all succeed), reads `/etc/powervr.ini`, then **unmaps everything and
  rejects the device at instance creation**.
- **GLES 3.1 compute** (`glDispatchCompute` via `libGLESv2_PVR_MESA`): same DDK, same
  BXM gate → same outcome (untested end-to-end but provably gated by the blob strings).

**Converged root cause:** one BVNC allowlist mismatch across the *whole* DDK, not
missing hardware.

---

## 2. Can we rebuild / bypass / swap the vendor stack? — No (definitive closure)

| Attempt | Result |
|---|---|
| **Rebuild the DDK** with a BXE target | ❌ The CL/VK/GLES runtime + USC shader compiler are **proprietary IMG/NDA closed source**. The Gitee `img-gpu-powervr` repo is only a **buildroot packaging wrapper** around the same prebuilt blobs — not compilable source. |
| **Find a BXE-matched prebuilt** in any SpaceMiT/Bianbu archive | ❌ Every `img-gpu-powervr` package is the **same build `24.2@6603887`** (bb1..bb22 are Debian packaging revisions of one identical blob). The only other series (`23.2-6460340`) has a userspace↔kernel **bridge-ABI mismatch** against our built-in `24.2@6603887` `pvrsrvkm` → dead end without also swapping the kernel. No BXE usermode exists publicly. |
| **Talk to `pvrsrvkm` directly** | ❌ Means its closed, version-locked `PVRSRV_BRIDGE` ioctl ABI; submitting compute still needs BXE **USC codegen + CDM command-stream** construction — i.e. reimplementing the closed compiler. Not practical. |
| **Cross-vendor / supplier-direct BXE DDK** | ❌ No public source; no supplier-direct route. |

**→ No self-buildable path exists on the closed stack.** (DDK left installed; graphics +
GLES/VK are functional for *rendering*, just not for compute on our SKU. Reversible via
`dpkg -r img-gpu-powervr`.)

---

## 3. The one remaining route: the **open** Mesa `pvr` + `drm/imagination` stack

There are two mutually-exclusive stacks. You cannot mix them:

```
PROPRIETARY (what we have):  app → libVK_IMG / libPVROCL → libsrv_um → [pvrsrvkm builtin] → FW(rgx.fw.*)
OPEN (the only buildable one): app → Mesa libvulkan_powervr → [drm/imagination] → FW(powervr/rogue_*_v1.fw)
                                                               ^ different kernel uAPI, different DT bind,
                                                                 different firmware format — NOT interchangeable
```

Committing to the open stack means **removing the proprietary driver from the kernel**
for the boot you test on. That is the crux of why this is deferred: it requires a
**rebuilt kernel** and a **spare boot medium**, and we are explicitly **not switching
this board's kernel for the next weeks**.

### 3.1 ARM as the basis — yes, TI's AM62x is the reference

The open stack was **co-developed by Texas Instruments** for the ARM **AM62x** SoC
(also a Rogue GPU). That work is **architecture-independent source** we compile for
riscv64. Reusable as-is:

1. **Kernel driver** `drivers/gpu/drm/imagination/` (mainline ≥ 6.8) — same source builds
   for riscv64.
2. **Device-tree binding** — TI's `k3-am62-main.dtsi` `gpu@fd00000` node is the porting
   template (the single most reusable artifact).
3. **Open firmware packaging** — freedesktop `gitlab.freedesktop.org/imagination/linux-firmware`.
4. **Mesa build recipe** — meson flags + runtime env from `meta-ti` / Mesa CI.

**Not** reusable: TI's *proprietary* `ti-img-rogue-umlibs` (ARM-only binaries — the same
closed dead end).

### 3.2 Phase-0 feasibility gates — all researched, results below

| Gate | Result |
|---|---|
| **0a — open firmware for BVNC `36.29.52.182`?** | ✅ **Exists.** `powervr/rogue_36.29.52.182_v1.fw` on freedesktop `linux-firmware` branch `powervr` (added 2024-09-12 commit `965a9656` "add firmware for BXE-2-32", refreshed 2026-04-15). Driver builds the name deterministically via `pvr_build_firmware_filename()` → `powervr/rogue_36.29.52.182_v1.fw`. The proprietary `rgx.fw.*` is **not** reusable — the open driver validates a `PVR_FW_FLAGS_OPEN_SOURCE` header flag it lacks. |
| **0b — open driver supports riscv64 + our BVNC?** | ✅ / 1-line patch. `CONFIG_DRM_POWERVR` **`depends on (ARM64 \|\| RISCV && 64BIT)`** — riscv64 is first-class. Our BVNC isn't in `pvr_device.c`'s table, but a sibling BXE-2-32 (`36.52.104.182`) is already `PVR_GPU_EXPERIMENTAL`. Two ways in: boot param `powervr.exp_hw_support=1`, **or** a 1-line patch adding `case PVR_PACKED_BVNC(36, 29, 52, 182):` (preferred — deterministic). |
| **0c — a driver-capable, K1-booting kernel source?** | ⚠️ **This is the hard gate.** See §4. |
| **0d — spare boot media** | ⏳ **User action required** before any install/boot test. The build-only phase does not need it. |

Mesa's official driver docs list BXE-2-32 / `36.29.52.182` as supported (behind
`PVR_I_WANT_A_BROKEN_VULKAN_DRIVER=1`), so the userspace pieces exist too — but BXE-2-32
is "unsupported, no active maintainer," graphics-first (compute on B-Series is untested
in the open driver).

---

## 4. Why it is deferred: the kernel is the wall (Path A backport)

**No K1 kernel at any version ships the open driver.** Exhaustive July-2026 survey:

- **SpaceMiT** `spacemit-com/linux` (6.6.y … 6.18.y): all use `CONFIG_POWERVR_ROGUE` (proprietary).
- **Bianbu**: only maintained K1 branch is `linux-6.6` LTS, proprietary.
- **Armbian** spacemit (6.6 / 6.18 / edge 7.2): all boot the K1 but all `CONFIG_POWERVR_ROGUE`.
- **OpenWrt** PR #23231 (K1 on 6.18.26): explicitly chose `CONFIG_POWERVR_ROGUE`.
- **Mainline** (≥ 6.18): has the open `CONFIG_DRM_POWERVR` **and** riscv64 — but **no K1 GPU
  DT node upstream**, so it won't drive *our* GPU out of the box.

So "boot a newer kernel that already has the open driver working on K1" **does not exist
anywhere**. The only route is **Path A — backport the open driver onto the K1's
`6.6.63-ky` vendor kernel** (which keeps the working K1 BSP: display, clocks, PM,
peripherals).

### Backport scope (source-verified on board — smaller than feared)

The K1 tree = `github.com/orangepi-xunlong/linux-orangepi` branch `orange-pi-6.6-ky`
(Makefile `SUBLEVEL=63`). DRM-core dependency audit:

| Open-driver dependency | In `6.6.63-ky`? |
|---|---|
| `drm_exec.c` (`DRM_EXEC`) | ✅ present |
| `drm_gem_shmem_helper.c` | ✅ present |
| `scheduler/sched_main.c` (`DRM_SCHED`) | ✅ present |
| `drm_gpuvm.c` (`DRM_GPUVM`) | ❌ **absent — the only missing core file** |
| `drivers/gpu/drm/imagination/` | ❌ absent (the driver we add) |

**Build-only recipe (nothing installed):**

1. Add `drm_gpuvm.c` + `drm_gpuvm.h` from mainline (available ≥ v6.11/v6.12) + wire `CONFIG_DRM_GPUVM`.
2. Drop in `drivers/gpu/drm/imagination/` (~75 files, v6.18) — **already includes
   `pvr_fw_riscv.c` + `pvr_rogue_riscv.h`**, so riscv64 firmware-core support ships with
   the driver (our BXE-2-32 uses a RISC-V firmware processor); no separate riscv patch needed.
3. Wire `CONFIG_DRM_POWERVR=m` (`obj-$(CONFIG_DRM_POWERVR) += imagination/`).
4. 1-line BVNC patch in `pvr_device.c` (see 0b).
5. `CONFIG_POWERVR_ROGUE=n` (frees the GPU + avoids DT double-bind).
6. Write a **K1 GPU DT node** — port the TI AM62x node shape onto `imggpu@cac00000`
   (reg `0xcac00000`; K1 clock/irq/power refs), keeping the `img,*` compatible fallbacks
   (swap `axe` → `img,img-bxe-2-32`).
7. Place `powervr/rogue_36.29.52.182_v1.fw` in the rootfs/initramfs firmware path.

**TI AM62x template node** (`k3-am62-main.dtsi`) — the shape to port:

```dts
gpu: gpu@fd00000 {
    compatible = "ti,am62-gpu", "img,img-axe-1-16m", "img,img-axe", "img,img-rogue";
    reg = <0x00 0x0fd00000 0x00 0x20000>;
    clocks = <&k3_clks 187 0>;
    clock-names = "core";
    interrupts = <GIC_SPI 86 IRQ_TYPE_LEVEL_HIGH>;
    power-domains = <&k3_pds 187 TI_SCI_PD_EXCLUSIVE>;
    power-domain-names = "a";
};
```

### Difficulty: HIGH / uncertain

The backport must bring **DRM-core infra** (`drm_gpuvm`) across the 6.6 → 6.8+ gap and
**reconcile its API** against 6.6's already-present `drm_exec` / `drm_sched` /
`drm_gem_shmem_helper` — that reconciliation is the real risk. Plus a K1 GPU DT node
written from scratch (none upstream). Treat as R&D with a genuine chance of "does not
cleanly backport", **not** a scheduled task. Fallback to watch: mainline ≥ 6.20 gaining
a K1 GPU DT node (would enable a clean rebase later).

---

## 5. What would count as success (the PoC ladder)

Deferred; documented so it can be picked up cold. Each phase gates the next.

| PoC | Success criterion | Rough odds |
|---|---|---|
| **#1 — kernel access** (primary milestone) | `drm/imagination` binds the node (new `/dev/dri/renderD*`); a ~40-line C program does `drmGetVersion(fd)` → name **`"powervr"`** and `DRM_IOCTL_PVR_DEV_QUERY` → returns BVNC **`36.29.52.182`**. Public open uAPI only (`include/uapi/drm/pvr_drm.h`) — no NDA, no RE. | ~40–50% |
| **#2 — userspace enumeration** | Mesa built with `-Dvulkan-drivers=imagination-experimental`; `PVR_I_WANT_A_BROKEN_VULKAN_DRIVER=1 vulkaninfo --summary` lists a PowerVR/Rogue device with a **compute-capable queue family** (rendering may be broken — irrelevant). | ~25–35% |
| **#3 — trivial compute dispatch** (future) | Smallest `vkCmdDispatch`: increment a small SSBO, readback, verify one value. **Correctness only.** | < 15% |

---

## 6. Current status (this board)

- **GPU compute via vendor stack: CLOSED** (definitive — §1, §2). CPU/RVV is the path.
- **Open-stack Phase 0: complete.** Gates 0a (firmware) and 0b (driver support) are GREEN;
  0c resolved to "Path A backport is the only route"; 0d (spare media) pending.
- **Kernel rebuild: staged build-only, then paused.** A baseline `make Image modules dtbs`
  of the unmodified `6.6.63-ky` tree was built on-board to prove toolchain/config sanity
  **before** layering the driver. **Nothing is installed. The board's kernel is not being
  switched for weeks** (explicit constraint). The backport (§4) is the next build-only step
  whenever the GPU side-quest is resumed.
- **The useful accelerator on the K1 is not the GPU** — it's the CPU's **X60 IME** int8
  path (see [`../ime/`](../ime)) and RVV. Even a fully working open GPU stack tops out
  ~50–100× below that for our int8 LLM GEMM. This directory exists to record *why*, so the
  door isn't re-opened without reason.

## References

- Open firmware: `gitlab.freedesktop.org/imagination/linux-firmware` branch `powervr`
- Open kernel driver: `drivers/gpu/drm/imagination/` (mainline ≥ 6.8; riscv64 fw core ~6.16–6.18)
- Mesa `pvr`: docs.mesa3d.org/drivers/powervr.html
- K1 kernel source: `github.com/orangepi-xunlong/linux-orangepi` branch `orange-pi-6.6-ky`
- DT template: TI `k3-am62-main.dtsi` `gpu@fd00000`; binding `img,powervr-rogue.yaml`
