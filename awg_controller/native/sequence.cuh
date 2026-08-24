/* MEMORY mode: precompute the round, upload it, let the card replay it.
 *
 * Why this mode exists: FIFO streaming needs sample_rate*4 B/s sustained over
 * PCIe forever, and an M4i's link is Gen2 x8 (~3.4 GB/s practical), so the
 * card cannot be fed at its own top speed. Uploading once and replaying from
 * the card's DRAM removes the sustained-rate requirement entirely, so MEMORY
 * mode runs at the full 1.25 GS/s. The cost is a bounded round length.
 *
 * Two sequence segments do all the work:
 *
 *   step 0 -> segment 0 (the round), played once, then falls through to
 *   step 1 -> segment 1 (the park), whose next-step pointer is itself, so the
 *             card loops it forever with no host involvement.
 *
 * The park segment must contain a whole number of cycles of every tone or
 * each wrap injects a phase step; awg_schedule_hold_tail() guarantees that.
 */

#ifndef AWG_ENGINE_SEQUENCE_CUH
#define AWG_ENGINE_SEQUENCE_CUH

#include "render.cuh"

#include "../../scapp/c_header/dlltyp.h"
#include "../../scapp/c_header/regs.h"
#include "../../scapp/c_header/spcerr.h"
#include "../../scapp/c_header/spcm_drv.h"

enum {
    AWG_SEQ_SEG_ROUND = 0,
    AWG_SEQ_SEG_HOLD = 1,
    AWG_SEQ_N_SEGMENTS = 2,
    /* Segment lengths must be a multiple of the card's memory granularity.
     * 1024 clears the M4i requirement (32) and its minimum segment size with
     * room to spare; the padding renders as more of the final hold, so it is
     * waveform-correct rather than merely harmless. */
    AWG_SEQ_ALIGN = 1024,
};

static inline int64_t awg_seq_align_up(int64_t n) {
    return ((n + AWG_SEQ_ALIGN - 1) / AWG_SEQ_ALIGN) * AWG_SEQ_ALIGN;
}

/* Sequence step word: low 32 bits are segment + next<<16, high 32 are the
 * loop count OR'd with the step flags. */
static inline int64 awg_seq_step(int32_t segment, int32_t next, int32_t loops,
                                 uint32_t flags) {
    const uint32_t lo = ((uint32_t)next << 16) | (uint32_t)segment;
    const uint32_t hi = ((uint32_t)loops & SPCSEQ_LOOPMASK) | flags;
    return (int64)(((uint64_t)hi << 32) | lo);
}

/* Render `frames` of `sched` into the RDMA staging buffer and hand it to the
 * card as sequence segment `index`. Blocking: one shot, then done. */
static inline uint32_t awg_seq_upload(drv_handle hCard, void* staging,
                                      const AwgDeviceSchedule* sched, int64_t abs_start,
                                      int32_t index, int64_t frames, float max_value,
                                      cudaStream_t stream, cudaError_t* cerr) {
    *cerr = awg_render_span(sched, abs_start, frames, max_value, (int16_t*)staging, stream);
    if (*cerr == cudaSuccess) {
        *cerr = cudaStreamSynchronize(stream);
    }
    if (*cerr != cudaSuccess) {
        return ERR_OK; /* caller checks *cerr first */
    }

    uint32_t e = spcm_dwSetParam_i32(hCard, SPC_SEQMODE_WRITESEGMENT, index);
    if (e == ERR_OK) {
        e = spcm_dwSetParam_i32(hCard, SPC_SEQMODE_SEGMENTSIZE, (int32)frames);
    }
    if (e == ERR_OK) {
        /* notify size 0: transfer the whole segment in one go */
        e = spcm_dwDefTransfer_i64(hCard, SPCM_BUF_DATA, SPCM_DIR_GPUTOCARD, 0, staging, 0,
                                   (uint64)(frames * 4));
    }
    if (e == ERR_OK) {
        e = spcm_dwSetParam_i32(hCard, SPC_M2CMD, M2CMD_DATA_STARTDMA | M2CMD_DATA_WAITDMA);
    }
    return e;
}

/* Wire step 0 -> step 1 -> step 1. Call after both segments are uploaded. */
static inline uint32_t awg_seq_write_steps(drv_handle hCard) {
    uint32_t e = spcm_dwSetParam_i64(
        hCard, SPC_SEQMODE_STEPMEM0,
        awg_seq_step(AWG_SEQ_SEG_ROUND, 1, 1, SPCSEQ_ENDLOOPALWAYS));
    if (e == ERR_OK) {
        /* next == own index, so the card parks here until CARD_STOP */
        e = spcm_dwSetParam_i64(
            hCard, SPC_SEQMODE_STEPMEM0 + 1,
            awg_seq_step(AWG_SEQ_SEG_HOLD, 1, 1, SPCSEQ_ENDLOOPALWAYS));
    }
    return e;
}

#endif /* AWG_ENGINE_SEQUENCE_CUH */
