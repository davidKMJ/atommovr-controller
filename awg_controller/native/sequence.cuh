/* MEMORY mode: precompute the round, upload it, let the card replay it. */

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
    AWG_SEQ_ALIGN = 1024,
};

static inline int64_t awg_seq_align_up(int64_t n) {
    return ((n + AWG_SEQ_ALIGN - 1) / AWG_SEQ_ALIGN) * AWG_SEQ_ALIGN;
}

static inline int64 awg_seq_step(int32_t segment, int32_t next, int32_t loops,
                                 uint32_t flags) {
    const uint32_t lo = ((uint32_t)next << 16) | (uint32_t)segment;
    const uint32_t hi = ((uint32_t)loops & SPCSEQ_LOOPMASK) | flags;
    return (int64)(((uint64_t)hi << 32) | lo);
}

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
