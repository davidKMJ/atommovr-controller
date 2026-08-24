/* Card/link diagnostic. Opens the card, prints what it negotiated, and works
 * out the streaming ceiling. No GPU, no render, no output -- safe to run with
 * an amplifier connected.
 *
 *   make card-info                 (defaults to /dev/spcm0)
 *   ./tests/card_info /dev/spcm1
 *
 * The question this answers: when the pump reports most of its wall time
 * blocked in WAITDMA, is the card's own PCIe link even capable of the rate we
 * are asking for? A FIFO replay needs
 *
 *     sample_rate * n_channels * 2 bytes
 *
 * sustained across the link, forever. If that exceeds the practical link
 * bandwidth printed below, no amount of ring/notify tuning helps -- the only
 * fixes are a lower sample rate, fewer channels, or replaying from the card's
 * own on-board memory instead of streaming.
 */

#include "../../../scapp/c_header/dlltyp.h"
#include "../../../scapp/c_header/regs.h"
#include "../../../scapp/c_header/spcerr.h"
#include "../../../scapp/c_header/spcm_drv.h"

#include <stdio.h>

static int32 q(drv_handle h, int32 reg) {
    int32 v = -1;
    spcm_dwGetParam_i32(h, reg, &v);
    return v;
}

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "/dev/spcm0";

    drv_handle hCard = spcm_hOpen((char*)path);
    if (hCard == NULL) {
        fprintf(stderr, "no card at %s\n", path);
        return 1;
    }

    const int32 gen = q(hCard, SPC_PCIEXPGENERATION);
    const int32 lanes = q(hCard, SPC_PCIEXPLANES);
    const int32 payload = q(hCard, SPC_PCIEXPPAYLOAD);
    const int32 readreq = q(hCard, SPC_PCIEXPREADREQUESTSIZE);
    const int32 n_ch = q(hCard, SPC_MIINST_MODULES) * q(hCard, SPC_MIINST_CHPERMODULE);

    printf("card           : type %d, %d channels\n", q(hCard, SPC_PCITYP), n_ch);
    printf("bus slot       : %02d:%02d.%d\n", q(hCard, SPC_PCIHWBUSNO),
           q(hCard, SPC_PCIHWDEVNO), q(hCard, SPC_PCIHWFNCNO));
    printf("on-board mem   : %d MB\n", (int)(q(hCard, SPC_PCIMEMSIZE) / (1024 * 1024)));
    printf("max samplerate : %.1f MS/s\n", q(hCard, SPC_PCISAMPLERATE) / 1e6);
    printf("\n");
    printf("PCIe link      : Gen%d x%d\n", gen, lanes);
    printf("max payload    : %d bytes\n", payload);
    printf("max read req   : %d bytes   <- card reads the DMA buffer; small = slow\n", readreq);

    /* Per-lane raw rate. Gen1/2 are 8b/10b (2.5/5.0 GT/s -> 250/500 MB/s);
     * Gen3+ is 128b/130b (8.0 GT/s -> ~985 MB/s). The 0.85 factor is a
     * rule-of-thumb allowance for TLP headers and flow control -- real
     * sustained DMA lands near it, not at the raw number. */
    double per_lane_mb = 0.0;
    if (gen == 1) per_lane_mb = 250.0;
    else if (gen == 2) per_lane_mb = 500.0;
    else if (gen == 3) per_lane_mb = 985.0;
    else if (gen >= 4) per_lane_mb = 1969.0;

    if (per_lane_mb > 0.0 && lanes > 0) {
        const double raw = per_lane_mb * lanes / 1000.0;   /* GB/s */
        const double practical = raw * 0.85;
        printf("\nlink bandwidth : %.2f GB/s raw, ~%.2f GB/s practical\n", raw, practical);
        printf("\n  max FIFO sample rate at this link:\n");
        for (int ch = 1; ch <= (n_ch > 2 ? n_ch : 2); ch <<= 1) {
            printf("    %d channel%s : %6.0f MS/s\n", ch, ch > 1 ? "s" : " ",
                   practical * 1e9 / (ch * 2) / 1e6);
        }
        printf("\n  (2 bytes/sample/channel; exceeding this underruns no matter\n"
               "   how the ring or notify_samples are tuned)\n");
    }

    spcm_vClose(hCard);
    return 0;
}
