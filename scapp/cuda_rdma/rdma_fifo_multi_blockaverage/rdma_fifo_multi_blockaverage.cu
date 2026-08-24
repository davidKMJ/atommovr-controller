/*
**************************************************************************

rdma_fifo_kernel.cu                             (c) Spectrum GmbH , 6/2017

**************************************************************************

Example for all M4i/M4x/M2p analog acquisition cards.

Data is transfered in FIFO Multi mode from card to GPU. A CUDA kernel is used
to add a number of segments together to reduce noise floor (block averaging).

Feel free to use this source for own projects and modify it in any kind.

Documentation for the API as well as a detailed description of the hardware
can be found in the manual for each device which can be found on our website:
www.spectrum-instrumentation.com/en/downloads

Further information can be found online in the Knowledge Base:
www.spectrum-instrumentation.com/en/knowledge-base-overview

**************************************************************************
*/


// ----- include standard driver header from library -----
#include "../../c_header/dlltyp.h"
#include "../../c_header/regs.h"
#include "../../c_header/spcerr.h"
#include "../../c_header/spcm_drv.h"

// ----- standard c include files -----
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../common/spcm_cuda_common.h"

// ----- CUDA include -----
#   include <cuda_runtime.h>

// CUDA-C includes
#   include <cuda.h>


// ----- CUDA kernels -----

// ############################################################################
// ----- 8bit averaging kernel -----
// -- pcIn:         values as acquired by the card
// -- plOut:        sum of samples at same position over multiple blocks
__global__ void CudaKernelBlockAverage (const int8* pcIn, int32* plOut)
    {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    plOut[i] += pcIn[i];
    }

// ----- 12/14/16bit averaging kernel -----
// -- pnIn:         values as acquired by the card
// -- plOut:        sum of samples at same position over multiple blocks
__global__ void CudaKernelBlockAverage (const int16* pnIn, int32* plOut)
    {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    plOut[i] += pnIn[i];
    }


// ############################################################################
// ----- kernels for zero noise suppression / noise suppressed accumulation (NSA) -----
// only use sampled value if it is above threshold, or use use-defined baseline value otherwise

// ----- 8 bit averaging kernel with zero noise suppression -----
// -- pcIn:         values as acquired by the card
// -- plOut:        sum of samples at same position over multiple blocks
// -- cThreshold:   if sample is above this level it will be added to sum, else the baseline value will be used. In LSB, not Volts!
// -- cBaseline:    see cThreshold. In LSB, not Volts!
__global__ void CudaKernelBlockAverageNoiseSuppression (const int8* pcIn, int32* plOut, int8 cThreshold, int8 cBaseline)
    {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (pcIn[i] >= cThreshold)
        plOut[i] += pcIn[i];
    else
        plOut[i] += cBaseline;
    }

// ----- 12/14/16 bit averaging kernel with zero noise suppression -----
// -- pnIn:         values as acquired by the card
// -- plOut:        sum of samples at same position over multiple blocks
// -- nThreshold:   if sample is above this level it will be added to sum, else the baseline value will be used. In LSB, not Volts!
// -- nBaseline:    see nThreshold. In LSB, not Volts!
__global__ void CudaKernelBlockAverageNoiseSuppression (const int16* pnIn, int32* plOut, int16 nThreshold, int16 nBaseline)
    {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (pnIn[i] >= nThreshold)
        plOut[i] += pnIn[i];
    else
        plOut[i] += nBaseline;
    }


// ############################################################################
// ----- kernel for demultiplexing averaged data -----
// -- plIn: averaged data (multiplexed)
// -- lNumCh: number of active channels
// -- plOutCh0: averaged samples of channel 0
// -- plOutCh0: averaged samples of channel 1 (can be NULL if lNumCh < 2)
// -- ...
// -- plOutCh7: averaged samples of channel 7 (can be NULL if lNumCh < 8)
__global__ void CudaKernelDemuxAveragedChannels (const int32* plIn, int lNumCh, int32* plOutCh0, int32* plOutCh1, int32* plOutCh2, int32* plOutCh3, int32* plOutCh4, int32* plOutCh5, int32* plOutCh6, int32* plOutCh7)
    {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int lChIdx = i % lNumCh;
    int lSample = i / lNumCh;
    switch (lChIdx)
        {
        // up to 4 channels with linear sorting (M4i)
        // up to 8 channels with linear sorting (M2p)
        case 0: plOutCh0[lSample] = plIn[i]; break;
        case 1: plOutCh1[lSample] = plIn[i]; break;
        case 2: plOutCh2[lSample] = plIn[i]; break;
        case 3: plOutCh3[lSample] = plIn[i]; break;
        case 4: plOutCh4[lSample] = plIn[i]; break;
        case 5: plOutCh5[lSample] = plIn[i]; break;
        case 6: plOutCh6[lSample] = plIn[i]; break;
        case 7: plOutCh7[lSample] = plIn[i]; break;
        }
    }

// ############################################################################



/*
**************************************************************************
main
**************************************************************************
*/

int main ()
    {
    drv_handle  hCard = 0;
    int32       lCardType, lSerialNumber, lFncType, lNumCh, lBytesPerSample;
    char        szErrorTextBuffer[ERRORTEXTLEN];
    uint32      dwError;
    int32       lStatus, lAvailUser, lPCPos;
    uint64      qwTotalMem = 0;
    uint64      qwToTransfer = GIGA_B(16);

    // settings for noise suppression kernels (in LSB, not in Volt)
    //int8  cThreshold = 64;
    //int8  cBaseline =  0;
    //int16 nThreshold = 1024;
    //int16 nBaseline =  0;

    int32 lNumSegmentsToAverage = 1024;



    // ------------------------------------------------------------------------
    // CARD SETUP

    // ----- open Spectrum card -----
    hCard = spcm_hOpen ((char*)"/dev/spcm0");
    if (!hCard)
        {
        printf ("no card found...\n");
        return 0;
        }

    // read type, function and sn and check for A/D card
    spcm_dwGetParam_i32 (hCard, SPC_PCITYP,                &lCardType);
    spcm_dwGetParam_i32 (hCard, SPC_PCISERIALNO,           &lSerialNumber);
    spcm_dwGetParam_i32 (hCard, SPC_FNCTYPE,               &lFncType);
    spcm_dwGetParam_i32 (hCard, SPC_MIINST_CHPERMODULE,    &lNumCh);
    spcm_dwGetParam_i32 (hCard, SPC_MIINST_BYTESPERSAMPLE, &lBytesPerSample);

    char szType[50];
    spcm_dwGetParam_ptr (hCard, SPC_PCITYP, szType, sizeof (szType));

    switch (lFncType)
        {
        case SPCM_TYPE_AI:
            {
            switch (lCardType & TYP_SERIESMASK)
                {
                case TYP_M4IEXPSERIES:
                case TYP_M4XEXPSERIES:
                case TYP_M2PEXPSERIES:
                case TYP_M5IEXPSERIES:
                    printf ("Found: %s sn %05d\n", szType, lSerialNumber);
                    break;
                default:
                    printf ("Card: %s sn %05d not supported by example\n", szType, lSerialNumber);
                    return EXIT_FAILURE;
                }
            break;
            }

        default:
            printf ("Card: %s sn %05d not supported by example\n", szType, lSerialNumber);
            return 0;
        }

    int32 lSegmentSize_samples = MEGA_B(1);                                       // Segments of 1 Megasample per channel
    int32 lSegmentPre_samples =  lSegmentSize_samples - KILO_B(1);                // 1kS pre trigger per segment
    int32 lNotifySize_bytes =    lSegmentSize_samples * lNumCh * lBytesPerSample; // DMA shall wait until a complete segment is available
    int32 lBufferSize_bytes =    4*lNotifySize_bytes;

    // do a simple FIFO Multi setup
    spcm_dwSetParam_i32 (hCard, SPC_CHENABLE,        (0x1 << lNumCh) - 1);   // enable all channels
    spcm_dwSetParam_i32 (hCard, SPC_CARDMODE,        SPC_REC_FIFO_MULTI);    // FIFO Multi mode
    spcm_dwSetParam_i32 (hCard, SPC_SEGMENTSIZE,     lSegmentSize_samples);
    spcm_dwSetParam_i32 (hCard, SPC_POSTTRIGGER,     lSegmentPre_samples);
    spcm_dwSetParam_i32 (hCard, SPC_TIMEOUT,         5000);                  // timeout 5 s
    spcm_dwSetParam_i32 (hCard, SPC_TRIG_ORMASK,     0);                     // disable software trigger
    spcm_dwSetParam_i32 (hCard, SPC_TRIG_ANDMASK,    0);
    spcm_dwSetParam_i32 (hCard, SPC_TRIG_CH_ORMASK0, SPC_TMASK0_CH0);        // enable channel trigger on channel 0
    spcm_dwSetParam_i32 (hCard, SPC_TRIG_CH0_MODE,   SPC_TM_POS);            // rising edge
    spcm_dwSetParam_i32 (hCard, SPC_TRIG_CH0_LEVEL0, 0);                     //
    spcm_dwSetParam_i32 (hCard, SPC_CLOCKMODE,       SPC_CM_INTPLL);         // clock mode internal PLL
    spcm_dwSetParam_i32 (hCard, SPC_CLOCKOUT,        0);                     // no clock output

    int lIR = 1000;                                                          // +/- 1 Volt
    spcm_dwSetParam_i32 (hCard, SPC_AMP0,           lIR);

    //spcm_dwSetParam_i64 (hCard, SPC_SPECIALCLOCK, 1);
    int64 llSamplerate = MEGA(125);
    spcm_dwSetParam_i64 (hCard, SPC_SAMPLERATE, llSamplerate);
    spcm_dwGetParam_i64 (hCard, SPC_SAMPLERATE, &llSamplerate);
    printf ("Samplerate: %lld\n", llSamplerate);
    // Card Setup finished
    // ------------------------------------------------------------------------


    // ----- DMA BUFFER SETUP -----
    // ----- get buffer on GPU that will be used as target for RDMA transfer -----
    int lCUDADeviceIdx = 0;         // index of used CUDA device
    void* pvDMABuffer_gpu = pvGetRDMABuffer (lCUDADeviceIdx, lBufferSize_bytes);
    if (pvDMABuffer_gpu == NULL)
        {
        spcm_vClose (hCard);
        return EXIT_FAILURE;
        }

    // setup DMA transfer from Spectrum card to GPU
    spcm_dwDefTransfer_i64 (hCard, SPCM_BUF_DATA, SPCM_DIR_CARDTOGPU, lNotifySize_bytes, pvDMABuffer_gpu, 0, lBufferSize_bytes);


    // ----- create a second buffer on the GPU that will be used as target for the kernel function -----
    int lAveragedSize_bytes = sizeof (int32) * lNotifySize_bytes / lBytesPerSample; // resulting samples will be 32bit wide
    void* pvBufferProcessed_gpu = 0;
    cudaError_t eCudaErr = cudaMalloc (&pvBufferProcessed_gpu, lAveragedSize_bytes);
    if (eCudaErr != cudaSuccess)
        {
        // cleanup
        printf ("ERROR in cudaMalloc(): %s\n", cudaGetErrorString(eCudaErr));

        spcm_vClose (hCard);

        cudaFree (pvDMABuffer_gpu);

        return EXIT_FAILURE;
        }
    // init buffer
    cudaMemset (pvBufferProcessed_gpu, 0, lAveragedSize_bytes);

    // ----- create buffers for each channel on GPU and host side. -----
    // ----- These will be used to demux the data and to transfer the processed data from the GPU -----
    int32* aplCh_host[8] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
    int32* aplCh_gpu[8]  = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
    for (int lChIdx = 0; lChIdx < lNumCh; ++lChIdx)
        {
        eCudaErr = cudaMallocHost (aplCh_host + lChIdx, lSegmentSize_samples * sizeof (int32));
        if (eCudaErr == cudaSuccess)
            eCudaErr = cudaMalloc (aplCh_gpu + lChIdx, lSegmentSize_samples * sizeof (int32));
        if (eCudaErr != cudaSuccess)
            {
            // cleanup
            printf ("ERROR in cudaMallocHost(): %s\n", cudaGetErrorString(eCudaErr));

            spcm_vClose (hCard);

            // free allocated CUDA buffers on GPU
            cudaFree (pvBufferProcessed_gpu);
            cudaFree (pvDMABuffer_gpu);
            for (int lChIdx2 = 0; lChIdx2 < lChIdx; ++lChIdx2)
                {
                cudaFreeHost (aplCh_host[lChIdx2]);
                cudaFree (aplCh_gpu[lChIdx2]);
                }

            return EXIT_FAILURE;
            }
        }

    // ----- start everything -----
    time_t startTime = time (NULL);
    dwError = spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER | M2CMD_DATA_STARTDMA);
    if (dwError != ERR_OK)
        {
        // cleanup
        spcm_dwGetErrorInfo_i32 (hCard, NULL, NULL, szErrorTextBuffer);
        printf ("%s\n", szErrorTextBuffer);

        spcm_vClose (hCard);

        // free allocated CUDA buffers on host and GPU
        for (int lChIdx = 0; lChIdx < lNumCh; ++lChIdx)
            {
            cudaFreeHost (aplCh_host[lChIdx]);
            cudaFree (aplCh_gpu[lChIdx]);
            }
        cudaFree (pvBufferProcessed_gpu);
        cudaFree (pvDMABuffer_gpu);

        return EXIT_FAILURE;
        }


    // run the FIFO mode and loop through the data
    // the control of the DMA transfer is the same as without RDMA.
    // The difference is that the driver reports a certain amount of data as available to the user,
    // but the data has been transfered into the GPU memory.
    int lAveragedSegments = 0;
    while (qwTotalMem < qwToTransfer)
        {
        if ((dwError = spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_DATA_WAITDMA)) != ERR_OK)
            {
            if (dwError == ERR_TIMEOUT)
                printf ("\n... Timeout\n");
            else
                printf ("\n... Error: %d\n", dwError);
            break;
            }

        else
            {
            spcm_dwGetParam_i32 (hCard, SPC_M2STATUS,             &lStatus);
            spcm_dwGetParam_i32 (hCard, SPC_DATA_AVAIL_USER_LEN,  &lAvailUser);
            spcm_dwGetParam_i32 (hCard, SPC_DATA_AVAIL_USER_POS,  &lPCPos);

            // ----- we selected the notify size so that it contains exactly one segment -----
            if (lAvailUser >= lNotifySize_bytes)
                {
                qwTotalMem += lNotifySize_bytes;

                time_t curTime = time (NULL);
                printf ("\rStat:%08x Pos:%08x Avail:%08x Total:%.2f MiB @ %.2lf MiB/s", lStatus, lPCPos, lAvailUser, (double) (int64) qwTotalMem / MEGA_B(1), (((double)qwTotalMem) / MEGA_B(1)) / (curTime - startTime));

                // this is the point to do anything with the data on the GPU

                // start kernel on the GPU to process the transfered data
                const int lThreadsPerBlock = 1024;
                if (lBytesPerSample == 1)
                    CudaKernelBlockAverage <<< (lNotifySize_bytes / sizeof (int8)) / lThreadsPerBlock, lThreadsPerBlock >>> ((int8*)((char*)pvDMABuffer_gpu + lPCPos), (int32*)((char*)pvBufferProcessed_gpu));
                    //CudaKernelBlockAverageNoiseSuppression <<< (lNotifySize_bytes / sizeof (int8)) / lThreadsPerBlock, lThreadsPerBlock >>> ((int8*)((char*)pvDMABuffer_gpu + lPCPos), (int32*)((char*)pvBufferProcessed_gpu), cThreshold, cBaseline);
                else
                    CudaKernelBlockAverage <<< (lNotifySize_bytes / sizeof (int16)) / lThreadsPerBlock, lThreadsPerBlock >>> ((int16*)((char*)pvDMABuffer_gpu + lPCPos), (int32*)((char*)pvBufferProcessed_gpu));
                    //CudaKernelBlockAverageNoiseSuppression <<< (lNotifySize_bytes / sizeof (int16)) / lThreadsPerBlock, lThreadsPerBlock >>> ((int16*)((char*)pvDMABuffer_gpu + lPCPos), (int32*)((char*)pvBufferProcessed_gpu), nThreshold, nBaseline);

                // ----- when we have added up the defined number of segments, we de-multiplex the result and copy it into the host's memory -----
                lAveragedSegments++;
                if (lAveragedSegments == lNumSegmentsToAverage)
                    {
                    // demux the channels using a kernel
                    if (lBytesPerSample == 1)
                        CudaKernelDemuxAveragedChannels <<< (lNotifySize_bytes / sizeof (int8)) / lThreadsPerBlock, lThreadsPerBlock >>>  ((int32*)((char*)pvBufferProcessed_gpu), lNumCh, aplCh_gpu[0], aplCh_gpu[1], aplCh_gpu[2], aplCh_gpu[3], aplCh_gpu[4], aplCh_gpu[5], aplCh_gpu[6], aplCh_gpu[7]);
                    else
                        CudaKernelDemuxAveragedChannels <<< (lNotifySize_bytes / sizeof (int16)) / lThreadsPerBlock, lThreadsPerBlock >>>  ((int32*)((char*)pvBufferProcessed_gpu), lNumCh, aplCh_gpu[0], aplCh_gpu[1], aplCh_gpu[2], aplCh_gpu[3], aplCh_gpu[4], aplCh_gpu[5], aplCh_gpu[6], aplCh_gpu[7]);

                    // for each channel copy processed data from GPU to host
                    // implicit cudaDeviceSynchronize, so the data is in host memory after cudaMemcpy has returned
                    for (int lChIdx = 0; lChIdx < lNumCh; ++lChIdx)
                        {
                        eCudaErr = cudaMemcpy (aplCh_host[lChIdx], aplCh_gpu[lChIdx], lSegmentSize_samples * sizeof (int32), cudaMemcpyDeviceToHost);
                        if (eCudaErr != cudaSuccess)
                            {
                            printf ("ERROR in cudaMemcpy(): %s\n", cudaGetErrorString(eCudaErr));
                            break;
                            }
                        }

                    // reset averaging buffer
                    cudaMemset (pvBufferProcessed_gpu, 0, lAveragedSize_bytes);

                    lAveragedSegments = 0;

                    printf ("\nSum of %d segments has been copied to host\n", lNumSegmentsToAverage);

                    // now the processed data is in the host memory and can be processed further, e.g. written to disk
                    }
                else
                    {
                    // wait until kernel has completed
                    cudaDeviceSynchronize ();
                    }

                spcm_dwSetParam_i32 (hCard, SPC_DATA_AVAIL_CARD_LEN, lNotifySize_bytes); // mark the segment as processed
                }
            }
        }

    // send the stop command
    dwError = spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_CARD_STOP | M2CMD_DATA_STOPDMA);

    // clean up
    printf ("\nFinished...\n");

    spcm_vClose (hCard);

    // free CUDA buffers on GPU and host
    cudaFree (pvBufferProcessed_gpu);
    cudaFree (pvDMABuffer_gpu);
    for (int lChIdx = 0; lChIdx < lNumCh; ++lChIdx)
        {
        cudaFreeHost (aplCh_host[lChIdx]);
        cudaFree (aplCh_gpu[lChIdx]);
        }

    return EXIT_SUCCESS;
    }

