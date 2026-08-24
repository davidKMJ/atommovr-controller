/*
**************************************************************************

rdma_fifo_kernel_DA.cu                          (c) Spectrum GmbH , 9/2018

**************************************************************************

Example for all M4i/M4x analog replay cards. 

Data is transfered in FIFO mode from GPU to card. A CUDA kernel is used
to multiplex the data into the correct format.
  
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


// CUDA kernel that muxes the samples of up to four channels into one array
__global__ void CudaKernelMuxChannels (const short* pnInCh0, const short* pnInCh1, const short* pnInCh2, const short* pnInCh3, int lNumCh, short* pnOut)
    {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int lChIdx = i % lNumCh;
    int lSample = i / lNumCh;
    switch (lChIdx)
        {
        // up to 4 channels with linear sorting (M4i)
        case 0: pnOut[i] = pnInCh0[lSample]; break;
        case 1: pnOut[i] = pnInCh1[lSample]; break;
        case 2: pnOut[i] = pnInCh2[lSample]; break;
        case 3: pnOut[i] = pnInCh3[lSample]; break;
        }
    }


/*
**************************************************************************
main 
**************************************************************************
*/

int main ()
    {
    int32       lCardType, lSerialNumber, lFncType, lNumCh, lBytesPerSample;
    char        szErrorTextBuffer[ERRORTEXTLEN];
    uint32      dwError;
    int32       lStatus, lAvailUser, lUserPos, lFillsize;
    uint64      qwTotalMem = 0;
    uint64      qwToTransfer = GIGA_B(16);

    // settings for the FIFO mode buffer handling
    int32       lNotifySize =   MEGA_B(2);
    int32       lBufferSize =   4*lNotifySize;


    // ------------------------------------------------------------------------
    // CARD SETUP

    // ----- open Spectrum card -----
    drv_handle hCard = spcm_hOpen ((char*)"/dev/spcm0");
    if (!hCard)
        {
        printf ("no card found...\n");
        return 0;
        }

    // ----- read type, function and sn and check for A/D card -----
    spcm_dwGetParam_i32 (hCard, SPC_PCITYP,                &lCardType);
    spcm_dwGetParam_i32 (hCard, SPC_PCISERIALNO,           &lSerialNumber);
    spcm_dwGetParam_i32 (hCard, SPC_FNCTYPE,               &lFncType);
    spcm_dwGetParam_i32 (hCard, SPC_MIINST_CHPERMODULE,    &lNumCh);
    spcm_dwGetParam_i32 (hCard, SPC_MIINST_BYTESPERSAMPLE, &lBytesPerSample);

    // ----- print used card or error message if the found card is not supported by this example -----
    char szType[50];
    spcm_dwGetParam_ptr (hCard, SPC_PCITYP, szType, sizeof (szType));
    switch (lFncType)
        {
        case SPCM_TYPE_AO:
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
            return EXIT_FAILURE;
        }

    // ----- do a simple FIFO setup for 66xx -----
    spcm_dwSetParam_i32 (hCard, SPC_CHENABLE,       (0x1 << lNumCh) - 1);   // enable all channels
    spcm_dwSetParam_i32 (hCard, SPC_CARDMODE,       SPC_REP_FIFO_SINGLE);   // single FIFO mode
    spcm_dwSetParam_i64 (hCard, SPC_SEGMENTSIZE,    1024);
    spcm_dwSetParam_i64 (hCard, SPC_LOOPS,          0);                     // forever
    spcm_dwSetParam_i32 (hCard, SPC_TRIG_ORMASK,    SPC_TMASK_SOFTWARE);    // trigger set to software
    spcm_dwSetParam_i32 (hCard, SPC_TRIG_ANDMASK,   0);                     // ...
    int64 llSamplerate = MEGA(100);
    spcm_dwSetParam_i32 (hCard, SPC_CLOCKMODE,      SPC_CM_INTPLL);         // clock mode internal PLL
    spcm_dwSetParam_i64 (hCard, SPC_SAMPLERATE,     llSamplerate);
    spcm_dwSetParam_i32 (hCard, SPC_TIMEOUT,        5*1000);
    int32 lMaxOutputLevel = 1000; // +-1 Volt
    for (int lChIdx = 0; lChIdx < lNumCh; ++lChIdx)
        {
        spcm_dwSetParam_i32 (hCard, SPC_ENABLEOUT0 + lChIdx * (SPC_ENABLEOUT1 - SPC_ENABLEOUT0), 1);
        spcm_dwSetParam_i32 (hCard, SPC_AMP0       + lChIdx * (SPC_AMP1        - SPC_AMP0),      lMaxOutputLevel);
        }

    dwError = spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_CARD_WRITESETUP);

    // Card Setup finished
    // ------------------------------------------------------------------------


    // ----- DMA BUFFER SETUP -----
    // ----- get buffer on GPU that will be used as source for RDMA transfer -----
    int lCUDADeviceIdx = 0;         // index of used CUDA device
    void* pvDMABuffer_gpu = NULL;
    pvDMABuffer_gpu = pvGetRDMABuffer (lCUDADeviceIdx, lBufferSize);
    if (pvDMABuffer_gpu == NULL)
        {
        spcm_vClose (hCard);
        return EXIT_FAILURE;
        }

    // ----- allocate memory for each channel on GPU host to use for copying the waveform data -----
    cudaError_t eCudaErr = cudaSuccess;
    int16* apnCh_host[4] = { NULL, NULL, NULL, NULL };
    int16* apnCh_gpu[4]  = { NULL, NULL, NULL, NULL };
    int lBytesPerChannelInNotifySize = lNotifySize / lNumCh;
    int lSamplesPerChannelInNotifySize = lBytesPerChannelInNotifySize / sizeof (int16);
    for (int lChIdx = 0; lChIdx < lNumCh; ++lChIdx)
        {
        eCudaErr = cudaMallocHost (apnCh_host + lChIdx, lBytesPerChannelInNotifySize);
        if (eCudaErr != cudaSuccess)
            {
            printf ("Allocating memory for Ch%d on host failed\n", lChIdx);
            cudaFree (pvDMABuffer_gpu);
            for (int i = 0; i < lChIdx; ++i)
                {
                cudaFreeHost (apnCh_host + i);
                cudaFree     (apnCh_gpu + i);
                }
            spcm_vClose (hCard);
            return EXIT_FAILURE;
            }
        eCudaErr = cudaMalloc (apnCh_gpu + lChIdx, lBytesPerChannelInNotifySize);
        if (eCudaErr != cudaSuccess)
            {
            printf ("Allocating memory for Ch%d on GPU failed\n", lChIdx);

            spcm_vClose (hCard);

            cudaFree (pvDMABuffer_gpu);
            cudaFreeHost (apnCh_host + lChIdx);
            for (int i = 0; i < lChIdx; ++i)
                {
                cudaFreeHost (apnCh_host + i);
                cudaFree     (apnCh_gpu + i);
                }
                
            return EXIT_FAILURE;
            }

        // ----- calculate some waveforms for each channel -----
        if (lChIdx == 0)
            {
            // rect
            for (int lSample = 0; lSample < lSamplesPerChannelInNotifySize; ++lSample)
                apnCh_host[lChIdx][lSample] = (lSample < lSamplesPerChannelInNotifySize/2? 16384 : -16384);
            }
        else if (lChIdx == 1)
            {
            // sine
            for (int lSample = 0; lSample < lSamplesPerChannelInNotifySize; ++lSample)
                apnCh_host[lChIdx][lSample] = 16384 * sin (2.0 * M_PI * lSample / lSamplesPerChannelInNotifySize);
            }
        else if (lChIdx == 2)
            {
            // sawtooth
            for (int lSample = 0; lSample < lSamplesPerChannelInNotifySize; ++lSample)
                apnCh_host[lChIdx][lSample] = -16384 + (32768. * lSample) / lSamplesPerChannelInNotifySize;
            }
        else
            {
            // cosine with double speed
            for (int lSample = 0; lSample < lSamplesPerChannelInNotifySize; ++lSample)
                apnCh_host[lChIdx][lSample] = 16384 * cos (2.0 * M_PI * lSample / (lSamplesPerChannelInNotifySize / 2));
            }
        }

    // ----- setup DMA transfer from GPU to Spectrum card -----
    spcm_dwDefTransfer_i64 (hCard, SPCM_BUF_DATA, SPCM_DIR_GPUTOCARD, lNotifySize, pvDMABuffer_gpu, 0, lBufferSize);

    // ----- fill the software buffer before we start the card -----
    for (int32 lPosInBuf = 0; lPosInBuf < lBufferSize; lPosInBuf += lNotifySize)
        {
        // copy waveforms from host to GPU
        // for simplicity we will reuse the same data
        for (int lChIdx = 0; lChIdx < lNumCh; ++lChIdx)
            cudaMemcpy (apnCh_gpu[lChIdx], apnCh_host[lChIdx], lBytesPerChannelInNotifySize, cudaMemcpyHostToDevice);

        // mux waveforms using a CUDA kernel
        const int lThreadsPerBlock = 1024;
        CudaKernelMuxChannels <<< (lNotifySize / sizeof (int16)) / lThreadsPerBlock, lThreadsPerBlock >>> (apnCh_gpu[0], apnCh_gpu[1], apnCh_gpu[2], apnCh_gpu[3], lNumCh, (int16*)((char*)pvDMABuffer_gpu + lPosInBuf));

        // mark data as valid
        spcm_dwSetParam_i32 (hCard, SPC_DATA_AVAIL_CARD_LEN,  lNotifySize);
        }

    // ----- start transfer from GPU into card and wait until it has finished -----
    dwError = spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_DATA_STARTDMA | M2CMD_DATA_WAITDMA);
    if (dwError != ERR_OK)
        {
        spcm_dwGetErrorInfo_i32 (hCard, NULL, NULL, szErrorTextBuffer);
        printf ("Error on STARTDMA | WAITDMA: %u (%s)\n", dwError, szErrorTextBuffer);

        spcm_vClose (hCard);

        // free CUDA buffers on GPU and host
        cudaFree (pvDMABuffer_gpu);
        for (int lChIdx = 0; lChIdx < lNumCh; ++lChIdx)
            {
            cudaFreeHost (apnCh_host + lChIdx);
            cudaFree (apnCh_gpu + lChIdx);
            }

        return EXIT_FAILURE;
        }

    // ----- start everything -----
    time_t startTime = time (NULL);
    dwError = spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER);
    if (dwError != ERR_OK)
        {
        // cleanup
        spcm_dwGetErrorInfo_i32 (hCard, NULL, NULL, szErrorTextBuffer);
        printf ("CARD_START failed: %u (%s)\n", dwError, szErrorTextBuffer);

        spcm_vClose (hCard);

        // free allocated CUDA buffers on host and GPU
        cudaFree (pvDMABuffer_gpu);
        for (int lChIdx = 0; lChIdx < lNumCh; ++lChIdx)
            {
            cudaFreeHost (apnCh_host + lChIdx);
            cudaFree (apnCh_gpu + lChIdx);
            }

        return EXIT_FAILURE;
        }


    // run the FIFO mode and loop through the data
    // the control of the DMA transfer is the same as without RDMA.
    // The difference is that the driver reports a certain amount of data as available to the user,
    // but the data has been transfered into the GPU memory.
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
            spcm_dwGetParam_i32 (hCard, SPC_DATA_AVAIL_USER_POS,  &lUserPos);
            spcm_dwGetParam_i32 (hCard, SPC_FILLSIZEPROMILLE,     &lFillsize);

            if (lAvailUser >= lNotifySize)
                {
                qwTotalMem += lNotifySize;

                time_t curTime = time (NULL);
                printf ("\rStat:%08x Fill:%d%% Pos:%08x Avail:%08x Total:%.2f MiB @ %.2lf MiB/s", lStatus, lFillsize/10, lUserPos, lAvailUser, (double) (int64) qwTotalMem / MEGA_B(1), (((double)qwTotalMem) / MEGA_B(1)) / (curTime - startTime));

                // copy waveforms from host to GPU
                // for simplicity we will reuse the same data
                for (int lChIdx = 0; lChIdx < lNumCh; ++lChIdx)
                    cudaMemcpy (apnCh_gpu[lChIdx], apnCh_host[lChIdx], lBytesPerChannelInNotifySize, cudaMemcpyHostToDevice);

                // this is the point to do anything with the data on the GPU

                // mux waveforms using a CUDA kernel
                const int lThreadsPerBlock = 1024;
                CudaKernelMuxChannels <<< (lNotifySize / sizeof (int16)) / lThreadsPerBlock, lThreadsPerBlock >>> (apnCh_gpu[0], apnCh_gpu[1], apnCh_gpu[2], apnCh_gpu[3], lNumCh, (int16*)((char*)pvDMABuffer_gpu + lUserPos));

                // wait until kernel has finished before we mark the data as valid
                cudaDeviceSynchronize ();


                // mark data as valid and transfer it to card
                spcm_dwSetParam_i32 (hCard, SPC_DATA_AVAIL_CARD_LEN,  lNotifySize);
                }
            }
        }

    // send the stop command
    dwError = spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_CARD_STOP | M2CMD_DATA_STOPDMA);

    // clean up
    printf ("\nFinished...\n");

    spcm_vClose (hCard);

    // free CUDA buffers on GPU and host
    cudaFree (pvDMABuffer_gpu);
    for (int lChIdx = 0; lChIdx < lNumCh; ++lChIdx)
        {
        cudaFreeHost (apnCh_host + lChIdx);
        cudaFree (apnCh_gpu + lChIdx);
        }

    return EXIT_SUCCESS;
    }

