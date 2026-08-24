/*
**************************************************************************

rdma_fifo_kernel.cu                             (c) Spectrum GmbH , 6/2017

**************************************************************************

Example for all M4i/M4x/M2p synchronized analog acquisition cards. 

Data is transfered in FIFO mode from cards to GPU. A CUDA kernel is used
to invert the data.
  
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

#include "../../common/ostools/spcm_oswrap.h"
#include "../../common/ostools/spcm_ostools.h"

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
__global__ void CudaKernelInvert (int8* pcIn, int8* pcOut, int N)
    {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    pcOut[i] = -1*(pcIn[i]);
    }
__global__ void CudaKernelInvert (short* pnIn, short* pnOut, int N)
    {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    pnOut[i] = -1*(pnIn[i]);
    }


__global__ void CudaKernelDemuxChannels (short* pnIn, int lNumCh, short* pnOutCh0, short* pnOutCh1, short* pnOutCh2, short* pnOutCh3, short* pnOutCh4, short* pnOutCh5, short* pnOutCh6, short* pnOutCh7)
    {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int lChIdx = i % lNumCh;
    int lSample = i / lNumCh;
    switch (lChIdx)
        {
        // up to 4 channels with linear sorting (M4i)
        // up to 8 channels with linear sorting (M2p)
        case 0: pnOutCh0[lSample] = pnIn[i]; break;
        case 1: pnOutCh1[lSample] = pnIn[i]; break;
        case 2: pnOutCh2[lSample] = pnIn[i]; break;
        case 3: pnOutCh3[lSample] = pnIn[i]; break;
        case 4: pnOutCh4[lSample] = pnIn[i]; break;
        case 5: pnOutCh5[lSample] = pnIn[i]; break;
        case 6: pnOutCh6[lSample] = pnIn[i]; break;
        case 7: pnOutCh7[lSample] = pnIn[i]; break;
        }
    }


struct ST_THREADARGUMENTS
    {
    drv_handle hCard;
    void* pvDMABuffer_gpu;
    void* pvBufferProcessed_gpu;
    void* pvBuffer_host;
    uint64 qwToTransfer;
    int32 lNotifySize;
    uint32 dwCardIdx;
    };

SPCM_THREAD_RETURN SPCM_THREAD_CALLTYPE vDMAThread (void* pvThreadArguments);

/*
**************************************************************************
main 
**************************************************************************
*/

#define MAX_CARDS 16

int main ()
    {
    drv_handle  ahCard[MAX_CARDS] = { NULL };
    char        szErrorTextBuffer[ERRORTEXTLEN];
    uint32      dwError;
    uint64      qwToTransfer = GIGA_B(16);

    // settings for the FIFO mode buffer handling
    int32       lNotifySize =   MEGA_B(2);
    int32       lBufferSize =   4*lNotifySize;


    // ------------------------------------------------------------------------
    // CARD SETUP

    // ----- open Spectrum cards -----
    uint32 dwCardIdx =          0;
    uint32 dwNumCards =         0;
    uint16 wStarHubCarrierIdx = 0xFFFF;
    while (true)
        {
        char szCard[16];
        sprintf (szCard, "/dev/spcm%u", dwCardIdx);
        ahCard[dwNumCards] = spcm_hOpen ((char*)szCard);
        if (!ahCard[dwNumCards])
            {
            if (dwCardIdx == 0)
                {
                printf ("no card found...\n");
                return 0;
                }
            else
                // end while loop
                break;
            }

        // read type, function and sn and check for A/D card
        int32 lCardType =     0;
        int32 lSerialNumber = 0;
        int32 lFncType =      0;
        spcm_dwGetParam_i32 (ahCard[dwNumCards], SPC_PCITYP,                &lCardType);
        spcm_dwGetParam_i32 (ahCard[dwNumCards], SPC_PCISERIALNO,           &lSerialNumber);
        spcm_dwGetParam_i32 (ahCard[dwNumCards], SPC_FNCTYPE,               &lFncType);

        char szType[50];
        spcm_dwGetParam_ptr (ahCard[dwNumCards], SPC_PCITYP, szType, sizeof (szType));

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

                // skip this card
                spcm_vClose (ahCard[dwNumCards]);
                ahCard[dwNumCards] = NULL;
                break;
            }

        // increase index for /dev/spcmX
        dwCardIdx++;

        // count cards that are usable for this example
        if (ahCard[dwNumCards] != NULL)
            {
            // check if this card carries a starhub
            int32 lFeatures = 0;
            spcm_dwGetParam_i32 (ahCard[dwNumCards], SPC_PCIFEATURES, &lFeatures);
            if (lFeatures & (SPCM_FEAT_STARHUB8_EXTM | SPCM_FEAT_STARHUB6_EXTM | SPCM_FEAT_STARHUB16_EXTM))
               wStarHubCarrierIdx = dwNumCards; 

            dwNumCards++;
            }
        }
    if (dwNumCards < 2)
        {
        printf ("This example needs at least two cards to run.\n");
        return EXIT_FAILURE;
        }
    if (wStarHubCarrierIdx == 0xFFFF)
        {
        printf ("No starhub found.\n");
        return EXIT_FAILURE;
        }


    // do a simple FIFO setup
    for (uint32 dwCardIdx = 0; dwCardIdx < dwNumCards; ++dwCardIdx)
        {
        int32 lNumCh = 1; // we use only one channel on each card
        spcm_dwSetParam_i32 (ahCard[dwCardIdx], SPC_CHENABLE,       (0x1 << lNumCh) - 1);
        spcm_dwSetParam_i32 (ahCard[dwCardIdx], SPC_PRETRIGGER,     1024);                  // 1k of pretrigger data at start of FIFO mode
        spcm_dwSetParam_i32 (ahCard[dwCardIdx], SPC_CARDMODE,       SPC_REC_FIFO_SINGLE);   // single FIFO mode
        spcm_dwSetParam_i32 (ahCard[dwCardIdx], SPC_TIMEOUT,        5000);                  // timeout 5 s
        spcm_dwSetParam_i32 (ahCard[dwCardIdx], SPC_TRIG_ORMASK,    SPC_TMASK_SOFTWARE);    // trigger set to software
        spcm_dwSetParam_i32 (ahCard[dwCardIdx], SPC_TRIG_ANDMASK,   0);                     // ...
        spcm_dwSetParam_i32 (ahCard[dwCardIdx], SPC_CLOCKMODE,      SPC_CM_INTPLL);         // clock mode internal PLL
        spcm_dwSetParam_i32 (ahCard[dwCardIdx], SPC_CLOCKOUT,       0);                     // no clock output

        int lIR = 1000;                                                         // +/- 1 Volt
        spcm_dwSetParam_i32 (ahCard[dwCardIdx], SPC_AMP0,           lIR);

        //spcm_dwSetParam_i64 (hCard, SPC_SPECIALCLOCK, 1);
        int64 llSamplerate = MEGA(125);
        spcm_dwSetParam_i64 (ahCard[dwCardIdx], SPC_SAMPLERATE, llSamplerate);
        spcm_dwGetParam_i64 (ahCard[dwCardIdx], SPC_SAMPLERATE, &llSamplerate);
        printf ("Samplerate: %lld\n", llSamplerate);
        }
    // Card Setup finished
    // ------------------------------------------------------------------------

    // ------------------------------------------------------------------------
    // Starhub Setup
    drv_handle hSync;
    hSync = spcm_hOpen ((char*)"sync0");
    if (!hSync)
        {
        printf ("Could not open Starhub handle\n");
        return EXIT_FAILURE;
        }

    // sync setup, all cards activated, starhub carrier is clock master
    spcm_dwSetParam_i32 (hSync, SPC_SYNC_ENABLEMASK, (1 << dwNumCards) - 1);
    spcm_dwSetParam_i32 (hSync, SPC_SYNC_CLKMASK, (1 << wStarHubCarrierIdx));

    // Starhub Setup finished
    // ------------------------------------------------------------------------


    void* apvDMABuffer_gpu[MAX_CARDS];
    void* apvBufferProcessed_gpu[MAX_CARDS];
    void* apvBuffer_host[MAX_CARDS];
    for (uint32 dwCardIdx = 0; dwCardIdx < dwNumCards; ++dwCardIdx)
        {
        // ----- DMA BUFFER SETUP -----
        // ----- get buffer on GPU that will be used as target for RDMA transfer -----
        int lCUDADeviceIdx = 0;         // index of used CUDA device
        apvDMABuffer_gpu[dwCardIdx] = pvGetRDMABuffer (lCUDADeviceIdx, lBufferSize);
        if (apvDMABuffer_gpu[dwCardIdx] == NULL)
            {
            spcm_vClose (ahCard[dwCardIdx]);
            return EXIT_FAILURE;
            }

        // setup DMA transfer from Spectrum card to GPU
        spcm_dwDefTransfer_i64 (ahCard[dwCardIdx], SPCM_BUF_DATA, SPCM_DIR_CARDTOGPU, lNotifySize, apvDMABuffer_gpu[dwCardIdx], 0, lBufferSize);

        // ----- create a second buffer on the GPU that will be used as target for the kernel function -----
        cudaError_t eCudaErr = cudaMalloc (apvBufferProcessed_gpu + dwCardIdx, lNotifySize);
        if (eCudaErr != cudaSuccess)
            {
            // cleanup
            printf ("ERROR in cudaMalloc(): %s\n", cudaGetErrorString(eCudaErr));

            spcm_vClose (ahCard[dwCardIdx]);

            cudaFree (apvDMABuffer_gpu[dwCardIdx]);

            return EXIT_FAILURE;
            }

        // ----- create a buffer on the host side. This will be used to transfer the processed data from the GPU -----
        eCudaErr = cudaMallocHost (apvBuffer_host + dwCardIdx, lNotifySize);
        if (eCudaErr != cudaSuccess)
            {
            // cleanup
            printf ("ERROR in cudaMallocHost(): %s\n", cudaGetErrorString(eCudaErr));

            spcm_vClose (ahCard[dwCardIdx]);

            // free allocated CUDA buffers on GPU
            cudaFree (apvBufferProcessed_gpu[dwCardIdx]);
            cudaFree (apvDMABuffer_gpu[dwCardIdx]);

            return EXIT_FAILURE;
            }
        memset (apvBuffer_host[dwCardIdx], 0, lNotifySize);
        }

    // ----- create one thread for each card to handle DMA -----
    SPCM_THREAD_HANDLE ahThreads[MAX_CARDS];
    struct ST_THREADARGUMENTS astThreadArguments[MAX_CARDS];
    for (uint32 dwCardIdx = 0; dwCardIdx < dwNumCards; ++dwCardIdx)
        {
        // thread will immediately start waiting for DMA events, so we start DMA before we start the cards
        spcm_dwSetParam_i32 (ahCard[dwCardIdx], SPC_M2CMD, M2CMD_DATA_STARTDMA);

        astThreadArguments[dwCardIdx].hCard =                 ahCard[dwCardIdx];
        astThreadArguments[dwCardIdx].pvDMABuffer_gpu =       apvDMABuffer_gpu[dwCardIdx];
        astThreadArguments[dwCardIdx].pvBufferProcessed_gpu = apvBufferProcessed_gpu[dwCardIdx];
        astThreadArguments[dwCardIdx].pvBuffer_host =         apvBuffer_host[dwCardIdx];
        astThreadArguments[dwCardIdx].qwToTransfer =          qwToTransfer;
        astThreadArguments[dwCardIdx].lNotifySize =           lNotifySize;
        astThreadArguments[dwCardIdx].dwCardIdx =             dwCardIdx;

        spcm_bCreateThread (vDMAThread, ahThreads + dwCardIdx, (void*) &astThreadArguments[dwCardIdx]);
        }

    // ----- start all cards -----
    dwError = spcm_dwSetParam_i32 (hSync, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER);
    if (dwError != ERR_OK)
        {
        // cleanup
        spcm_dwGetErrorInfo_i32 (hSync, NULL, NULL, szErrorTextBuffer);
        printf ("%s\n", szErrorTextBuffer);

        for (uint32 dwCardIdx = 0; dwCardIdx < dwNumCards; ++dwCardIdx)
            {
            spcm_vClose (ahCard[dwCardIdx]);

            // free allocated CUDA buffers on host and GPU
            cudaFreeHost (apvBuffer_host[dwCardIdx]);
            cudaFree (apvBufferProcessed_gpu[dwCardIdx]);
            cudaFree (apvDMABuffer_gpu[dwCardIdx]);
            }

        return EXIT_FAILURE;
        }

    // wait for threads to finish
    for (uint32 dwCardIdx = 0; dwCardIdx < dwNumCards; ++dwCardIdx)
        {
        spcm_vJoinThread (&ahThreads[dwCardIdx], 0);
        }

    // send the stop command
    dwError = spcm_dwSetParam_i32 (hSync, SPC_M2CMD, M2CMD_CARD_STOP | M2CMD_DATA_STOPDMA);

    // clean up
    printf ("\nFinished...\n");

    for (uint32 dwCardIdx = 0; dwCardIdx < dwNumCards; ++dwCardIdx)
        {
        spcm_vClose (ahCard[dwCardIdx]);

        // free CUDA buffers on GPU
        cudaFree (apvBufferProcessed_gpu[dwCardIdx]);
        cudaFree (apvDMABuffer_gpu[dwCardIdx]);

        // free CUDA buffer on host
        cudaFreeHost (apvBuffer_host[dwCardIdx]);
        }

    return EXIT_SUCCESS;
    }


// ------ Thread function. One thread will be started for each card. ------
SPCM_THREAD_RETURN SPCM_THREAD_CALLTYPE vDMAThread (void* pvThreadArguments)
    {
    struct ST_THREADARGUMENTS* pstArguments = (struct ST_THREADARGUMENTS*)pvThreadArguments;

    // to improve readability we copy the variables from the struct
    drv_handle hCard =            pstArguments->hCard;
    void* pvDMABuffer_gpu =       pstArguments->pvDMABuffer_gpu;
    void* pvBufferProcessed_gpu = pstArguments->pvBufferProcessed_gpu;
    void* pvBuffer_host =         pstArguments->pvBuffer_host;
    uint64 qwToTransfer =         pstArguments->qwToTransfer;
    int32 lNotifySize =           pstArguments->lNotifySize;
    uint32 dwCardIdx =            pstArguments->dwCardIdx;

    uint32 dwError =    ERR_OK;
    uint64 qwTotalMem = 0;
    int32  lStatus =    0;
    int32  lAvailUser = 0;
    int32  lPCPos =     0;

    // bytes per sample will be used to select correct kernel (8 bit vs 12/24/16 bit)
    int32 lBytesPerSample = 0;
    spcm_dwGetParam_i32 (hCard, SPC_MIINST_BYTESPERSAMPLE, &lBytesPerSample);

    // store start time for speed calculations
    time_t startTime = time (NULL);

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
            spcm_dwGetParam_i32 (hCard, SPC_DATA_AVAIL_USER_POS,  &lPCPos);

            if (lAvailUser >= lNotifySize)
                {
                qwTotalMem += lNotifySize;

                time_t curTime = time (NULL);
                printf ("\r%u: Stat:%08x Pos:%08x Avail:%08x Total:%.2f MiB @ %.2lf MiB/s", dwCardIdx, lStatus, lPCPos, lAvailUser, (double) (int64) qwTotalMem / MEGA_B(1), (((double)qwTotalMem) / MEGA_B(1)) / (curTime - startTime));

                // this is the point to do anything with the data on the GPU

                // start kernel on the GPU to process the transfered data
                // !!! keep in mind that only one kernel at a time can run on the GPU !!!
                // !!! e.g. kernel for first card, then the kernel from another thread for a second card, and so on !!!
                const int lThreadsPerBlock = 1024;
                if (lBytesPerSample == 1)
                    CudaKernelInvert <<< (lNotifySize / sizeof (int8)) / lThreadsPerBlock, lThreadsPerBlock >>> ((int8*)((char*)pvDMABuffer_gpu + lPCPos), (int8*)((char*)pvBufferProcessed_gpu), lNotifySize);
                else
                    CudaKernelInvert <<< (lNotifySize / sizeof (int16)) / lThreadsPerBlock, lThreadsPerBlock >>> ((int16*)((char*)pvDMABuffer_gpu + lPCPos), (int16*)((char*)pvBufferProcessed_gpu), lNotifySize);

                // after kernel has finished we copy processed data from GPU to host
                cudaError_t eCudaErr = cudaMemcpy (pvBuffer_host, pvBufferProcessed_gpu, lNotifySize, cudaMemcpyDeviceToHost);
                if (eCudaErr != cudaSuccess)
                    {
                    printf ("ERROR in cudaMemcpy(): %s\n", cudaGetErrorString(eCudaErr));
                    break;
                    }

                // now the processed data is in the host memory

                spcm_dwSetParam_i32 (hCard, SPC_DATA_AVAIL_CARD_LEN,  lNotifySize);
                }
            }
        }

    return 0;
    }

