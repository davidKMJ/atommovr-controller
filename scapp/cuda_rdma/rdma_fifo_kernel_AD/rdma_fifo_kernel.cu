/*
**************************************************************************

rdma_fifo_kernel.cu                             (c) Spectrum GmbH , 6/2017

**************************************************************************

Example for all M4i/M4x analog acquisition cards. 

Data is transfered in FIFO mode from card to GPU. A CUDA kernel is used
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

    // settings for the FIFO mode buffer handling
    int32       lNotifySize =   MEGA_B(2);
    int32       lBufferSize =   4*lNotifySize;


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


    // do a simple FIFO setup
    spcm_dwSetParam_i32 (hCard, SPC_CHENABLE,       (0x1 << lNumCh) - 1);
    spcm_dwSetParam_i32 (hCard, SPC_PRETRIGGER,     1024);                  // 1k of pretrigger data at start of FIFO mode
    spcm_dwSetParam_i32 (hCard, SPC_CARDMODE,       SPC_REC_FIFO_SINGLE);   // single FIFO mode
    spcm_dwSetParam_i32 (hCard, SPC_TIMEOUT,        5000);                  // timeout 5 s
    spcm_dwSetParam_i32 (hCard, SPC_TRIG_ORMASK,    SPC_TMASK_SOFTWARE);    // trigger set to software
    spcm_dwSetParam_i32 (hCard, SPC_TRIG_ANDMASK,   0);                     // ...
    spcm_dwSetParam_i32 (hCard, SPC_CLOCKMODE,      SPC_CM_INTPLL);         // clock mode internal PLL
    spcm_dwSetParam_i32 (hCard, SPC_CLOCKOUT,       0);                     // no clock output

    int lIR = 1000;                                                         // +/- 1 Volt
    spcm_dwSetParam_i32 (hCard, SPC_AMP0,           lIR);

    //spcm_dwSetParam_i64 (hCard, SPC_SPECIALCLOCK, 1);
    int64 llSamplerate = MEGA(100);
    spcm_dwSetParam_i64 (hCard, SPC_SAMPLERATE, llSamplerate);
    spcm_dwGetParam_i64 (hCard, SPC_SAMPLERATE, &llSamplerate);
    printf ("Samplerate: %lld\n", llSamplerate);
    // Card Setup finished
    // ------------------------------------------------------------------------


    // ----- DMA BUFFER SETUP -----
    // ----- get buffer on GPU that will be used as target for RDMA transfer -----
    int lCUDADeviceIdx = 0;         // index of used CUDA device
    void* pvDMABuffer_gpu = pvGetRDMABuffer (lCUDADeviceIdx, lBufferSize);
    if (pvDMABuffer_gpu == NULL)
        {
        spcm_vClose (hCard);
        return EXIT_FAILURE;
        }

    // setup DMA transfer from Spectrum card to GPU
    spcm_dwDefTransfer_i64 (hCard, SPCM_BUF_DATA, SPCM_DIR_CARDTOGPU, lNotifySize, pvDMABuffer_gpu, 0, lBufferSize);


    // ----- create a second buffer on the GPU that will be used as target for the kernel function -----
    void* pvBufferProcessed_gpu = 0;
    cudaError_t eCudaErr = cudaMalloc (&pvBufferProcessed_gpu, lNotifySize);
    if (eCudaErr != cudaSuccess)
        {
        // cleanup
        printf ("ERROR in cudaMalloc(): %s\n", cudaGetErrorString(eCudaErr));

        spcm_vClose (hCard);

        cudaFree (pvDMABuffer_gpu);

        return EXIT_FAILURE;
        }

    // ----- create a buffer on the host side. This will be used to transfer the processed data from the GPU -----
    void* pvBuffer_host = NULL;
    eCudaErr = cudaMallocHost (&pvBuffer_host, lNotifySize);
    if (eCudaErr != cudaSuccess)
        {
        // cleanup
        printf ("ERROR in cudaMallocHost(): %s\n", cudaGetErrorString(eCudaErr));

        spcm_vClose (hCard);

        // free allocated CUDA buffers on GPU
        cudaFree (pvBufferProcessed_gpu);
        cudaFree (pvDMABuffer_gpu);

        return EXIT_FAILURE;
        }
    memset (pvBuffer_host, 0, lNotifySize);

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
        cudaFreeHost (pvBuffer_host);
        cudaFree (pvBufferProcessed_gpu);
        cudaFree (pvDMABuffer_gpu);

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
            spcm_dwGetParam_i32 (hCard, SPC_DATA_AVAIL_USER_POS,  &lPCPos);

            if (lAvailUser >= lNotifySize)
                {
                qwTotalMem += lNotifySize;

                time_t curTime = time (NULL);
                printf ("\rStat:%08x Pos:%08x Avail:%08x Total:%.2f MiB @ %.2lf MiB/s", lStatus, lPCPos, lAvailUser, (double) (int64) qwTotalMem / MEGA_B(1), (((double)qwTotalMem) / MEGA_B(1)) / (curTime - startTime));

                // this is the point to do anything with the data on the GPU

                // start kernel on the GPU to process the transfered data
                const int lThreadsPerBlock = 1024;
                if (lBytesPerSample == 1)
                    CudaKernelInvert <<< (lNotifySize / sizeof (int8)) / lThreadsPerBlock, lThreadsPerBlock >>> ((int8*)((char*)pvDMABuffer_gpu + lPCPos), (int8*)((char*)pvBufferProcessed_gpu), lNotifySize);
                else
                    CudaKernelInvert <<< (lNotifySize / sizeof (int16)) / lThreadsPerBlock, lThreadsPerBlock >>> ((int16*)((char*)pvDMABuffer_gpu + lPCPos), (int16*)((char*)pvBufferProcessed_gpu), lNotifySize);

                // after kernel has finished we copy processed data from GPU to host
                eCudaErr = cudaMemcpy (pvBuffer_host, pvBufferProcessed_gpu, lNotifySize, cudaMemcpyDeviceToHost);
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

    // send the stop command
    dwError = spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_CARD_STOP | M2CMD_DATA_STOPDMA);

    // clean up
    printf ("\nFinished...\n");

    spcm_vClose (hCard);

    // free CUDA buffers on GPU
    cudaFree (pvBufferProcessed_gpu);
    cudaFree (pvDMABuffer_gpu);

    // free CUDA buffer on host
    cudaFreeHost (pvBuffer_host);

    return EXIT_SUCCESS;
    }

