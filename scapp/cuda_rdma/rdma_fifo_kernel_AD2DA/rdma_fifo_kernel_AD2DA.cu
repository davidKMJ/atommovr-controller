/*
**************************************************************************

rdma_fifo_kernel_AD2DA.cu                       (c) Spectrum GmbH, 11/2020

**************************************************************************

Example for a pair of one analog acquisition card and one analog replay
card.

Data is transfered in FIFO mode from AD card to GPU. A CUDA kernel is used
to invert the data. Then the data is transfered to the DA card and replayed.
This example requires a connection between Clock output of the AD card and
Clock input of the DA card.
  
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
__global__ void CudaKernelInvert (short* pnIn, short* pnOut, int N)
    {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    pnOut[i] = -1*(pnIn[i]);
    }

/*
**************************************************************************
main 
**************************************************************************
*/

int main ()
    {
    int32       lCardType, lSerialNumber, lFncType, lNumCh;
    char        szErrorTextBuffer[ERRORTEXTLEN];
    uint32      dwError;
    int32       lStatus = 0, lAvailUser = 0, lPCPosAD = 0, lPCPosDA = 0;
    uint64      qwTotalMem = 0;
    uint64      qwToTransfer = GIGA_B(4); // the program will end after this amount of data has been transfered

    // settings for the FIFO mode buffer handling
    int32       lNotifySize =   KILO_B(512);
    int32       lBufferSize =   4*lNotifySize;
    // we will simply FORCETRIGGER the DA card after some data has been acquired by the AD card.
    // for ways to decrease the latency between input and output see ../../test/closed_loop_ad_da example


    // ------------------------------------------------------------------------
    // CARD SETUP

    // ----- open Spectrum card -----
    drv_handle hCardAD = spcm_hOpen ((char*)"/dev/spcm0");
    if (!hCardAD)
        {
        printf ("no AD card found...\n");
        return 0;
        }

    drv_handle hCardDA = spcm_hOpen ((char*)"/dev/spcm1");
    if (!hCardDA)
        {
        printf ("no DA card found...\n");
        return 0;
        }

    // read type, function and sn and check for A/D card
    spcm_dwGetParam_i32 (hCardAD, SPC_PCITYP,                &lCardType);
    spcm_dwGetParam_i32 (hCardAD, SPC_PCISERIALNO,           &lSerialNumber);
    spcm_dwGetParam_i32 (hCardAD, SPC_FNCTYPE,               &lFncType);

    char szType[50];
    spcm_dwGetParam_ptr (hCardAD, SPC_PCITYP, szType, sizeof (szType));

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
            printf ("Card: %s sn %05d is no A/D card\n", szType, lSerialNumber);            
            return 0;
        }

    // read type, function and sn and check for D/A card
    spcm_dwGetParam_i32 (hCardDA, SPC_PCITYP,                &lCardType);
    spcm_dwGetParam_i32 (hCardDA, SPC_PCISERIALNO,           &lSerialNumber);
    spcm_dwGetParam_i32 (hCardDA, SPC_FNCTYPE,               &lFncType);

    spcm_dwGetParam_ptr (hCardDA, SPC_PCITYP, szType, sizeof (szType));

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
            printf ("Card: %s sn %05d is no D/A card\n", szType, lSerialNumber);            
            return 0;
        }


    // -----------------------------------------------------------------------
    // AD Card Setup
    // do a simple FIFO setup
    lNumCh = 1; // use only one channel
    spcm_dwSetParam_i32 (hCardAD, SPC_CHENABLE,       (0x1 << lNumCh) - 1);   // enable channel(s)
    spcm_dwSetParam_i32 (hCardAD, SPC_PRETRIGGER,     1024);                  // 1k of pretrigger data at start of FIFO mode
    spcm_dwSetParam_i32 (hCardAD, SPC_CARDMODE,       SPC_REC_FIFO_SINGLE);   // single FIFO mode
    spcm_dwSetParam_i32 (hCardAD, SPC_TIMEOUT,        5000);                  // timeout 5 s
    spcm_dwSetParam_i32 (hCardAD, SPC_TRIG_ORMASK,    SPC_TMASK_SOFTWARE);    // trigger set to software
    spcm_dwSetParam_i32 (hCardAD, SPC_TRIG_ANDMASK,   0);                     // ...
    spcm_dwSetParam_i32 (hCardAD, SPC_CLOCKMODE,      SPC_CM_INTPLL);         // clock mode internal PLL
    spcm_dwSetParam_i32 (hCardAD, SPC_CLOCKOUT,       1);                     // clock output will be used as reference by DA card

    int lIR = 1000;                                                         // +/- 1 Volt
    spcm_dwSetParam_i32 (hCardAD, SPC_AMP0,           lIR);

    int32 lResolutionAD = 0;
    spcm_dwGetParam_i32 (hCardAD, SPC_MIINST_BITSPERSAMPLE, &lResolutionAD);

    //spcm_dwSetParam_i64 (hCardAD, SPC_SPECIALCLOCK, 1);
    int64 llSamplerate = MEGA(125);
    spcm_dwSetParam_i64 (hCardAD, SPC_SAMPLERATE, llSamplerate);
    spcm_dwGetParam_i64 (hCardAD, SPC_SAMPLERATE, &llSamplerate);
    printf ("Samplerate AD: %lld\n", llSamplerate);

    spcm_dwSetParam_i32 (hCardAD, SPC_M2CMD,           M2CMD_CARD_WRITESETUP); // write setup to activate clock output


    int32 lADClkOutput = 0;
    spcm_dwGetParam_i32 (hCardAD, SPC_CLOCKOUTFREQUENCY, &lADClkOutput);
    // AD Card Setup finished
    // ------------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // DA Card Setup
    // ----- do a simple FIFO setup for 66xx -----
    spcm_dwSetParam_i32 (hCardDA, SPC_CHENABLE,       (0x1 << lNumCh) - 1);   // enable channel(s)
    spcm_dwSetParam_i32 (hCardDA, SPC_CARDMODE,       SPC_REP_FIFO_SINGLE);   // single FIFO mode
    spcm_dwSetParam_i64 (hCardDA, SPC_SEGMENTSIZE,    1024);
    spcm_dwSetParam_i64 (hCardDA, SPC_LOOPS,          0);                     // forever
    spcm_dwSetParam_i32 (hCardDA, SPC_TRIG_ORMASK,    SPC_TMASK_EXT0);        // trigger set to EXT0 (do not connect! just as a dummy)
    spcm_dwSetParam_i32 (hCardDA, SPC_TRIG_EXT0_LEVEL0, 1500);                // a level that won't trigger accidentally. Card will be FORCETRIGGERED later on
    spcm_dwSetParam_i32 (hCardDA, SPC_TRIG_ANDMASK,   0);                     // ...
    spcm_dwSetParam_i32 (hCardDA, SPC_CLOCKMODE,      SPC_CM_EXTREFCLOCK);    // use clock signal from AD card
    spcm_dwSetParam_i32 (hCardDA, SPC_REFERENCECLOCK, lADClkOutput);
    spcm_dwSetParam_i64 (hCardDA, SPC_SAMPLERATE,     llSamplerate);
    spcm_dwSetParam_i32 (hCardDA, SPC_TIMEOUT,        5*1000);

    // ----- if resolutions of AD card and DA card do not match we use the data conversion feature of the DA card to get similar output levels -----
    int32 lResolutionDA = 0;
    spcm_dwGetParam_i32 (hCardDA, SPC_MIINST_BITSPERSAMPLE, &lResolutionDA);
    switch (lResolutionDA)
        {
        case 16:
            {
            if (lResolutionAD == 14)
                spcm_dwSetParam_i32 (hCardDA, SPC_DATACONVERSION, SPCM_DC_14BIT_TO_16BIT);
            else if (lResolutionAD == 12)
                spcm_dwSetParam_i32 (hCardDA, SPC_DATACONVERSION, SPCM_DC_12BIT_TO_16BIT);
            break;
            }
        case 14:
            {
            if (lResolutionAD == 16)
                spcm_dwSetParam_i32 (hCardDA, SPC_DATACONVERSION, SPCM_DC_16BIT_TO_14BIT);
            else if (lResolutionAD == 12)
                spcm_dwSetParam_i32 (hCardDA, SPC_DATACONVERSION, SPCM_DC_12BIT_TO_14BIT);
            break;
            }
        }

    int32 lMaxOutputLevel = lIR / 2; // we acquire data without 50Ohm termination, but setting for DA card is "into 50 Ohm", so to get same level we divide by 2
    for (int lChIdx = 0; lChIdx < lNumCh; ++lChIdx)
        {
        spcm_dwSetParam_i32 (hCardDA, SPC_ENABLEOUT0 + lChIdx * (SPC_ENABLEOUT1 - SPC_ENABLEOUT0), 1);
        spcm_dwSetParam_i32 (hCardDA, SPC_AMP0       + lChIdx * (SPC_AMP1        - SPC_AMP0),      lMaxOutputLevel);
        }
    spcm_dwGetParam_i64 (hCardDA, SPC_SAMPLERATE, &llSamplerate);
    printf ("Samplerate DA: %lld\n", llSamplerate);

    // DA Card Setup finished
    // ------------------------------------------------------------------------




    // ----- DMA BUFFER SETUP -----
    // ----- get buffer on GPU that will be used as target for RDMA transfer from AD card-----
    int lCUDADeviceIdx = 0;         // index of used CUDA device
    void* pvADDMABuffer_gpu = pvGetRDMABuffer (lCUDADeviceIdx, lBufferSize);
    if (pvADDMABuffer_gpu == NULL)
        {
        printf ("Failed to allocate DMA buffer for AD card\n");

        spcm_vClose (hCardAD);
        spcm_vClose (hCardDA);
        return EXIT_FAILURE;
        }

    // setup DMA transfer from Spectrum card to GPU
    spcm_dwDefTransfer_i64 (hCardAD, SPCM_BUF_DATA, SPCM_DIR_CARDTOGPU, lNotifySize, pvADDMABuffer_gpu, 0, lBufferSize);


    // ----- get buffer on GPU that will be used as source for RDMA transfer to DA card-----
    void* pvDADMABuffer_gpu = pvGetRDMABuffer (lCUDADeviceIdx, lBufferSize);
    if (pvDADMABuffer_gpu == NULL)
        {
        printf ("Failed to allocate DMA buffer for DA card\n");

        spcm_vClose (hCardAD);
        spcm_vClose (hCardDA);

        // free allocated CUDA buffers on GPU
        cudaFree (pvADDMABuffer_gpu);
        return EXIT_FAILURE;
        }

    // setup DMA transfer from GPU to Spectrum card
    spcm_dwDefTransfer_i64 (hCardDA, SPCM_BUF_DATA, SPCM_DIR_GPUTOCARD, lNotifySize, pvDADMABuffer_gpu, 0, lBufferSize);

    // start DMA for DA card. Will wait for trigger until forced later on
    dwError = spcm_dwSetParam_i32 (hCardDA, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER | M2CMD_DATA_STARTDMA);


    // ----- start AD card -----
    time_t startTime = time (NULL);
    dwError = spcm_dwSetParam_i32 (hCardAD, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER | M2CMD_DATA_STARTDMA);
    if (dwError != ERR_OK)
        {
        // cleanup
        spcm_dwGetErrorInfo_i32 (hCardAD, NULL, NULL, szErrorTextBuffer);
        printf ("%s\n", szErrorTextBuffer);

        spcm_vClose (hCardAD);
        spcm_vClose (hCardDA);

        // free allocated CUDA buffers on GPU
        cudaFree (pvADDMABuffer_gpu);
        cudaFree (pvDADMABuffer_gpu);

        return EXIT_FAILURE;
        }


    // run the FIFO mode and loop through the data
    // the control of the DMA transfer is the same as without RDMA.
    // The difference is that the driver reports a certain amount of data as available to the user,
    // but the data has been transfered into the GPU memory.
    bool bDAStarted = false;
    while (qwTotalMem < qwToTransfer)
        {
        if ((dwError = spcm_dwSetParam_i32 (hCardAD, SPC_M2CMD, M2CMD_DATA_WAITDMA)) != ERR_OK)
            {
            if (dwError == ERR_TIMEOUT)
                printf ("\n... AD Timeout\n");
            else
                printf ("\n... Error: %d\n", dwError);
            break;
            }

        else
            {
            spcm_dwGetParam_i32 (hCardAD, SPC_M2STATUS,             &lStatus);
            spcm_dwGetParam_i32 (hCardAD, SPC_DATA_AVAIL_USER_LEN,  &lAvailUser);
            spcm_dwGetParam_i32 (hCardAD, SPC_DATA_AVAIL_USER_POS,  &lPCPosAD);

            if (lAvailUser >= lNotifySize)
                {
                qwTotalMem += lNotifySize;

                lPCPosDA = lPCPosAD; // we run with the same speed, so data positions in the buffer will be the same

                time_t curTime = time (NULL);
                printf ("\rStat:%08x Pos:%08x Avail:%08x Total:%.2f MiB @ %.2lf MiB/s", lStatus, lPCPosAD, lAvailUser, (double) (int64) qwTotalMem / MEGA_B(1), (((double)qwTotalMem) / MEGA_B(1)) / (curTime - startTime));

                // start kernel on the GPU to process the transfered data
                // after the kernel has finished the processed data is in the DMA buffer of the DA card
                const int lThreadsPerBlock = 1024;
                CudaKernelInvert <<< (lNotifySize / sizeof (int16)) / lThreadsPerBlock, lThreadsPerBlock >>> ((int16*)((char*)pvADDMABuffer_gpu + lPCPosAD), (int16*)((char*)pvDADMABuffer_gpu + lPCPosDA), lNotifySize);

                // without a call to cudaMemcpy we need to wait for the kernel to finish ourselves
                cudaDeviceSynchronize ();

                // mark memory as free for the AD card, and as filled for the DA card
                spcm_dwSetParam_i32 (hCardAD, SPC_DATA_AVAIL_CARD_LEN,  lNotifySize);
                spcm_dwSetParam_i32 (hCardDA, SPC_DATA_AVAIL_CARD_LEN,  lNotifySize);

                if (!bDAStarted)
                    {
                    dwError = spcm_dwSetParam_i32 (hCardDA, SPC_M2CMD, M2CMD_CARD_FORCETRIGGER);
                    bDAStarted = true;
                    }
                }

            }
        }

    // send the stop command
    dwError = spcm_dwSetParam_i32 (hCardAD, SPC_M2CMD, M2CMD_CARD_STOP | M2CMD_DATA_STOPDMA);
    dwError = spcm_dwSetParam_i32 (hCardDA, SPC_M2CMD, M2CMD_CARD_STOP | M2CMD_DATA_STOPDMA);

    // clean up
    printf ("\nFinished...\n");

    spcm_vClose (hCardAD);
    spcm_vClose (hCardDA);

    // free CUDA buffers on GPU
    cudaFree (pvADDMABuffer_gpu);
    cudaFree (pvDADMABuffer_gpu);

    return EXIT_SUCCESS;
    }

