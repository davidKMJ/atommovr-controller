/*
**************************************************************************

rdma_fifo_digital_down_conversion_iq.cu          (c) Spectrum GmbH, 2/2024

**************************************************************************

Example for all M4i/M4x/M2p/M5i analog acquisition cards. 

Data is transfered in FIFO mode from card to GPU. CUDA kernels are used
to downconvert the signal to the baseband by multiplication with a generated
carrier signal. The result is then FIR-filtered and decimated, for both I and Q.
  
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
#include <fstream>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <aio.h>

#include "../common/spcm_cuda_common.h"

// ----- CUDA include -----
#   include <cuda_runtime.h>

// CUDA-C includes
#   include <cuda.h>


// ----- CUDA kernels -----

// ----- generate a sine and a cosine wave as an approximation of the carrier -----
__global__ void CudaKernelGenerateCarriers (int64 llSR_Hz, int64 llCarrierFrequency_Hz, float* afCarrierSin, float* afCarrierCos)
    {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    double dFactor = (2.0 * M_PI * i) / (static_cast < double > (llSR_Hz) / llCarrierFrequency_Hz);
    afCarrierSin[i] = sin (dFactor);
    afCarrierCos[i] = cos (dFactor);
    }

// ----- merge mixing, filtering, down-conversion into one kernel because it is faster than having three separate kernels -----
__global__ void CudaKernelMultiplyFIRAndDecimateTo8BitRAW (const float* afCarrier, const int8* pcSignal, int lNumCh, int lChIdx, float fVoltPerLSB, float* pfSource, int lSourceLen, const float* pfCoefficients, int lNumCoefficients, int lDecimationFactor, int8* pcDest)
    {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // copy coefficients to memory (shared for each block)
    __shared__ float sfCoefficients[32]; // ! lNumCoefficients should be smaller than the array size
    if (threadIdx.x < lNumCoefficients)
        sfCoefficients[threadIdx.x] = pfCoefficients[threadIdx.x];

    // mix with carrier
    pfSource[i] = afCarrier[i] * (fVoltPerLSB * pcSignal[i*lNumCh + lChIdx]);

    // wait until all threads have finished mixing because the following filter loops needs results from multiple threads
    __syncthreads();

    // filter
    float fSum = 0.0f;
    // FIR
    int lDestLen = (lSourceLen - lNumCoefficients); // filtered signal is shorter than original
    if (i < lDestLen)
        {
        for (int j = 0; j < lNumCoefficients; j++)
            {
            fSum += sfCoefficients[j] * pfSource[i + lNumCoefficients - j - 1];
            }
        }

    // decimate
    int j = i % lDecimationFactor;
    if (j == 0)
        pcDest[i / lDecimationFactor] = (int8)(fSum / fVoltPerLSB);
    }


// filter to 1/4 of samplerate
//const float afFIRLowPass_1_4[] = {
//2.5354e-03, 2.4823e-03, -1.6326e-04, -8.8185e-03, -2.1019e-02, -2.3837e-02, 6.6347e-04, 6.1110e-02, 1.4459e-01, 2.1845e-01, 2.4802e-01, 2.1845e-01, 1.4459e-01, 6.1110e-02, 6.6347e-04, -2.3837e-02, -2.1019e-02, -8.8185e-03, -1.6326e-04, 2.4823e-03, 2.5354e-03 };
//const int lNumCoefficients = sizeof (afFIRLowPass_1_4) / sizeof (afFIRLowPass_1_4[0]);

// filter to 1/4 of samplerate
// less coefficients => faster but uglier
const float afFIRLowPass_1_4_short[] = {
0.000092797, 0.019311595, 0.102082533, 0.230618167, 0.295789818, 0.230618167, 0.102082533, 0.019311595, 0.000092797
};
const int lNumCoefficients = sizeof (afFIRLowPass_1_4_short) / sizeof (afFIRLowPass_1_4_short[0]);


const float* afFIRLowPass = afFIRLowPass_1_4_short;

void vWaitForAsyncIO (void* pvData)
    {
    struct aiocb* apstAIOControlForSuspend[1] = {};
    apstAIOControlForSuspend[0] = (struct aiocb*)pvData;
    aio_suspend (apstAIOControlForSuspend, 1, NULL);
    }

void vWriteToDisk (void* pvData)
    {
    aio_write ((struct aiocb*)pvData);
    }



/*
**************************************************************************
main 
**************************************************************************
*/

int main (int argc, char* argv[])
    {
    drv_handle  hCard = 0;
    int32       lCardType = 0, lFncType, lNumCh = 2, lBytesPerSample;
    char        szErrorTextBuffer[ERRORTEXTLEN];
    uint32      dwError;
    int32       lStatus, lAvailUser, lPCPos;
    uint64      qwTotalMem = 0;
    int         lCardIdx = 0;

    int64 llSamplerate_Hz = MEGA(3200);
    int32 lNumDDCPerCh = 2; // I and Q
    bool bWriteToDisk = true;

    int64 llCarrierFrequency_Hz = MEGA(700); // change this to match your signal
    int lDecimationFactor = 3;

    // settings for the FIFO mode buffer handling
    int32       lNotifySize =   MEGA_B(15); // 15 to have a multiple of the decimation factor
    int32       lBufferSize =   8*lNotifySize;

    cudaDeviceProp stProp;
    cudaGetDeviceProperties (&stProp, 0);
    printf ("Prop MaxThreadsPerBlock: %d\n", stProp.maxThreadsPerBlock);
    printf ("Prop MaxThreadsDim[0]: %d\n", stProp.maxThreadsDim[0]);
    printf ("Prop MaxThreadsDim[1]: %d\n", stProp.maxThreadsDim[1]);
    printf ("Prop MaxThreadsDim[2]: %d\n", stProp.maxThreadsDim[2]);
    printf ("Prop SharedMemPerBlock: %lu\n", stProp.sharedMemPerBlock);


    // ----- check command line options
    for (int lArg = 1; lArg < argc; ++lArg)
        {
        if (strcmp (argv[lArg], "--sr") == 0)
            {
            lArg++;
            llSamplerate_Hz = MEGA(atoll (argv[lArg]));
            }
        else if (strcmp (argv[lArg], "--ch") == 0)
            {
            lArg++;
            lNumCh = atoi (argv[lArg]);
            }
        else if (strcmp (argv[lArg], "--ddcperch") == 0)
            {
            lArg++;
            lNumDDCPerCh = atoi (argv[lArg]);
            }
        else if (strcmp (argv[lArg], "--writetodisk") == 0)
            {
            lArg++;
            bWriteToDisk = (atoi (argv[lArg]) != 0);
            }
        else if (strcmp (argv[lArg], "--card") == 0)
            {
            lArg++;
            lCardIdx = atoi (argv[lArg]);
            }
        else if (strcmp (argv[lArg], "--help") == 0)
            {
            printf ("Available options:\n");
            printf ("  --card:          index of the used card, default: 0\n");
            printf ("  --sr:            set the sampling rate in MS/s\n");
            printf ("  --ch:            set the number of active channels\n");
            printf ("  --ddcperch:      set the number of DDC per channel (I/Q)\n");
            printf ("  --writetodisk:   1: write data to disk, 0: no write\n");
            printf ("\n");
            return EXIT_SUCCESS;
            }
        }
    // -----


    // ------------------------------------------------------------------------
    // CARD SETUP

    // ----- open Spectrum card -----
    char szCardName[12];
    sprintf (szCardName, "/dev/spcm%d", lCardIdx);
    hCard = spcm_hOpen (szCardName);
    if (!hCard)
        {
        spcm_dwGetErrorInfo_i32 (NULL, NULL, NULL, szErrorTextBuffer);
        printf ("Could not open card %s: %s\n", szCardName, szErrorTextBuffer);
        return 0;
        }

    // read some infos about the card
    spcm_dwGetParam_i32 (hCard, SPC_PCITYP,                &lCardType);
    spcm_dwGetParam_i32 (hCard, SPC_FNCTYPE,               &lFncType);
    spcm_dwGetParam_i32 (hCard, SPC_MIINST_CHPERMODULE,    &lNumCh);
    spcm_dwGetParam_i32 (hCard, SPC_MIINST_BYTESPERSAMPLE, &lBytesPerSample);

    // read card type name
    char szCardType[20] = {};
    spcm_dwGetParam_ptr (hCard, SPC_PCITYP, szCardType, sizeof (szCardType));

    // this example requires an A/D card. The older M2i and M3i series do not work.
    switch (lFncType)
        {
        case SPCM_TYPE_AI:  
            {
            switch (lCardType & TYP_SERIESMASK)
                {
                default:
                    printf ("Found: %s\n", szCardType);
                    break;
                case TYP_M2ISERIES:
                case TYP_M2IEXPSERIES:
                case TYP_M3ISERIES:
                case TYP_M3IEXPSERIES:
                    printf ("Card: %s not supported by example\n", szCardType);
                    return EXIT_FAILURE;
                }
            break;
            }

        default:
            printf ("Card: %s not supported by example\n", szCardType);
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

    spcm_dwSetParam_i32 (hCard, SPC_DATACONVERSION, SPCM_DC_12BIT_TO_8BIT); // 8bit mode
    lBytesPerSample = 1;

    int32 lIR_mV = 1000;                                                    // +/- 1 Volt
    spcm_dwSetParam_i32 (hCard, SPC_AMP0,           lIR_mV);

    int32 lMaxADCValue = 0;
    spcm_dwGetParam_i32 (hCard, SPC_MIINST_MAXADCVALUE, &lMaxADCValue);

    //spcm_dwSetParam_i64 (hCard, SPC_SPECIALCLOCK, 1);
    spcm_dwSetParam_i64 (hCard, SPC_SAMPLERATE, llSamplerate_Hz);
    spcm_dwGetParam_i64 (hCard, SPC_SAMPLERATE, &llSamplerate_Hz);
    printf ("Used samplerate: %lld\n", llSamplerate_Hz);
    // Card Setup finished
    // ------------------------------------------------------------------------


    // ----- DMA BUFFER SETUP -----
    // ----- get buffer on GPU that will be used as target for RDMA transfer -----
    int lCUDADeviceIdx = 0;         // index of used CUDA device
    int16* pnDMABuffer_gpu = (int16*)pvGetRDMABuffer (lCUDADeviceIdx, lBufferSize);
    if (pnDMABuffer_gpu == NULL)
        {
        spcm_vClose (hCard);
        return EXIT_FAILURE;
        }

    // setup DMA transfer from Spectrum card to GPU
    spcm_dwDefTransfer_i64 (hCard, SPCM_BUF_DATA, SPCM_DIR_CARDTOGPU, lNotifySize, pnDMABuffer_gpu, 0, lBufferSize);


    int lBlocksize_S = lNotifySize / (lBytesPerSample * lNumCh);

#define NUM_DDCS (2*2) // max two channels, I and Q on each channel
    float* afCoefficients_gpu = NULL;
    float* afCarrierSin_gpu   = NULL;
    float* afCarrierCos_gpu   = NULL;
    float* aafMixed_gpu[NUM_DDCS]     = { NULL, NULL, NULL, NULL };
    int8*  aacDecimated_gpu[NUM_DDCS] = { NULL, NULL, NULL, NULL };
    int8*  aacBuffer_host[NUM_DDCS]   = { NULL, NULL, NULL, NULL };

    // copy FIR coefficients to GPU memory
    cudaError_t eCudaErr = cudaMalloc (&afCoefficients_gpu, lNumCoefficients * sizeof (float));
    cudaMemcpy (afCoefficients_gpu, afFIRLowPass, lNumCoefficients * sizeof (float), cudaMemcpyHostToDevice);

    const int lNumDecimated = lBlocksize_S / lDecimationFactor;

    // ----- create a buffer on the GPU for the carrier -----
    eCudaErr = cudaMalloc (&afCarrierSin_gpu, lBlocksize_S * sizeof (float));
    if (eCudaErr == cudaSuccess)
        eCudaErr = cudaMalloc (&afCarrierCos_gpu, lBlocksize_S * sizeof (float));

    if (eCudaErr == cudaSuccess)
        {
        // ----- create a buffer on the GPU for the mixed signal -----
        for (int lChIdx = 0; lChIdx < lNumCh; ++lChIdx)
            {
            for (int lDDCIdx = 0; lDDCIdx < lNumDDCPerCh; ++lDDCIdx)
                {
                int lIdx = lChIdx * lNumDDCPerCh + lDDCIdx;

                // ----- create a buffer for the signal mixed with the carrier on the GPU -----
                eCudaErr = cudaMalloc (&(aafMixed_gpu[lIdx]), lBlocksize_S * sizeof (float));

                // ----- create a buffer on the GPU for the decimated signal -----
                if (eCudaErr == cudaSuccess)
                    eCudaErr = cudaMalloc (&(aacDecimated_gpu[lIdx]), lNumDecimated * sizeof (int8));

                // ----- create a buffer on the host side. This will be used to transfer the processed data from the GPU -----
                if (eCudaErr == cudaSuccess)
                    eCudaErr = cudaMallocHost (&(aacBuffer_host[lIdx]), lNumDecimated * sizeof (int8));
                }
            }
        }

    if (eCudaErr != cudaSuccess)
        {
        // cleanup
        printf ("ERROR in cudaMallocHost(): %s\n", cudaGetErrorString(eCudaErr));

        spcm_vClose (hCard);

        // free allocated CUDA buffers on GPU
        cudaFree (pnDMABuffer_gpu);
        cudaFree (afCoefficients_gpu);
        cudaFree (afCarrierSin_gpu);
        cudaFree (afCarrierCos_gpu);
        for (int lIdx = 0; lIdx < lNumCh * lNumDDCPerCh; ++lIdx)
            {
            cudaFree (aafMixed_gpu[lIdx]);
            cudaFree (aacDecimated_gpu[lIdx]);

            cudaFreeHost (aacBuffer_host[lIdx]);
            }

        return EXIT_FAILURE;
        }

    // ----- create carrier once -----
    const int lThreadsPerBlock = 1024; // 1024 is the max number of threads per block, but using more blocks with less threads might be beneficial
    CudaKernelGenerateCarriers <<< lBlocksize_S / lThreadsPerBlock, lThreadsPerBlock >>> (llSamplerate_Hz, llCarrierFrequency_Hz, afCarrierSin_gpu, afCarrierCos_gpu);
    
    // ----- create CUDA streams -----
    cudaStream_t ahStreams[NUM_DDCS];
    int hFileForStream[NUM_DDCS] = { -1, -1, -1, -1 };
    for (int lChIdx = 0; lChIdx < lNumCh; ++lChIdx)
        {
        for (int lDDCIdx = 0; lDDCIdx < lNumDDCPerCh; ++lDDCIdx)
            {
            const int lCudaStreamIdx = lChIdx * lNumDDCPerCh + lDDCIdx;
            cudaStreamCreate (ahStreams + lCudaStreamIdx);

            if (bWriteToDisk)
                {
                // the down-converted result of each CUDA stream can be stored to a file
                char szFilename[64];
                sprintf (szFilename, "/mnt/highpoint-raid/CudaDDC-Ch%d-%s.bin", lChIdx, (lDDCIdx == 0? "I" : "Q"));
                hFileForStream[lCudaStreamIdx] = open (szFilename, O_WRONLY | O_CREAT | O_DIRECT, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);
                if (hFileForStream[lCudaStreamIdx] <= 0)
                    {
                    printf ("Opening %s failed\n", szFilename);

                    // close files that have been opened up to now
                    for (int j = 0; j < lCudaStreamIdx; ++j)
                        close (hFileForStream[j]);

                    spcm_vClose (hCard);

                    // free allocated CUDA buffers on host and GPU
                    cudaFree (pnDMABuffer_gpu);
                    cudaFree (afCoefficients_gpu);
                    cudaFree (afCarrierSin_gpu);
                    cudaFree (afCarrierCos_gpu);
                    for (int lIdx = 0; lIdx < lNumCh * lNumDDCPerCh; ++lIdx)
                        {
                        cudaFree (aafMixed_gpu[lIdx]);
                        cudaFree (aacDecimated_gpu[lIdx]);

                        cudaFreeHost (aacBuffer_host[lIdx]);
                        }

                    return EXIT_FAILURE;
                    }
                }
            }
        }

    float fVoltPerLSB = (lIR_mV / 1000.) / lMaxADCValue;

    printf ("Decimation factor: %d  Write-to-SSD blocksize: %llu kiB\n", lDecimationFactor, (lNumDecimated * sizeof (int8)) / KILO_B(1));
    printf ("Num FIR coefficients: %d\n", lNumCoefficients);
    printf ("Num DDC per Ch: %d\n", lNumDDCPerCh);
    printf ("WriteToDisk: %s\n", bWriteToDisk? "true" : "false");
    printf ("Num Ch: %d\n", lNumCh);
    printf ("Notifysize: %d bytes\n", lNotifySize);
    printf ("Blocksize: %d samples\n", lBlocksize_S);
    printf ("Volt/LSB: %f\n", fVoltPerLSB);

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
        cudaFree (pnDMABuffer_gpu);
        cudaFree (afCoefficients_gpu);
        cudaFree (afCarrierSin_gpu);
        cudaFree (afCarrierCos_gpu);
        for (int lIdx = 0; lIdx < lNumCh * lNumDDCPerCh; ++lIdx)
            {
            cudaFree (aafMixed_gpu[lIdx]);
            cudaFree (aacDecimated_gpu[lIdx]);

            cudaFreeHost (aacBuffer_host[lIdx]);

            // close files
            if (bWriteToDisk)
                close (hFileForStream[lIdx]);
            }

        return EXIT_FAILURE;
        }


    // some control variables for Async disk IO
    struct aiocb astAIO[NUM_DDCS] = {};
    uint64 aqwOffset[NUM_DDCS] = {};
    int32 lFillsize = 0;
    int lAIO = 0;

    // run the FIFO mode and loop through the data
    // the control of the DMA transfer is the same as without RDMA.
    // The difference is that the driver reports a certain amount of data as available to the user,
    // but the data has been transfered into the GPU memory.
    while (true/*qwTotalMem < qwToTransfer*/)
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
            // get card status and some info about available data
            spcm_dwGetParam_i32 (hCard, SPC_M2STATUS,             &lStatus);
            spcm_dwGetParam_i32 (hCard, SPC_DATA_AVAIL_USER_LEN,  &lAvailUser);
            spcm_dwGetParam_i32 (hCard, SPC_DATA_AVAIL_USER_POS,  &lPCPos);
            spcm_dwGetParam_i32 (hCard, SPC_FILLSIZEPROMILLE,     &lFillsize);

            if (lAvailUser >= lNotifySize)
                {
                qwTotalMem += lNotifySize;

                time_t curTime = time (NULL);
                printf ("\rStat:%08x Fill:%4d/1000 Avail:%08x Total:%.2f MiB @ %.2lf MiB/s Datei[0]: %llu MiB", lStatus, lFillsize, lAvailUser, (double) (int64) qwTotalMem / MEGA_B(1), (((double)qwTotalMem) / MEGA_B(1)) / (curTime - startTime), (uint64)astAIO[0].aio_offset / MEGA_B(1));

                // this is the point to do anything with the data on the GPU


                // one CUDA stream for each channel
                for (int lChIdx = 0; lChIdx < lNumCh; ++lChIdx)
                    {
                    // and one stream for each DDC we run on the channel (I, Q)
                    for (int lDDCIdx = 0; lDDCIdx < lNumDDCPerCh; ++lDDCIdx)
                        {
                        const int lCudaStreamIdx = lChIdx * lNumDDCPerCh + lDDCIdx;

                        // multiply the signal from the digitizer with the carrier, apply an FIR filter, decimate, and return the resulting signal as 8bit RAW values
                        CudaKernelMultiplyFIRAndDecimateTo8BitRAW <<< lBlocksize_S / lThreadsPerBlock, lThreadsPerBlock, 0, ahStreams[lCudaStreamIdx] >>> ((lDDCIdx == 0? afCarrierSin_gpu : afCarrierCos_gpu), (int8*)((char*)pnDMABuffer_gpu + lPCPos), lNumCh, lChIdx, fVoltPerLSB, aafMixed_gpu[lCudaStreamIdx], lBlocksize_S, afCoefficients_gpu, lNumCoefficients, lDecimationFactor, aacDecimated_gpu[lCudaStreamIdx]);

                        if (bWriteToDisk)
                            {
                            lAIO = lCudaStreamIdx;

                            // from second processed data block onwards we wait until the previous ASyncIO has finished
                            if (qwTotalMem > lNotifySize)
                                {
                                cudaLaunchHostFunc (ahStreams[lCudaStreamIdx], vWaitForAsyncIO, astAIO + lAIO);
                                }
                            }

                        // after kernel has finished we copy processed data from GPU to host
                        eCudaErr = cudaMemcpyAsync (aacBuffer_host[lAIO], aacDecimated_gpu[lCudaStreamIdx], lNumDecimated * sizeof (int8), cudaMemcpyDeviceToHost, ahStreams[lCudaStreamIdx]);
                        if (eCudaErr != cudaSuccess)
                            {
                            printf ("ERROR in cudaMemcpy(): %s\n", cudaGetErrorString(eCudaErr));
                            break;
                            }

                        // now the processed data is in the host memory
                        if (bWriteToDisk)
                            {
                            astAIO[lAIO].aio_fildes = hFileForStream[lCudaStreamIdx];
                            astAIO[lAIO].aio_buf = aacBuffer_host[lAIO];
                            astAIO[lAIO].aio_nbytes = lNumDecimated * sizeof (int8);
                            astAIO[lAIO].aio_offset = aqwOffset[lCudaStreamIdx];
                            aqwOffset[lCudaStreamIdx] += astAIO[lAIO].aio_nbytes;
                            cudaLaunchHostFunc (ahStreams[lCudaStreamIdx], vWriteToDisk, astAIO + lAIO);
                            }


                        } // for (lDDCIdx
                    }// for (lChIdx

                // wait until all the queued kernels above have finished
                eCudaErr = cudaDeviceSynchronize ();
                if (eCudaErr != cudaSuccess)
                    {
                    printf ("ERROR in cudaDeviceSynchronize(): %s\n", cudaGetErrorString(eCudaErr));
                    break;
                    }

                // mark the data in the DMA buffer as freed
                spcm_dwSetParam_i32 (hCard, SPC_DATA_AVAIL_CARD_LEN, lNotifySize);
                } // if (lAvailUser
            }
        } // while (true

    // stop the card and the DMA transfer
    dwError = spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_CARD_STOP | M2CMD_DATA_STOPDMA);

    // clean up
    printf ("\nFinished...\n");

    spcm_vClose (hCard);

    // ----- close all files -----
    if (bWriteToDisk)
        {
        for (int lChIdx = 0; lChIdx < lNumCh; ++lChIdx)
            {
            for (int lDDCIdx = 0; lDDCIdx < lNumDDCPerCh; ++lDDCIdx)
                {
                const int lCudaStreamIdx = lChIdx * lNumDDCPerCh + lDDCIdx;
                close (hFileForStream[lCudaStreamIdx]);
                }
            }
        }


    // ----- free CUDA buffers on GPU and host -----
    cudaFree (pnDMABuffer_gpu);
    cudaFree (afCoefficients_gpu);
    cudaFree (afCarrierSin_gpu);
    cudaFree (afCarrierCos_gpu);
    for (int lIdx = 0; lIdx < lNumCh * lNumDDCPerCh; ++lIdx)
        {
        cudaFree (aafMixed_gpu[lIdx]);
        cudaFree (aacDecimated_gpu[lIdx]);

        cudaFreeHost (aacBuffer_host[lIdx]);
        }

    return EXIT_SUCCESS;
    }

