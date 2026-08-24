/*
**************************************************************************

rdma_fifo_digital_down_conversion.cu            (c) Spectrum GmbH , 7/2022

**************************************************************************

Example for all M4i/M4x/M2p/M5i analog acquisition cards. 

Data is transfered in FIFO mode from card to GPU. CUDA kernels are used
to downconvert the signal to the baseband by multiplication with a generated
carrier signal. The result is then FIR-filtered and decimated.
  
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

// ----- generate a sine wave as an approximation of the carrier -----
__global__ void CudaKernelGenerateCarriers (int64 llSR_Hz, int64 llCarrierFrequency_Hz, float* afCarrierSin, float* afCarrierCos)
    {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    afCarrierSin[i] = sin ((2.0 * M_PI * i) / (static_cast < double > (llSR_Hz) / llCarrierFrequency_Hz));
    afCarrierCos[i] = cos ((2.0 * M_PI * i) / (static_cast < double > (llSR_Hz) / llCarrierFrequency_Hz));
    }

// ----- multiply (mix) the carrier and the signal -----
__global__ void CudaKernelMultiply (const float* afCarrier, const short* pnSignal, float fIR_V, int lMaxADCValue, float* afMixed)
    {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    //afMixed[i] = afCarrier[i] * (1.0 * pnSignal[i]) / 32768;
    afMixed[i] = afCarrier[i] * (fIR_V * pnSignal[i]) / lMaxADCValue;
    }

// ----- Finite Impulse Response (FIR) filter. -----
__global__ void CudaKernelFIR (const float* pfSource, int lSourceLen, const float* pfCoefficients, int lNumCoefficients, float* pfDest)
    {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    float fSum = 0.0f;

    int lDestLen = (lSourceLen - lNumCoefficients); // filtered signal is shorter than original
    if (i < lDestLen)
        {
        for (int j = 0; j < lNumCoefficients; j++)
            {
            fSum += pfCoefficients[j] * pfSource[i + lNumCoefficients - j - 1];
            }
        }

    pfDest[i] = fSum;
    }

__global__ void CudaKernelMovingAverageFilter (const float* pfSource, int lSourceLen, int lNumAvg, float* pfDest)
    {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // stupid implementation...
    float fSum = 0.0f;
    for (int j = 0; j < lNumAvg; ++j)
        {
        fSum += pfSource[i + j];
        }

    pfDest[i] = fSum / lNumAvg;
    }


// ----- pick each n-th sample from the mixed and filtered signal to reduce sampling rate -----
__global__ void CudaKernelDecimate (const float* afMixedAndFiltered, int lDecimationFactor, float* afDecimated)
    {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = i % lDecimationFactor;
    if (j == 0)
        afDecimated[i / lDecimationFactor] = afMixedAndFiltered[i];
    }

/*
**************************************************************************
szTypeToName: doing name translation
**************************************************************************
*/

// FIR coefficients for lowpass with 1MHz bandwidth
// change these to match your signal
const float afFIRLowPass1M[] = { 0.00179543, 0.00186866, 0.00203566, 0.00229904, 0.00266031, 0.00311986, 0.00367691, 0.00432952, 0.00507457, 0.00590782, 0.00682387, 0.00781628, 0.0088776, 0.00999941, 0.0111725, 0.0123868, 0.0136316, 0.0148959, 0.016168, 0.017436, 0.018688, 0.0199118, 0.0210957, 0.022228, 0.0232975, 0.0242936, 0.0252063, 0.0260263, 0.0267454, 0.0273563, 0.0278527, 0.0282295, 0.0284829, 0.0286102, 0.0286102, 0.0284829, 0.0282295, 0.0278527, 0.0273563, 0.0267454, 0.0260263, 0.0252063, 0.0242936, 0.0232975, 0.022228, 0.0210957, 0.0199118, 0.018688, 0.017436, 0.016168, 0.0148959, 0.0136316, 0.0123868, 0.0111725, 0.00999941, 0.0088776, 0.00781628, 0.00682387, 0.00590782, 0.00507457, 0.00432952, 0.00367691, 0.00311986, 0.00266031, 0.00229904, 0.00203566, 0.00186866, 0.00179543 };
//const int lNumCoefficients = 68;

const float afFIRLowPass50MS_1M[] = {
0.00492919, 0.00589972, 0.00857936, 0.0129181, 0.0187441, 0.0257717, 0.0336192, 0.0418338, 0.049922, 0.0573847, 0.0637525, 0.068619, 0.0716711, 0.072711, 0.0716711, 0.068619, 0.0637525, 0.0573847, 0.049922, 0.0418338, 0.0336192, 0.0257717, 0.0187441, 0.0129181, 0.00857936, 0.00589972, 0.00492919 };
const int lNumCoefficients = 27;

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


    int64 llCarrierFrequency_Hz = MEGA(39); // change this to match your signal
    int lDecimationFactor = 32*16;

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
    lNumCh = 1;
    spcm_dwSetParam_i32 (hCard, SPC_CHENABLE,       (0x1 << lNumCh) - 1);
    spcm_dwSetParam_i32 (hCard, SPC_PRETRIGGER,     1024);                  // 1k of pretrigger data at start of FIFO mode
    spcm_dwSetParam_i32 (hCard, SPC_CARDMODE,       SPC_REC_FIFO_SINGLE);   // single FIFO mode
    spcm_dwSetParam_i32 (hCard, SPC_TIMEOUT,        5000);                  // timeout 5 s
    spcm_dwSetParam_i32 (hCard, SPC_TRIG_ORMASK,    SPC_TMASK_SOFTWARE);    // trigger set to software
    spcm_dwSetParam_i32 (hCard, SPC_TRIG_ANDMASK,   0);                     // ...
    spcm_dwSetParam_i32 (hCard, SPC_CLOCKMODE,      SPC_CM_INTPLL);         // clock mode internal PLL
    spcm_dwSetParam_i32 (hCard, SPC_CLOCKOUT,       0);                     // no clock output

    int32 lIR_mV = 1000;                                                    // +/- 1 Volt
    spcm_dwSetParam_i32 (hCard, SPC_AMP0,           lIR_mV);

    int32 lMaxADCValue = 0;
    spcm_dwGetParam_i32 (hCard, SPC_MIINST_MAXADCVALUE, &lMaxADCValue);

    //spcm_dwSetParam_i64 (hCard, SPC_SPECIALCLOCK, 1);
    //int64 llSamplerate_Hz = MEGA(400);
    int64 llSamplerate_Hz = MEGA(6400);
    spcm_dwSetParam_i64 (hCard, SPC_SAMPLERATE, llSamplerate_Hz);
    spcm_dwGetParam_i64 (hCard, SPC_SAMPLERATE, &llSamplerate_Hz);
    printf ("Samplerate: %lld\n", llSamplerate_Hz);
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

    float* afCoefficients_gpu = NULL;
    float* afCarrierSin_gpu = NULL;
    float* afCarrierCos_gpu = NULL;
    float* afMixed_gpu = NULL;
    float* afMixedAndFiltered_gpu = NULL;
    float* afDecimated_gpu = NULL;
    float* afFIRFiltered_gpu = NULL;
    float* afBuffer_host = NULL;
    cudaError_t eCudaErr = cudaMalloc (&afCoefficients_gpu, lNumCoefficients * sizeof (float));
    //cudaMemcpy (afCoefficients_gpu, afFIRLowPass1M, lNumCoefficients * sizeof (float), cudaMemcpyHostToDevice);
    cudaMemcpy (afCoefficients_gpu, afFIRLowPass50MS_1M, lNumCoefficients * sizeof (float), cudaMemcpyHostToDevice);

    // ----- create a buffer on the GPU for the carrier -----
    eCudaErr = cudaMalloc (&afCarrierSin_gpu, lBlocksize_S * sizeof (float));
    if (eCudaErr == cudaSuccess)
        eCudaErr = cudaMalloc (&afCarrierCos_gpu, lBlocksize_S * sizeof (float));

    // ----- create a buffer on the GPU for the mixed signal -----
    if (eCudaErr == cudaSuccess)
        eCudaErr = cudaMalloc (&afMixed_gpu, lBlocksize_S * sizeof (float));

    // ----- create a buffer on the GPU for the filtered signal -----
    if (eCudaErr == cudaSuccess)
        eCudaErr = cudaMalloc (&afMixedAndFiltered_gpu, lBlocksize_S * sizeof (float));

    const int lNumDecimated = lBlocksize_S / lDecimationFactor;
    // ----- create a buffer on the GPU for the decimated signal -----
    if (eCudaErr == cudaSuccess)
        eCudaErr = cudaMalloc (&afDecimated_gpu, lNumDecimated * sizeof (float));

    if (eCudaErr == cudaSuccess)
        eCudaErr = cudaMalloc (&afFIRFiltered_gpu, lNumDecimated * sizeof (float));

    // ----- create a buffer on the host side. This will be used to transfer the processed data from the GPU -----
    if (eCudaErr == cudaSuccess)
        eCudaErr = cudaMallocHost (&afBuffer_host, lNumDecimated *lDecimationFactor * sizeof (float)); // TODO: decimation factor wieder raus, nur für debug
    if (eCudaErr != cudaSuccess)
        {
        // cleanup
        printf ("ERROR in cudaMallocHost(): %s\n", cudaGetErrorString(eCudaErr));

        spcm_vClose (hCard);

        // free allocated CUDA buffers on GPU
        cudaFree (pnDMABuffer_gpu);
        cudaFree (afCarrierSin_gpu);
        cudaFree (afCarrierCos_gpu);
        cudaFree (afMixed_gpu);
        cudaFree (afMixedAndFiltered_gpu);
        cudaFree (afDecimated_gpu);

        return EXIT_FAILURE;
        }

    // ----- create carrier once -----
    const int lThreadsPerBlock = 1024;
    CudaKernelGenerateCarriers <<< lBlocksize_S / lThreadsPerBlock, lThreadsPerBlock >>> (llSamplerate_Hz, llCarrierFrequency_Hz, afCarrierSin_gpu, afCarrierCos_gpu);
    

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
        cudaFreeHost (afBuffer_host);
        cudaFree (pnDMABuffer_gpu);
        cudaFree (afCarrierSin_gpu);
        cudaFree (afCarrierCos_gpu);
        cudaFree (afMixed_gpu);
        cudaFree (afMixedAndFiltered_gpu);
        cudaFree (afDecimated_gpu);

        return EXIT_FAILURE;
        }


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
            spcm_dwGetParam_i32 (hCard, SPC_M2STATUS,             &lStatus);
            spcm_dwGetParam_i32 (hCard, SPC_DATA_AVAIL_USER_LEN,  &lAvailUser);
            spcm_dwGetParam_i32 (hCard, SPC_DATA_AVAIL_USER_POS,  &lPCPos);

            if (lAvailUser >= lNotifySize)
                {
                qwTotalMem += lNotifySize;

                time_t curTime = time (NULL);
                printf ("\rStat:%08x Pos:%08x Avail:%08x Total:%.2f MiB @ %.2lf MiB/s", lStatus, lPCPos, lAvailUser, (double) (int64) qwTotalMem / MEGA_B(1), (((double)qwTotalMem) / MEGA_B(1)) / (curTime - startTime));

                // this is the point to do anything with the data on the GPU

#ifdef DEBUG
                // copy original data from GPU to host for debugging purposes
                eCudaErr = cudaMemcpy (afBuffer_host, pnDMABuffer_gpu, lBlocksize_S * sizeof (int16), cudaMemcpyDeviceToHost);
                std::ofstream oDebugFile ("1 - original.txt");
                for (int i = 0; i < lBlocksize_S; ++i)
                    {
                    if (i > 0)
                        oDebugFile << "\n";
                    oDebugFile << ((int16*)afBuffer_host)[i];
                    }
                oDebugFile.close ();
#endif

                // start kernels on the GPU to process the transfered data
                CudaKernelMultiply <<< lBlocksize_S / lThreadsPerBlock, lThreadsPerBlock >>> (afCarrierSin_gpu, (int16*)((char*)pnDMABuffer_gpu + lPCPos), lIR_mV / 1000., lMaxADCValue, afMixed_gpu);
#ifdef DEBUG
                eCudaErr = cudaMemcpy (afBuffer_host, afMixed_gpu, lBlocksize_S * sizeof (float), cudaMemcpyDeviceToHost);
                std::ofstream oDebugFile1 ("2 - after_multiply.txt");
                for (int i = 0; i < lBlocksize_S; ++i)
                    {
                    oDebugFile1 << afBuffer_host[i] << std::endl;
                    }
                oDebugFile1.close ();
#endif

               
                // the FIR filter takes too much time to run at 6.4 GS/s, so we first use a moving average and add a FIR after decimation
                // CudaKernelFIR <<< lBlocksize_S / lThreadsPerBlock, lThreadsPerBlock >>> (afMixed_gpu, lBlocksize_S, afCoefficients_gpu, lNumCoefficients, afMixedAndFiltered_gpu);
                int lNumAvg = 16;
                CudaKernelMovingAverageFilter <<< lBlocksize_S / lThreadsPerBlock, lThreadsPerBlock >>> (afMixed_gpu, lBlocksize_S, lNumAvg, afMixedAndFiltered_gpu);
#ifdef DEBUG
                eCudaErr = cudaMemcpy (afBuffer_host, afMixedAndFiltered_gpu, lBlocksize_S * sizeof (float), cudaMemcpyDeviceToHost);
                std::ofstream oDebugFile2 ("3 - after_movingaverage.txt");
                for (int i = 0; i < lBlocksize_S; ++i)
                    {
                    oDebugFile2 << afBuffer_host[i] << std::endl;
                    }
                oDebugFile2.close ();
#endif

                CudaKernelDecimate <<< lBlocksize_S / lThreadsPerBlock, lThreadsPerBlock >>> (afMixedAndFiltered_gpu, lDecimationFactor, afDecimated_gpu);
#ifdef DEBUG
                eCudaErr = cudaMemcpy (afBuffer_host, afDecimated_gpu, lNumDecimated * sizeof (float), cudaMemcpyDeviceToHost);
                std::ofstream oDebugFile3 ("4 - after_decimation.txt");
                for (int i = 0; i < lNumDecimated; ++i)
                    {
                    oDebugFile3 << afBuffer_host[i] << std::endl;
                    }
                oDebugFile3.close ();
#endif

                CudaKernelFIR <<< lNumDecimated / lThreadsPerBlock, lThreadsPerBlock >>> (afDecimated_gpu, lNumDecimated, afCoefficients_gpu, lNumCoefficients, afFIRFiltered_gpu);

                // after kernel has finished we copy processed data from GPU to host
                //eCudaErr = cudaMemcpy (afBuffer_host, afDecimated_gpu, lNumDecimated * sizeof (float), cudaMemcpyDeviceToHost);
                eCudaErr = cudaMemcpy (afBuffer_host, afFIRFiltered_gpu, lNumDecimated * sizeof (float), cudaMemcpyDeviceToHost);
                if (eCudaErr != cudaSuccess)
                    {
                    printf ("ERROR in cudaMemcpy(): %s\n", cudaGetErrorString(eCudaErr));
                    break;
                    }

                // now the processed data is in the host memory

#ifdef DEBUG
                // and we store it to disc
                std::ofstream oFile ("5 - ddc.txt");
                for (int i = 0; i < lNumDecimated; ++i)
                    {
                    oFile << afBuffer_host[i] << std::endl;
                    }
                oFile.close ();
#endif

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
    cudaFree (pnDMABuffer_gpu);
    cudaFree (afCarrierSin_gpu);
    cudaFree (afCarrierCos_gpu);
    cudaFree (afMixed_gpu);
    cudaFree (afMixedAndFiltered_gpu);
    cudaFree (afDecimated_gpu);

    // free CUDA buffer on host
    cudaFreeHost (afBuffer_host);

    return EXIT_SUCCESS;
    }

