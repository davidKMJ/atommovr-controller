/*
**************************************************************************

rdma_fifo_fft.cu                               (c) Spectrum GmbH , 06/2017

**************************************************************************

Example for all M4i/M4x analog acquisition cards. 
Shows calculation of FFT on a GPU in FIFO mode using RMDA.
  
Feel free to use this source for own projects and modify it in any kind

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

// ----- CUDA include -----
#   include <cuda_runtime.h>


#   include <cufft.h>
#   include <cufftXt.h>

// CUDA-C includes
#   include <cuda.h>

#include "../common/spcm_cuda_common.h"
#include "../common/spcm_cuda_kernels.h"


/*
**************************************************************************
main 
**************************************************************************
*/

int main ()
    {
    drv_handle  hCard = 0;
    int32       lCardType, lSerialNumber, lFncType, lNumCh, lBytesPerSample, lMaxADCValue;
    char        szErrorTextBuffer[ERRORTEXTLEN];
    uint32      dwError;
    int32       lStatus, lAvailUser, lPCPos;
    uint64      qwTotalMem = 0;

    // settings for the FIFO mode buffer handling
    int32       lBufferSize =   MEGA_B(4);
    int32       lNotifySize =   MEGA_B(1);


    // CARD SETUP

    // open card
    hCard = spcm_hOpen ((char*)"/dev/spcm0");
    if (!hCard)
        {
        printf ("no card found...\n");
        return EXIT_FAILURE;
        }

    // read type, function and sn and check for A/D card
    spcm_dwGetParam_i32 (hCard, SPC_PCITYP,                &lCardType);
    spcm_dwGetParam_i32 (hCard, SPC_PCISERIALNO,           &lSerialNumber);
    spcm_dwGetParam_i32 (hCard, SPC_FNCTYPE,               &lFncType);
    spcm_dwGetParam_i32 (hCard, SPC_MIINST_CHPERMODULE,    &lNumCh);
    spcm_dwGetParam_i32 (hCard, SPC_MIINST_BYTESPERSAMPLE, &lBytesPerSample);
    spcm_dwGetParam_i32 (hCard, SPC_MIINST_MAXADCVALUE,    &lMaxADCValue);

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
            return EXIT_FAILURE;
        }

    // do a simple standard setup
    lNumCh = 1;
    spcm_dwSetParam_i32 (hCard, SPC_CHENABLE,       (0x1 << lNumCh) - 1);   // all channels enabled
    spcm_dwSetParam_i32 (hCard, SPC_PRETRIGGER,     1024);                  // 1k of pretrigger data at start of FIFO mode
    spcm_dwSetParam_i32 (hCard, SPC_CARDMODE,       SPC_REC_FIFO_SINGLE);   // single FIFO mode
    spcm_dwSetParam_i32 (hCard, SPC_TIMEOUT,        5000);                  // timeout 5 s
    spcm_dwSetParam_i32 (hCard, SPC_TRIG_ORMASK,    SPC_TMASK_SOFTWARE);    // trigger set to software
    spcm_dwSetParam_i32 (hCard, SPC_TRIG_ANDMASK,   0);                     // ...
    spcm_dwSetParam_i32 (hCard, SPC_CLOCKMODE,      SPC_CM_INTPLL);         // clock mode internal PLL
    spcm_dwSetParam_i32 (hCard, SPC_CLOCKOUT,       0);                     // no clock output

    int lIR = 1000;                                                         // +/- 1 Volt
    spcm_dwSetParam_i32 (hCard, SPC_AMP0,           lIR);

    int64 llSamplerate = MEGA (250);
    spcm_dwSetParam_i64 (hCard, SPC_SAMPLERATE, llSamplerate);
    spcm_dwGetParam_i64 (hCard, SPC_SAMPLERATE, &llSamplerate);
    printf ("Used sample rate: %lld\n", llSamplerate);

    // ----- DMA BUFFER SETUP -----
    // ----- get buffer on GPU that will be used as target for RDMA transfer -----
    int lCUDADeviceIdx = 0;         // index of used CUDA device
    void* pvDMABuffer_gpu = pvGetRDMABuffer (lCUDADeviceIdx, lBufferSize);
    if (pvDMABuffer_gpu == NULL)
        {
        spcm_vClose (hCard);
        return EXIT_FAILURE;
        }

    // setup DMA transfer
    spcm_dwDefTransfer_i64 (hCard, SPCM_BUF_DATA, SPCM_DIR_CARDTOGPU, lNotifySize, pvDMABuffer_gpu, 0, lBufferSize);


    const int lSamplesPerCh  = (lNotifySize / lBytesPerSample) / lNumCh;
    const int lNumFFTSamples = lSamplesPerCh / 2 + 1; // number of samples after R2C FFT

    // ----- we will convert the raw data to float and at the same time scale them to volts -----
    cufftReal* afScaledAndDemuxed_gpu = NULL;
    cudaError_t eCudaErr = cudaMalloc (&afScaledAndDemuxed_gpu, lNumCh * lSamplesPerCh * sizeof (cufftReal));
    if (eCudaErr != cudaSuccess)
        {
        printf ("ERROR in cuMemAlloc(): %s\n", cudaGetErrorString(eCudaErr));

        spcm_vClose (hCard);

        cudaFree (pvDMABuffer_gpu);

        return EXIT_FAILURE;
        }

    // ----- this buffer will hold FFT'd data on GPU -----
    void* pvBufferFFT_gpu = NULL;
    eCudaErr = cudaMalloc (&pvBufferFFT_gpu, lNumCh * lNumFFTSamples * sizeof (cufftComplex));
    if (eCudaErr != cudaSuccess)
        {
        printf ("ERROR in cuMemAlloc(): %s\n", cudaGetErrorString(eCudaErr));

        spcm_vClose (hCard);

        cudaFree (afScaledAndDemuxed_gpu);
        cudaFree (pvDMABuffer_gpu);

        return EXIT_FAILURE;
        }

    // ----- this buffer will hold FFT'd data on host -----
    void* pvBufferFFT_host = NULL;
    eCudaErr = cudaMallocHost (&pvBufferFFT_host, lNumCh * lNumFFTSamples * sizeof (cufftComplex));
    if (eCudaErr != cudaSuccess)
        {
        printf ("ERROR in cuMemAlloc(): %s\n", cudaGetErrorString(eCudaErr));

        spcm_vClose (hCard);

        cudaFree (pvBufferFFT_gpu);
        cudaFree (afScaledAndDemuxed_gpu);
        cudaFree (pvDMABuffer_gpu);

        return EXIT_FAILURE;
        }

    // ----- setup CUDA FFT plan -----
    // CUFFT plan simple API
    cufftHandle hCudaFFTplan;
    //cufftPlan1d (&hCudaFFTplan, lNotifySize, CUFFT_R2C, 1);
    
    // CUFFT plan advanced API
    const int DATASIZE=lSamplesPerCh;
    int rank = 1;                                 // --- 1D FFTs
    int n[] = { DATASIZE };                       // --- Size of the Fourier transform
    int istride = 1, ostride = 1;                 // --- Distance between two successive input/output elements
    int idist = DATASIZE, odist = lNumFFTSamples; // --- Distance between batches
    int inembed[] = { 0, 0, 0, 0 };               // --- Input size with pitch (ignored for 1D transforms)
    int onembed[] = { 0, 0, 0, 0 };               // --- Output size with pitch (ignored for 1D transforms)
    int batch = lNumCh;                           // --- Number of batched executions
    cufftResult eCudaFFTErr = cufftPlanMany (&hCudaFFTplan, rank, n, 
              inembed, istride, idist,
              onembed, ostride, odist, CUFFT_R2C, batch);
    if (eCudaFFTErr != CUFFT_SUCCESS)
        {
        printf ("ERROR in cufftMakePlanMany(): %s (%d)\n",szCudaGetErrorText (eCudaFFTErr), eCudaFFTErr);

        spcm_vClose (hCard);

        cudaFreeHost (pvBufferFFT_host);
        cudaFree (pvBufferFFT_gpu);
        cudaFree (afScaledAndDemuxed_gpu);
        cudaFree (pvDMABuffer_gpu);

        return EXIT_FAILURE;
        }


    // ----- scale result of FFT by using a CUDA callback -----
    struct ST_CARDSETUP stCardSetup;
    stCardSetup.lNumCh       = lNumCh;
    stCardSetup.lMaxADCValue = lMaxADCValue;
    stCardSetup.lIR          = lIR;
    stCardSetup.lLen         = lSamplesPerCh;

    // allocate memory for a struct that is used as parameter for the FFT callback
    struct ST_CARDSETUP* d_pstCardSetup;
    eCudaErr = cudaMalloc (&d_pstCardSetup, sizeof (struct ST_CARDSETUP));
    if (eCudaErr != cudaSuccess)
        {
        printf ("ERROR in cudaMalloc() for CARDSETUP: %s\n", cudaGetErrorString(eCudaErr));

        spcm_vClose (hCard);

        cudaFree (afScaledAndDemuxed_gpu);
        cudaFree (pvDMABuffer_gpu);
        cudaFree (pvBufferFFT_gpu);
        cudaFreeHost (pvBufferFFT_host);

        return EXIT_FAILURE;
        }
    // copy struct data to GPU
    eCudaErr = cudaMemcpy (d_pstCardSetup, &stCardSetup, sizeof (struct ST_CARDSETUP), cudaMemcpyHostToDevice);
    if (eCudaErr != cudaSuccess)
        {
        printf ("ERROR in cudaMemcpy() for CARDSETUP: %s\n", cudaGetErrorString(eCudaErr));

        spcm_vClose (hCard);

        cudaFree (d_pstCardSetup);
        cudaFreeHost (pvBufferFFT_host);
        cudaFree (pvBufferFFT_gpu);
        cudaFree (afScaledAndDemuxed_gpu);
        cudaFree (pvDMABuffer_gpu);

        return EXIT_FAILURE;
        }

    // load the FFT callback function
    cufftCallbackStoreC h_storeCallbackPtr;
    cudaMemcpyFromSymbol (&h_storeCallbackPtr, d_storeCallbackPtr, sizeof (h_storeCallbackPtr));

    // add FFT callback and its parameters to FFT plan
    eCudaFFTErr = cufftXtSetCallback (hCudaFFTplan, (void**)&h_storeCallbackPtr, CUFFT_CB_ST_COMPLEX, (void**)&d_pstCardSetup);
    if (eCudaFFTErr != CUFFT_SUCCESS)
        {
        printf ("ERROR in cufftXtSetCallback(): %d\n", eCudaFFTErr);

        spcm_vClose (hCard);

        cudaFree (d_pstCardSetup);
        cudaFreeHost (pvBufferFFT_host);
        cudaFree (pvBufferFFT_gpu);
        cudaFree (afScaledAndDemuxed_gpu);
        cudaFree (pvDMABuffer_gpu);

        return EXIT_FAILURE;
        }


    // start everything
    dwError = spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER | M2CMD_DATA_STARTDMA);

    // check for error
    if (dwError != ERR_OK)
        {
        spcm_dwGetErrorInfo_i32 (hCard, NULL, NULL, szErrorTextBuffer);
        printf ("%s\n", szErrorTextBuffer);

        spcm_vClose (hCard);

        cudaFree (d_pstCardSetup);
        cudaFreeHost (pvBufferFFT_host);
        cudaFree (pvBufferFFT_gpu);
        cudaFree (afScaledAndDemuxed_gpu);
        cudaFree (pvDMABuffer_gpu);

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
                printf ("\rStat:%08x Pos:%08x Avail:%08x Total:%.2fMB", lStatus, lPCPos, lAvailUser, (double) (int64) qwTotalMem / MEGA_B(1));


                // ----- scale and sort the channels on the GPU -----
                #define CUDA_MAX_THREADS_PER_BLOCK 1024
                int lBlocks =  (lNotifySize / lBytesPerSample) / CUDA_MAX_THREADS_PER_BLOCK;
                int lThreadsPerBlock = CUDA_MAX_THREADS_PER_BLOCK;
                CudaKernelScaleAndDemux <<< lBlocks, lThreadsPerBlock >>> ((short*)((char*)pvDMABuffer_gpu + lPCPos), afScaledAndDemuxed_gpu, d_pstCardSetup);

                // ----- execute the FFT on the GPU -----
                cufftResult eCudaFFTErr = cufftExecR2C (hCudaFFTplan, afScaledAndDemuxed_gpu, (cufftComplex*)pvBufferFFT_gpu);
                if (eCudaFFTErr != CUFFT_SUCCESS)
                    {
                    printf ("ERROR in cufftExecR2C: %s (%d)\n", szCudaGetErrorText (eCudaFFTErr), eCudaFFTErr);
                    break;
                    }

                // after FFT has finished we convert to dB full scale
                //CudaKernelToDBFS <<< 1, lNotify_samples / 2 >>> (pvBufferFFT_gpu, pvBufferFFT_gpu, lIR);

                // ----- after FFT (and conversion to dBFS) have finished we copy FFT'd data back to host -----
                eCudaErr = cudaMemcpy (pvBufferFFT_host, pvBufferFFT_gpu, lNumCh * lNumFFTSamples * sizeof (cufftComplex), cudaMemcpyDeviceToHost);
                if (eCudaErr != cudaSuccess)
                    {
                    printf ("ERROR in cudaMemcpy(InvData): %s\n", cudaGetErrorString(eCudaErr));
                    break;
                    }

                // this is the point to do something with the FFT'd data
                //cufftComplex* afComplex = static_cast < cufftComplex* > (pvBufferFFT_host);

                spcm_dwSetParam_i32 (hCard, SPC_DATA_AVAIL_CARD_LEN,  lNotifySize);
                }

            }
        }

    // send the stop command
    dwError = spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_CARD_STOP | M2CMD_DATA_STOPDMA);

    // clean up
    printf ("Finished...\n");

    spcm_vClose (hCard);

    cudaFree (d_pstCardSetup);
    cudaFreeHost (pvBufferFFT_host);
    cudaFree (pvBufferFFT_gpu);
    cudaFree (afScaledAndDemuxed_gpu);
    cudaFree (pvDMABuffer_gpu);

    return EXIT_SUCCESS;
    }

