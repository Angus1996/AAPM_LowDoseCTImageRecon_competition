
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 * Author : Rui Liu
 * Date : Apr. 4, 2016
 */
#include "DD2Back_4gpus.h"
#include "utilities.cuh"

#define BACK_BLKX 64
#define BACK_BLKY 4
#define BACK_BLKZ 1

enum BackProjectionMethod{ _BRANCHLESS, _PSEUDODD, _ZLINEBRANCHLESS, _VOLUMERENDERING };

__global__ void addOneSidedZeroBoarder_multiSlice_Fan(const float* prj_in, float* prj_out, int DNU, int SLN, int PN)
{
	int idv = threadIdx.x + blockIdx.x * blockDim.x;
	int idu = threadIdx.y + blockIdx.y * blockDim.y;
	int pn = threadIdx.z + blockIdx.z * blockDim.z;
	if (idu < DNU && idv < SLN && pn < PN)
	{
		int i = (pn * DNU + idu) * SLN + idv;
		int ni = (pn * (DNU + 1) + (idu + 1)) * SLN + idv;
		prj_out[ni] = prj_in[i];
	}
}


__global__ void heorizontalIntegral_multiSlice_Fan(float* prj, int DNU, int SLN, int PN)
{
	int idv = threadIdx.x + blockIdx.x * blockDim.x;
	int pIdx = threadIdx.y + blockIdx.y * blockDim.y;
	if (idv < SLN && pIdx < PN)
	{
		int headPrt = pIdx * DNU * SLN + idv;
		for (int ii = 1; ii < DNU; ++ii)
		{
			prj[headPrt + ii * SLN] = prj[headPrt + ii * SLN] + prj[headPrt + (ii - 1) * SLN];
		}
	}
}

thrust::device_vector<float> genSAT_of_Projection_multiSliceFan(
	float* hprj,
	int DNU, int SLN, int PN)
{
	const int siz = DNU * SLN * PN;
	const int nsiz = (DNU + 1) * SLN * PN;
	thrust::device_vector<float> prjSAT(nsiz, 0);
	thrust::device_vector<float> prj(hprj, hprj + siz);
	dim3 copyBlk(64, 16, 1); //MAY CHANGED
	dim3 copyGid(
		(SLN + copyBlk.x - 1) / copyBlk.x,
		(DNU + copyBlk.y - 1) / copyBlk.y,
		(PN + copyBlk.z - 1) / copyBlk.z);

	addOneSidedZeroBoarder_multiSlice_Fan << <copyGid, copyBlk >> >(
		thrust::raw_pointer_cast(&prj[0]),
		thrust::raw_pointer_cast(&prjSAT[0]),
		DNU, SLN, PN);

	const int nDNU = DNU + 1;

	copyBlk.x = 64; // MAY CHANGED
	copyBlk.y = 16;
	copyBlk.z = 1;
	copyGid.x = (SLN + copyBlk.x - 1) / copyBlk.x;
	copyGid.y = (PN + copyBlk.y - 1) / copyBlk.y;
	copyGid.z = 1;

	heorizontalIntegral_multiSlice_Fan << <copyGid, copyBlk >> >(
		thrust::raw_pointer_cast(&prjSAT[0]),
		nDNU, SLN, PN);

	return prjSAT;
}


void createTextureObject_multiSliceFan(
	cudaTextureObject_t& texObj,
	cudaArray* d_prjArray,
	int Width, int Height, int Depth,
	float* sourceData,
	cudaMemcpyKind memcpyKind,
	cudaTextureAddressMode addressMode,
	cudaTextureFilterMode textureFilterMode,
	cudaTextureReadMode textureReadMode,
	bool isNormalized)
{
	cudaExtent prjSize;
	prjSize.width = Width;
	prjSize.height = Height;
	prjSize.depth = Depth;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

	cudaMalloc3DArray(&d_prjArray, &channelDesc, prjSize);
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr(
		(void*) sourceData, prjSize.width * sizeof(float),
		prjSize.width, prjSize.height);
	copyParams.dstArray = d_prjArray;
	copyParams.extent = prjSize;
	copyParams.kind = memcpyKind;
	cudaMemcpy3D(&copyParams);
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = d_prjArray;
	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = addressMode;
	texDesc.addressMode[1] = addressMode;
	texDesc.addressMode[2] = addressMode;
	texDesc.filterMode = textureFilterMode;
	texDesc.readMode = textureReadMode;
	texDesc.normalizedCoords = isNormalized;

	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
}

void destroyTextureObject_multiSliceFan(cudaTextureObject_t& texObj, cudaArray* d_array)
{
	cudaDestroyTextureObject(texObj);
	cudaFreeArray(d_array);
}




__global__ void DD2_gpu_back_ker_multiSlice_Fan(
	cudaTextureObject_t prjTexObj,
	float* vol,
	const byte* __restrict__ msk,
	const float2* __restrict__ cossin,
	float2 s,
	float S2D,
	float2 curvox, // imgCenter index
	float dx, float dbeta, float detCntIdx,
	int2 VN, int SLN, int PN)
{
	int3 id;
	id.z = threadIdx.x + __umul24(blockIdx.x, blockDim.x);
	id.x = threadIdx.y + __umul24(blockIdx.y, blockDim.y);
	id.y = threadIdx.z + __umul24(blockIdx.z, blockDim.z);
	if(id.z < SLN && id.x < VN.x && id.y < VN.y)
	{
		if(msk[id.y * VN.x + id.x] != 1)
		{
			return;
		}
		curvox = make_float2((id.x - curvox.x) * dx, (id.y - curvox.y) * dx);
		float2 cursour;
		float idxL, idxR;
		float cosVal;
		float summ = 0;

		float2 cossinT;
		float inv_sid = 1.0f / sqrtf(s.x * s.x + s.y * s.y);

		float2 dir;
		float l_square;
		float l;

		float alpha;
		float deltaAlpha;
		//S2D /= ddv;
		dbeta = 1.0 / dbeta;

		float ddv;
		for(int angIdx = 0; angIdx < PN; ++angIdx)
		{
			cossinT = cossin[angIdx * SLN + id.z];
			cursour = make_float2(
					s.x * cossinT.x - s.y * cossinT.y,
					s.x * cossinT.y + s.y * cossinT.x);

			dir = curvox - cursour;

			l_square = dir.x * dir.x + dir.y * dir.y;

			l = rsqrtf(l_square); // 1 / sqrt(l_square);
			alpha = asinf((cursour.y * dir.x - cursour.x * dir.y) * inv_sid * l);

			if(fabsf(cursour.x) > fabsf(cursour.y))
			{
				ddv = dir.x;
			}
			else
			{
				ddv = dir.y;
			}

			deltaAlpha  = ddv / l_square * dx * 0.5;
			cosVal = dx / ddv * sqrtf(l_square);

			idxL = (alpha - deltaAlpha) * dbeta + detCntIdx + 1.0;
			idxR = (alpha + deltaAlpha) * dbeta + detCntIdx + 1.0;

			summ += (tex3D<float>(prjTexObj,id.z + 0.5, idxR, angIdx + 0.5) -
					 tex3D<float>(prjTexObj,id.z + 0.5, idxL, angIdx + 0.5)) * cosVal;
		}
		__syncthreads();
		vol[(id.y * VN.x + id.x) * SLN + id.z] = summ;

	}
}


void DD2_gpu_back(float x0, float y0,
		int DNU,
		float* xds, float* yds,
		float detCntIdx,
		float imgXCenter, float imgYCenter,
		float* hangs, int PN, int XN, int YN, int SLN,
		float* hvol, float* hprj, float dx,
		byte* mask, int gpunum)
{
	cudaSetDevice(gpunum);
	cudaDeviceReset();

	float2 objCntIdx = make_float2(
			(XN - 1.0) * 0.5 - imgXCenter / dx,
			(YN - 1.0) * 0.5 - imgYCenter / dx); // set the center of the image
	float2 sour = make_float2(x0, y0);

	thrust::device_vector<byte> msk(mask,mask + XN * YN);
	thrust::device_vector<float> vol(XN * YN * SLN, 0);

	const float S2D = hypotf(xds[0] - x0, yds[0] - y0);

	thrust::device_vector<float2> cossin(PN * SLN);
	thrust::device_vector<float> angs(hangs, hangs + PN * SLN);
	thrust::transform(angs.begin(), angs.end(), cossin.begin(),	CTMBIR::Constant_MultiSlice(x0,y0));

	//Calculate the corresponding parameters such as
	// return make_float4(detCtrIdxU, detCtrIdxV, dbeta, ddv);
	float detTransverseSize = sqrt(powf(xds[1] - xds[0],2) + powf(yds[1] - yds[0],2));
	float dbeta = atanf(detTransverseSize / S2D * 0.5) * 2.0f;

	cudaArray* d_prjArray = nullptr;
	cudaTextureObject_t texObj;

	dim3 blk;
	dim3 gid;

	thrust::device_vector<float> prjSAT;

	//Generate the SAT along XY direction;
	prjSAT = genSAT_of_Projection_multiSliceFan(hprj, DNU, SLN, PN);
	createTextureObject_multiSliceFan(texObj,d_prjArray, SLN, DNU + 1, PN,
			thrust::raw_pointer_cast(&prjSAT[0]),
			cudaMemcpyDeviceToDevice,
			cudaAddressModeClamp, cudaFilterModeLinear, cudaReadModeElementType, false);
	prjSAT.clear();

	blk.x = BACK_BLKX; // May be changed
	blk.y = BACK_BLKY;
	blk.z = BACK_BLKZ;
	gid.x = (SLN + blk.x - 1) / blk.x;
	gid.y = (XN + blk.y - 1) / blk.y;
	gid.z = (YN + blk.z - 1) / blk.z;

	DD2_gpu_back_ker_multiSlice_Fan<< <gid, blk >> >(texObj,
		thrust::raw_pointer_cast(&vol[0]),
		thrust::raw_pointer_cast(&msk[0]),
		thrust::raw_pointer_cast(&cossin[0]),
		make_float2(x0, y0),
		S2D,
		make_float2(objCntIdx.x, objCntIdx.y),
		dx, dbeta, detCntIdx,	make_int2(XN, YN), SLN, PN);

	thrust::copy(vol.begin(),vol.end(),hvol);
	destroyTextureObject_multiSliceFan(texObj, d_prjArray);

	vol.clear();
	msk.clear();
	angs.clear();
	cossin.clear();


}



void DD2Back_gpu(
	float x0, float y0,
	int DNU,
	float* xds, float* yds,
	float detCntIdx,
	float imgXCenter, float imgYCenter,
	float* hangs, int PN,
	int XN, int YN, int SLN,
	float* hvol, float* hprj,
	float dx,
	byte* mask, int gpunum)
{

	DD2_gpu_back(x0,y0,DNU,xds,yds,detCntIdx, imgXCenter,imgYCenter,
					hangs, PN, XN, YN, SLN, hvol, hprj, dx, mask, gpunum);

}



extern "C"
void DD2Back_3gpus(
	float x0, float y0,
	int DNU,
	float* xds, float* yds,
	float detCntIdx,
	float imgXCenter, float imgYCenter,
	float* hangs, int PN,
	int XN, int YN, int SLN,
	float* hvol, float* hprj,
	float dx,
	byte* mask, int* startIdx)
{
    
	// Divide the volume into three parts
	int* SLNn = new int[3];
	SLNn[0] = startIdx[1] - startIdx[0];
	SLNn[1] = startIdx[2] - startIdx[1];
	SLNn[2] = SLN - startIdx[2];
	float** shvol = new float*[3];
	shvol[0] = new float[XN * YN * SLNn[0]];
	shvol[1] = new float[XN * YN * SLNn[1]];
	shvol[2] = new float[XN * YN * SLNn[2]];

	float** shprj = new float*[3];
	shprj[0] = new float[DNU * SLNn[0] * PN];
	shprj[1] = new float[DNU * SLNn[1] * PN];
	shprj[2] = new float[DNU * SLNn[2] * PN];


	float** shang = new float*[3];
	shang[0] = new float[PN * SLNn[0]];
	shang[1] = new float[PN * SLNn[1]];
	shang[2] = new float[PN * SLNn[2]];
	
	for(int pIdx = 0; pIdx != PN; ++pIdx)
	{
		for(int sIdx = 0; sIdx != SLN; ++sIdx)
		{
			if(sIdx >= startIdx[0] && sIdx < startIdx[1])
			{
				shang[0][pIdx * SLNn[0] + (sIdx - startIdx[0])] = hangs[pIdx * SLN + sIdx];
			}
			else if(sIdx >= startIdx[1] && sIdx < startIdx[2])
			{
				shang[1][pIdx * SLNn[1] + (sIdx - startIdx[1])] = hangs[pIdx * SLN + sIdx];
			}
			else if(sIdx >= startIdx[2] && sIdx < SLN)
			{
				shang[2][pIdx * SLNn[2] + (sIdx - startIdx[2])] = hangs[pIdx * SLN + sIdx];
			}
		}	
	}

	//int kk = 0;
	//Split the projection
	omp_set_num_threads(32);
#pragma omp parallel for
	for(int i = 0; i < DNU * PN; ++i)
	{
		for(int v = 0; v != SLN; ++v)
		{
			if(v >= startIdx[0] && v < startIdx[1])
			{
				shprj[0][i * SLNn[0] + (v - startIdx[0])] = hprj[i * SLN + v];
			}
			else if( v >= startIdx[1] && v < startIdx[2])
			{
				shprj[1][i * SLNn[1] + (v - startIdx[1])] = hprj[i * SLN + v];
			}
			else if (v >= startIdx[2] && v < SLN)
			{
				shprj[2][i * SLNn[2] + (v - startIdx[2])] = hprj[i * SLN + v];
			}
		}
	}

    //cudaDeviceReset();
    cudaDeviceSynchronize();
	omp_set_num_threads(3);
#pragma omp parallel for
	for(int i = 0; i < 3; ++i)
	{
		DD2Back_gpu(x0, y0, DNU, xds, yds, detCntIdx,
			imgXCenter, imgYCenter, shang[i],
			PN, XN, YN, SLNn[i], shvol[i], shprj[i], dx,mask, i);
	}

	//Gather all

	omp_set_num_threads(32);
#pragma omp parallel for
	for(int i = 0; i < XN * YN; ++i)
	{
		for(int v = 0; v != SLN; ++v)
		{
			float val = 0;
			if(v >= startIdx[0] && v < startIdx[1])
			{
				val = shvol[0][i * SLNn[0] + (v - startIdx[0])];
			}
			else if( v >= startIdx[1] && v < startIdx[2])
			{
				val = shvol[1][i * SLNn[1] + (v - startIdx[1])];
			}
			else if (v >= startIdx[2] && v < SLN)
			{
				val = shvol[2][i * SLNn[2] + (v - startIdx[2])];
			}
			hvol[i * SLN + v] = val;
		}
	}
	delete[] shprj[0];
	delete[] shprj[1];
	delete[] shprj[2];
	delete[] shvol[0];
	delete[] shvol[1];
	delete[] shvol[2];
	delete[] shprj;
	delete[] shvol;
	delete[] SLNn;

    //cudaDeviceReset();
    //cudaDeviceSynchronize();
}


