
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
 * Date   : Apr. 4, 2016
 */
#include "DD2Proj_4gpus.h"
#include "utilities.cuh"
#include <omp.h>
#define BLKX 32
#define BLKY 8
#define BLKZ 1

namespace DD2{


// Copy the volume from the original to
template<typename Ta, typename Tb>
__global__ void naive_copyToTwoVolumes(Ta* in_ZXY,
	Tb* out_ZXY, Tb* out_ZYX,
	int XN, int YN, int ZN)
{
	int idz = threadIdx.x + blockIdx.x * blockDim.x;
	int idx = threadIdx.y + blockIdx.y * blockDim.y;
	int idy = threadIdx.z + blockIdx.z * blockDim.z;
	if (idx < XN && idy < YN && idz < ZN)
	{
		int i = (idy * XN + idx) * ZN + idz;
		int ni = (idy * (XN + 1) + (idx + 1)) * ZN + idz;
		int nj = (idx * (YN + 1) + (idy + 1)) * ZN + idz;

		out_ZXY[ni] = in_ZXY[i];
		out_ZYX[nj] = in_ZXY[i];
	}
}


__global__ void horizontalIntegral(float* prj, int DNU, int DNV, int PN)
{
	int idv = threadIdx.x + blockIdx.x * blockDim.x;
	int pIdx = threadIdx.y + blockIdx.y * blockDim.y;
	if (idv < DNV && pIdx < PN)
	{
		int headPrt = pIdx * DNU * DNV + idv;
		for (int ii = 1; ii < DNU; ++ii)
		{
			prj[headPrt + ii * DNV] = prj[headPrt + ii * DNV] + prj[headPrt + (ii - 1) * DNV];
		}
	}
}






void genSAT_for_Volume_MultiSlice(float* hvol,
	thrust::device_vector<float>&ZXY,
	thrust::device_vector<float>&ZYX,
	int XN, int YN, int ZN)
{
	const int siz = XN * YN * ZN;
	const int nsiz_ZXY = ZN * (XN + 1) * YN; //Only XN or YN dimension changes
	const int nsiz_ZYX = ZN * (YN + 1) * XN;
	ZXY.resize(nsiz_ZXY);
	ZYX.resize(nsiz_ZYX);

	thrust::device_vector<float> vol(hvol, hvol + siz);

	dim3 blk(64, 16, 1);
	dim3 gid(
		(ZN + blk.x - 1) / blk.x,
		(XN + blk.y - 1) / blk.y,
		(YN + blk.z - 1) / blk.z);

	naive_copyToTwoVolumes << <gid, blk >> >(
		thrust::raw_pointer_cast(&vol[0]),
		thrust::raw_pointer_cast(&ZXY[0]),
		thrust::raw_pointer_cast(&ZYX[0]),
		XN, YN, ZN);

	vol.clear();

	blk.x = 64;
	blk.y = 16;
	blk.z = 1;
	gid.x = (ZN + blk.x - 1) / blk.x;
	gid.y = (YN + blk.y - 1) / blk.y;
	gid.z = 1;

	horizontalIntegral << <gid, blk >> >(
		thrust::raw_pointer_cast(&ZXY[0]),
		XN+1, ZN, YN);

	blk.x = 64;
	blk.y = 16;
	blk.z = 1;
	gid.x = (ZN + blk.x - 1) / blk.x;
	gid.y = (XN + blk.y - 1) / blk.y;
	gid.z = 1;

	horizontalIntegral << <gid, blk >> >(
		thrust::raw_pointer_cast(&ZYX[0]),
		YN+1, ZN, XN);
}


__global__  void DD2_gpu_proj_branchless_sat2d_ker(
	cudaTextureObject_t volTex1,
	cudaTextureObject_t volTex2,
	float* proj,
	float2 s, // source position
	const float2* __restrict__ cossin,
	const float* __restrict__ xds,
	const float* __restrict__ yds,
	const float* __restrict__ bxds,
	const float* __restrict__ byds,
	float2 objCntIdx,
	float dx,
	int XN, int YN, int SLN,
	int DNU, int PN)
{
	int slnIdx = threadIdx.x + blockIdx.x * blockDim.x;
	int detIdU = threadIdx.y + blockIdx.y * blockDim.y;
	int angIdx = threadIdx.z + blockIdx.z * blockDim.z;
	if(slnIdx < SLN && detIdU < DNU && angIdx < PN)
	{
		float2 dir = cossin[angIdx * SLN + slnIdx]; // cossin;

		float2 cursour = make_float2(
			s.x * dir.x - s.y * dir.y,
			s.x * dir.y + s.y * dir.x); // current source position;
		s = dir;

		float2 curDet = make_float2(
				xds[detIdU] * s.x - yds[detIdU] * s.y,
				xds[detIdU] * s.y + yds[detIdU] * s.x);

		float2 curDetL = make_float2(
				bxds[detIdU] * s.x - byds[detIdU] * s.y,
				bxds[detIdU] * s.y + byds[detIdU] * s.x);

		float2 curDetR = make_float2(
				bxds[detIdU+1] * s.x - byds[detIdU+1] * s.y,
				bxds[detIdU+1] * s.y + byds[detIdU+1] * s.x);

		dir = normalize(curDet - cursour);

		float factL = 0;
		float factR = 0;
		float constVal = 0;
		float obj = 0;
		float realL = 0;
		float realR = 0;
		float intersectLength = 0;

		float invdx = 1.0f / dx;
		//float summ[BLKX];
		float summ;
		if(fabsf(s.x) <= fabsf(s.y))
		{

			summ = 0;
			factL = (curDetL.y - cursour.y) / (curDetL.x - cursour.x);
			factR = (curDetR.y - cursour.y) / (curDetR.x - cursour.x);

			constVal = dx / fabsf(dir.x);
#pragma unroll
			for (int ii = 0; ii < XN; ++ii)
			{
				obj = (ii - objCntIdx.x) * dx;

				realL = (obj - curDetL.x) * factL + curDetL.y;
				realR = (obj - curDetR.x) * factR + curDetR.y;

				intersectLength = realR - realL;
				realL = realL * invdx + objCntIdx.y + 1;
				realR = realR * invdx + objCntIdx.y + 1;

				summ += (tex3D<float>(volTex2, slnIdx + 0.5f, realR, ii + 0.5) - tex3D<float>(volTex2, slnIdx + 0.5, realL, ii + 0.5)) / intersectLength;

			}
			__syncthreads();
			proj[(angIdx * DNU + detIdU) * SLN + slnIdx] = summ * constVal;

		}
		else
		{

			summ = 0;
			factL = (curDetL.x - cursour.x) / (curDetL.y - cursour.y);
			factR = (curDetR.x - cursour.x) / (curDetR.y - cursour.y);

			constVal = dx / fabsf(dir.y);
#pragma unroll
			for (int ii = 0; ii < YN; ++ii)
			{
				obj = (ii - objCntIdx.y) * dx;

				realL = (obj - curDetL.y) * factL + curDetL.x;
				realR = (obj - curDetR.y) * factR + curDetR.x;

				intersectLength = realR - realL;
				realL = realL * invdx + objCntIdx.x + 1;
				realR = realR * invdx + objCntIdx.x + 1;

				summ += (tex3D<float>(volTex1, slnIdx + 0.5f, realR, ii + 0.5) - tex3D<float>(volTex1, slnIdx + 0.5, realL, ii + 0.5)) / intersectLength;
			}
			__syncthreads();
			proj[(angIdx * DNU + detIdU) * SLN + slnIdx] = summ * constVal;
			//__syncthreads();
		}
	}
}




void DD2_gpu_proj_branchless_sat2d(
	float x0, float y0,
	int DNU,
	float* xds, float* yds,
	float imgXCenter, float imgYCenter,
	float* hangs, int PN,
	int XN, int YN, int SLN, // SLN is the slice number, it is the same as the rebinned projection slices
	float* vol, float* hprj, float dx, byte* mask, int gpunum)
{
	for (int i = 0; i != XN * YN; ++i)
	{
		byte v = mask[i];
		for (int z = 0; z != SLN; ++z)
		{
			vol[i * SLN + z] = vol[i * SLN + z] * v;
		}
	}

	CUDA_SAFE_CALL(cudaSetDevice(gpunum));
	cudaDeviceReset();

	float* bxds = new float[DNU + 1];
	float* byds = new float[DNU + 1];

	DD3Boundaries(DNU + 1, xds, bxds);
	DD3Boundaries(DNU + 1, yds, byds);

	float objCntIdxX = (XN - 1.0) * 0.5 - imgXCenter / dx;
	float objCntIdxY = (YN - 1.0) * 0.5 - imgYCenter / dx;


	thrust::device_vector<float> SATZXY;
	thrust::device_vector<float> SATZYX;
	genSAT_for_Volume_MultiSlice(vol, SATZXY, SATZYX, XN, YN, SLN);

	cudaExtent volumeSize1;
	cudaExtent volumeSize2;
	volumeSize1.width = SLN;
	volumeSize1.height = XN + 1;
	volumeSize1.depth = YN;

	volumeSize2.width = SLN;
	volumeSize2.height = YN + 1;
	volumeSize2.depth = XN;

	cudaChannelFormatDesc channelDesc1 = cudaCreateChannelDesc<float>();
	cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float>();

	cudaArray* d_volumeArray1;
	cudaArray* d_volumeArray2;

	cudaMalloc3DArray(&d_volumeArray1, &channelDesc1, volumeSize1);
	cudaMalloc3DArray(&d_volumeArray2, &channelDesc2, volumeSize2);

	cudaMemcpy3DParms copyParams1 = { 0 };
	copyParams1.srcPtr = make_cudaPitchedPtr((void*)
		thrust::raw_pointer_cast(&SATZXY[0]),
		volumeSize1.width * sizeof(float),
		volumeSize1.width, volumeSize1.height);
	copyParams1.dstArray = d_volumeArray1;
	copyParams1.extent = volumeSize1;
	copyParams1.kind = cudaMemcpyDeviceToDevice;

	cudaMemcpy3DParms copyParams2 = { 0 };
	copyParams2.srcPtr = make_cudaPitchedPtr((void*)
		thrust::raw_pointer_cast(&SATZYX[0]),
		volumeSize2.width * sizeof(float),
		volumeSize2.width, volumeSize2.height);
	copyParams2.dstArray = d_volumeArray2;
	copyParams2.extent = volumeSize2;
	copyParams2.kind = cudaMemcpyDeviceToDevice;

	cudaMemcpy3D(&copyParams1);
	cudaMemcpy3D(&copyParams2);

	SATZXY.clear();
	SATZYX.clear();

	cudaResourceDesc resDesc1;
	cudaResourceDesc resDesc2;
	memset(&resDesc1, 0, sizeof(resDesc1));
	memset(&resDesc2, 0, sizeof(resDesc2));

	resDesc1.resType = cudaResourceTypeArray;
	resDesc2.resType = cudaResourceTypeArray;

	resDesc1.res.array.array = d_volumeArray1;
	resDesc2.res.array.array = d_volumeArray2;

	cudaTextureDesc texDesc1;
	cudaTextureDesc texDesc2;

	memset(&texDesc1, 0, sizeof(texDesc1));
	memset(&texDesc2, 0, sizeof(texDesc2));

	texDesc1.addressMode[0] = cudaAddressModeClamp;
	texDesc1.addressMode[1] = cudaAddressModeClamp;
	texDesc1.addressMode[2] = cudaAddressModeClamp;

	texDesc2.addressMode[0] = cudaAddressModeClamp;
	texDesc2.addressMode[1] = cudaAddressModeClamp;
	texDesc2.addressMode[2] = cudaAddressModeClamp;

	texDesc1.filterMode = cudaFilterModeLinear;
	texDesc2.filterMode = cudaFilterModeLinear;

	texDesc1.readMode = cudaReadModeElementType;
	texDesc2.readMode = cudaReadModeElementType;

	texDesc1.normalizedCoords = false;
	texDesc2.normalizedCoords = false;

	cudaTextureObject_t texObj1 = 0;
	cudaTextureObject_t texObj2 = 0;

	cudaCreateTextureObject(&texObj1, &resDesc1, &texDesc1, nullptr);
	cudaCreateTextureObject(&texObj2, &resDesc2, &texDesc2, nullptr);

	thrust::device_vector<float> prj(DNU * SLN * PN, 0);
	thrust::device_vector<float> d_xds(xds, xds + DNU);
	thrust::device_vector<float> d_yds(yds, yds + DNU);

	thrust::device_vector<float> d_bxds(bxds, bxds + DNU + 1);
	thrust::device_vector<float> d_byds(byds, byds + DNU + 1);
	thrust::device_vector<float> angs(hangs, hangs + PN * SLN);

	thrust::device_vector<float2> cossin(PN * SLN);
	thrust::transform(angs.begin(), angs.end(), cossin.begin(), CTMBIR::Constant_MultiSlice(x0, y0));


	dim3 blk(BLKX, BLKY, BLKZ);
	dim3 gid(
		(SLN + blk.x - 1) / blk.x,
		(DNU + blk.y - 1) / blk.y,
		(PN + blk.z - 1) / blk.z);

	DD2_gpu_proj_branchless_sat2d_ker<<<gid,blk>>>(texObj1,texObj2,
		thrust::raw_pointer_cast(&prj[0]),
		make_float2(x0,y0),
		thrust::raw_pointer_cast(&cossin[0]),
		thrust::raw_pointer_cast(&d_xds[0]),
		thrust::raw_pointer_cast(&d_yds[0]),
		thrust::raw_pointer_cast(&d_bxds[0]),
		thrust::raw_pointer_cast(&d_byds[0]),
		make_float2(objCntIdxX, objCntIdxY),dx,XN,YN,SLN,DNU,PN);

	thrust::copy(prj.begin(), prj.end(), hprj);
	cudaDestroyTextureObject(texObj1);
	cudaDestroyTextureObject(texObj2);

	cudaFreeArray(d_volumeArray1);
	cudaFreeArray(d_volumeArray2);

	prj.clear();
	angs.clear();


	d_xds.clear();
	d_yds.clear();

	d_bxds.clear();
	d_byds.clear();
	cossin.clear();

	delete[] bxds;
	delete[] byds;

}

}; //End NAMESPACE DD2


void DD2Proj_gpu(
	float x0, float y0,
	int DNU,
	float* xds, float* yds,
	float imgXCenter, float imgYCenter,
	float* hangs, int PN,
	int XN, int YN, int SLN,
	float* hvol, float* hprj,
	float dx,
	byte* mask, int gpunum)
{
	DD2::DD2_gpu_proj_branchless_sat2d(x0, y0, DNU, xds, yds, imgXCenter, imgYCenter,
				hangs, PN, XN, YN, SLN, hvol, hprj, dx, mask, gpunum);
}




extern "C"
void DD2Proj_3gpus(
		float x0, float y0,
		int DNU,
		float* xds, float* yds,
		float imgXCenter, float imgYCenter,
		float* hangs, int PN,
		int XN, int YN, int SLN,
		float* hvol, float* hprj,
		float dx, byte* mask, int* startIdx)
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
	
	omp_set_num_threads(32);
#pragma omp parallel for
	for(int i = 0; i < XN * YN; ++i)
	{
		for(int v = 0; v != SLN; ++v)
		{
			if(v >= startIdx[0] && v < startIdx[1])
			{
				shvol[0][i * SLNn[0] + (v - startIdx[0])] = hvol[i * SLN + v];
			}
			else if( v >= startIdx[1] && v < startIdx[2])
			{
				shvol[1][i * SLNn[1] + (v - startIdx[1])] = hvol[i * SLN + v];
			}
			else if (v >= startIdx[2] && v < SLN)
			{
				shvol[2][i * SLNn[2] + (v - startIdx[2])] = hvol[i * SLN + v];
			}
		}
	}


    //cudaDeviceReset();
    cudaDeviceSynchronize();
	omp_set_num_threads(3);

#pragma omp parallel for
	for(int i = 0; i < 3; ++i)
	{
		DD2Proj_gpu(x0, y0, DNU, xds, yds,
				imgXCenter, imgYCenter, shang[i],
				PN, XN, YN, SLNn[i], shvol[i], shprj[i], dx, mask, i);
	}

	//Gather all
	omp_set_num_threads(32);
#pragma omp parallel for
	for(int i = 0; i < DNU * PN; ++i)
	{
		for(int v = 0; v != SLN; ++v)
		{
			float val = 0;
			if(v >= startIdx[0] && v < startIdx[1])
			{
				val = shprj[0][i * SLNn[0] + (v - startIdx[0])];
			}
			else if( v >= startIdx[1] && v < startIdx[2])
			{
				val = shprj[1][i * SLNn[1] + (v - startIdx[1])];
			}
			else if (v >= startIdx[2] && v < SLN)
			{
				val = shprj[2][i * SLNn[2] + (v - startIdx[2])];
			}
			hprj[i * SLN + v]  = val;
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



