#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/find.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/binary_search.h>

#include <thread>
#include <omp.h>

#include <cstring>
#include <iostream>

typedef unsigned char byte;

#ifndef __PI__
#define __PI__
#define PI			3.14159265358979323846
#define PI_2		1.57079632679489661923
#define PI_4		0.785398163397448309616
#define PI_3_4		2.356194490192344
#define PI_5_4		3.926990816987241
#define PI_7_4		5.497787143782138
#define TWOPI       6.283185307179586
#endif


#define FORCEINLINE 1
#if FORCEINLINE
#define INLINE __forceinline__
#else
#define INLINE inline
#endif

#if DEBUG
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }
// Same function as CUDA_CHECK_RETURN
#define CUDA_SAFE_CALL(call) do{ cudaError_t err = call; if (cudaSuccess != err) {  fprintf (stderr, "Cuda error in file '%s' in line %i : %s.", __FILE__, __LINE__, cudaGetErrorString(err) );  exit(EXIT_FAILURE);  } } while (0)
#else
#define CUDA_CHECK_RETURN(value) {value;}
#define CUDA_SAFE_CALL(value) {value;}
#endif



typedef unsigned char byte;
typedef thrust::device_vector<float> d_vec_t;
typedef thrust::host_vector<float> h_vec_t;


#ifndef nullptr
#define nullptr NULL
#endif

INLINE __host__ __device__ const float2 operator/(const float2& a, float b)
{
	return make_float2(a.x / b, a.y / b);
}

INLINE __host__ __device__ const float3 operator+(const float3& a, const float3& b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

INLINE __host__ __device__ const float3 operator-(const float3& a, const float3& b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

INLINE __host__ __device__ const double3 operator-(const double3& a, const double3& b)
{
	return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}
INLINE __host__ __device__ const float2 operator-(const float2& a, const float2& b)
{
	return make_float2(a.x - b.x, a.y - b.y);
}

INLINE __host__ __device__ const float3 operator*(const float3& a, const float3& b)
{
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

INLINE __host__ __device__ const float3 operator*(const float3& a, float b)
{
	return make_float3(a.x * b, a.y * b, a.z * b);
}

INLINE __host__ __device__ const float3 operator/(const float3& a, const float3& b)
{
	return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

INLINE __host__ __device__ const float3 operator/(const float3& a, float b)
{
	return make_float3(a.x / b, a.y / b, a.z / b);
}


INLINE __host__ __device__ const double3 operator/(const double3& a, double b)
{
	return make_double3(a.x / b, a.y / b, a.z / b);
}


INLINE __host__ __device__ const float3 operator-(const int3& a, const float3& b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

INLINE __host__ __device__ float length(const float2& a)
{
	return sqrtf(a.x * a.x + a.y * a.y);
}

INLINE __host__ __device__ float length(const float3& a)
{
	return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}

INLINE __host__ __device__ double length(const double3& a)
{
	return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}


INLINE __host__ __device__ const float2 normalize(const float2& a)
{
	return a / length(a);
}

INLINE __host__ __device__ const float3 normalize(const float3& a)
{
	return a / length(a);
}

INLINE __host__ __device__ const double3 normalize(const double3& a)
{
	return a / length(a);
}

INLINE __host__ __device__ float fminf(const float2& a)
{
	return fminf(a.x, a.y);
}

INLINE __host__ __device__ float fminf(const float3& a)
{
	return fminf(a.x, fminf(a.y, a.z));
}

INLINE __host__ __device__ float fmaxf(const float2& a)
{
	return fmaxf(a.x, a.y);
}

INLINE __host__ __device__ float fmaxf(const float3& a)
{
	return fmaxf(a.x, fmaxf(a.y, a.z));
}

INLINE __host__ __device__ const float3 fminf(const float3& a, const float3& b)
{
	return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

INLINE __host__ __device__ const float3 fmaxf(const float3& a, const float3& b)
{
	return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

INLINE __host__ __device__ const float2 fminf(const float2& a, const float2& b)
{
	return make_float2(fminf(a.x, b.x), fminf(a.y, b.y));
}

INLINE __host__ __device__ const float2 fmaxf(const float2& a, const float2& b)
{
	return make_float2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
}

template<typename T>
INLINE __host__ __device__ T lerp(T v0, T v1, T t)
{
	return fma(t, v1, fma(-t, v0, v0));
}

template<typename T>
INLINE __host__ __device__ T bilierp(T v0, T v1, T v2, T v3, T t1, T t2)
{
	T vv0 = fma(t1, v1, fma(-t1, v0, v0));
	T vv1 = fma(t1, v3, fma(-t1, v2, v2));
	return fma(t2, vv1, fma(-t2, vv0, vv0));
}


INLINE __device__ double bilerp(int2 v0, int2 v1, int2 v2, int2 v3, float t1, float t2)
{
	double v0_ = __hiloint2double(v0.y, v0.x);
	double v1_ = __hiloint2double(v1.y, v1.x);
	double v2_ = __hiloint2double(v2.y, v2.x);
	double v3_ = __hiloint2double(v3.y, v3.x);

	double vv0 = v0_ * (1.0 - t1) + v1_ * t1;
	double vv1 = v2_ * (1.0 - t1) + v3_ * t1;
	return vv0 * (1 - t2) + vv1 * t2;
}


INLINE __device__ double bilerp(int2 v0, int2 v1, int2 v2, int2 v3, double t1, double t2)
{
	double v0_ = __hiloint2double(v0.y, v0.x);
	double v1_ = __hiloint2double(v1.y, v1.x);
	double v2_ = __hiloint2double(v2.y, v2.x);
	double v3_ = __hiloint2double(v3.y, v3.x);

	double vv0 = v0_ * (1.0 - t1) + v1_ * t1;
	double vv1 = v2_ * (1.0 - t1) + v3_ * t1;
	return vv0 * (1 - t2) + vv1 * t2;
}



INLINE __host__ __device__ bool intersectBox(
	const float3& sour,
	const float3& dir,
	const float3& boxmin,
	const float3& boxmax,
	float* tnear, float* tfar)
{
	const float3 invR = make_float3(1.0 / dir.x, 1.0 / dir.y, 1.0 / dir.z);
	const float3 tbot = invR * (boxmin - sour);
	const float3 ttop = invR * (boxmax - sour);

	const float3 tmin = fminf(ttop, tbot);
	const float3 tmax = fmaxf(ttop, tbot);

	const float largest_tmin = fmaxf(tmin);
	const float smallest_tmax = fminf(tmax);
	*tnear = largest_tmin;
	*tfar = smallest_tmax;
	return smallest_tmax > largest_tmin;
}

template<typename T>
INLINE __host__ __device__ T regularizeAngle(T curang)
{
	T c = curang;
	while (c >= TWOPI){ c -= TWOPI; }
	while (c < 0){ c += TWOPI; }
	return c;
}


INLINE __host__ __device__ void invRotVox(
	const float3& curVox,
	float3& virVox,
	const float2& cossinT,
	const float zP)
{
	virVox.x = curVox.x * cossinT.x + curVox.y * cossinT.y;
	virVox.y =-curVox.x * cossinT.y + curVox.y * cossinT.x;
	virVox.z = curVox.z - zP;
}

INLINE __device__ float3 invRot(
	const float3 inV,
	const float2 cossin,
	const float zP)
{
	float3 outV;
	outV.x = inV.x * cossin.x + inV.y * cossin.y;
	outV.x =-inV.x * cossin.y + inV.y * cossin.x;
	outV.z = inV.z - zP;
	return outV;
}


namespace CTMBIR
{
	struct ConstantForBackProjection3
	{
		float x0;
		float y0;
		float z0;

		typedef thrust::tuple<float, float> InTuple;
		typedef thrust::tuple<float3, float3, float2> OutTuple;
		ConstantForBackProjection3(
			const float _x0, const float _y0, const float _z0) :x0(_x0),
			y0(_y0), z0(_z0){}

		__device__ OutTuple operator()(const InTuple& tp)
		{
			float curang = regularizeAngle(thrust::get<0>(tp));
			float zP = thrust::get<1>(tp);
			float cosT = cosf(curang);
			float sinT = sinf(curang);

			float3 cursour = make_float3(
				x0 * cosT - y0 * sinT,
				x0 * sinT + y0 * cosT,
				z0 + zP);

			float2 dirsour = normalize(make_float2(-cursour.x, -cursour.y));
			return thrust::make_tuple(make_float3(cosT, sinT, zP), cursour, dirsour);

		}

	};

	struct ConstantForBackProjection4{

		float x0;
		float y0;
		float z0;

		typedef thrust::tuple<float, float> InTuple;
		ConstantForBackProjection4(const float _x0, const float _y0, const float _z0)
			: x0(_x0), y0(_y0), z0(_z0){}

		__device__ float3 operator()(const InTuple& tp)
		{
			float curang = regularizeAngle(thrust::get<0>(tp));
			float zP = thrust::get<1>(tp);
			float cosT = cosf(curang);
			float sinT = sinf(curang);
			return make_float3(cosT, sinT, zP);
		}

	};

	struct Constant_MultiSlice
	{
		float x0;
		float y0;
		Constant_MultiSlice(const float _x0, const float _y0):x0(_x0),y0(_y0){}

		__device__ float2 operator()(const float& tp)
		{
			float curang = regularizeAngle(tp);
			return make_float2(cosf(curang), sinf(curang));

		}

	};

	template<typename T>
	struct ConstantForBackProjection{

		T x0;
		T y0;
		T z0;

		typedef thrust::tuple<T, T> InTuple;
		ConstantForBackProjection(const T _x0, const T _y0, const T _z0)
			: x0(_x0), y0(_y0), z0(_z0){}

		__device__ float3 operator()(const InTuple& tp)
		{
			T curang = regularizeAngle(thrust::get<0>(tp));
			T zP = thrust::get<1>(tp);
			T cosT = cosf(curang);
			T sinT = sinf(curang);
			return make_float3(cosT, sinT, zP);
		}
	};


	template<>
	struct ConstantForBackProjection<double>{

		double x0;
		double y0;
		double z0;

		typedef thrust::tuple<double, double> InTuple;
		ConstantForBackProjection(const double _x0, const double _y0, const double _z0)
			: x0(_x0), y0(_y0), z0(_z0){}

		__device__ double3 operator()(const InTuple& tp)
		{
			double curang = regularizeAngle(thrust::get<0>(tp));
			double zP = thrust::get<1>(tp);
			double cosT = cos(curang);
			double sinT = sin(curang);
			return make_double3(cosT, sinT, zP);
		}
	};

}


template<typename T>
void DD3Boundaries(int nrBoundaries, T*pCenters, T *pBoundaries)
{
	int i;
	if (nrBoundaries >= 3)
	{
		*pBoundaries++ = 1.5 * *pCenters - 0.5 * *(pCenters + 1);
		for (i = 1; i <= (nrBoundaries - 2); i++)
		{
			*pBoundaries++ = 0.5 * *pCenters + 0.5 * *(pCenters + 1);
			pCenters++;
		}
		*pBoundaries = 1.5 * *pCenters - 0.5 * *(pCenters - 1);
	}
	else
	{
		*pBoundaries = *pCenters - 0.5;
		*(pBoundaries + 1) = *pCenters + 0.5;
	}

}




///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
// Get one sub-volume from the whole volume.
// Assume that the volumes are stored in Z, X, Y order
template<typename T>
void getSubVolume(const T* vol,
		const size_t XN, const size_t YN, const size_t ZN,
		const size_t ZIdx_Start, const size_t ZIdx_End, T* subVol)
{
	const size_t SZN = ZIdx_End - ZIdx_Start;
	for (size_t yIdx = 0; yIdx != YN; ++yIdx)
	{
		for (size_t xIdx = 0; xIdx != XN; ++xIdx)
		{
			for (size_t zIdx = ZIdx_Start; zIdx != ZIdx_End; ++zIdx)
			{
				subVol[(yIdx * XN + xIdx) * SZN + (zIdx - ZIdx_Start)] = vol[(yIdx * XN + xIdx) * ZN + zIdx];
			}
		}
	}
}

template<typename T>
void getSubVolume(const T* vol,
		const size_t XYN, const size_t ZN,
		const size_t ZIdx_Start, const size_t ZIdx_End, T* subVol)
{
	const size_t SZN = ZIdx_End - ZIdx_Start;
	for (size_t xyIdx = 0; xyIdx != XYN; ++xyIdx)
	{
		for (size_t zIdx = ZIdx_Start; zIdx != ZIdx_End; ++zIdx)
		{
			subVol[xyIdx * SZN + (zIdx - ZIdx_Start)] = vol[xyIdx * ZN + zIdx];
		}
	}
}
///////////////////////////////////////////////////////////////////////////////////

// For projection, before we divide the volume into serveral sub-volumes, we have
// to calculate the Z index range
template<typename T>
void getVolZIdxPair(const thrust::host_vector<T>& zPos, // Z position of the source.
		//NOTE: We only assume the spiral CT case that zPos is increasing.
		const size_t PrjIdx_Start, const size_t PrjIdx_End,
		const T detCntIdxV, const T detStpZ, const int DNV,
		const T objCntIdxZ,	const T dz, const int ZN, // Size of the volume
		int& ObjIdx_Start, int& ObjIdx_End) // The end is not included
{
	const T lowerPart = (detCntIdxV + 0.5) * detStpZ;
	const T upperPart = DNV * detStpZ - lowerPart;
	const T startPos = zPos[PrjIdx_Start] - lowerPart;
	const T endPos = zPos[PrjIdx_End - 1] + upperPart;

	ObjIdx_Start = floor((startPos / dz) + objCntIdxZ - 1);
	ObjIdx_End = ceil((endPos / dz) + objCntIdxZ + 1) + 1;

	ObjIdx_Start = (ObjIdx_Start < 0) ? 0 : ObjIdx_Start;
	ObjIdx_Start = (ObjIdx_Start > ZN) ? ZN : ObjIdx_Start;

	ObjIdx_End = (ObjIdx_End < 0) ? 0 : ObjIdx_End;
	ObjIdx_End = (ObjIdx_End > ZN) ? ZN : ObjIdx_End;
}

///////////////////////////////////////////////////////////////////////////////////
// For backprojection, after decide the subvolume range, we have to decide the
// projection range to cover the subvolume.
template<typename T>
void getPrjIdxPair(const thrust::host_vector<T>& zPos, // Z Position of the source.
		// NOTE: we assume that it is pre sorted
		const size_t ObjZIdx_Start, const size_t ObjZIdx_End, // sub vol range,
		// NOTE: the objZIdx_End is not included
		const T objCntIdxZ, const T dz, const int ZN,
		const T detCntIdxV, const T detStpZ, const int DNV,
		int& prjIdx_Start, int& prjIdx_End)
{
	const int PN = zPos.size();

	const T lowerPartV = (ObjZIdx_Start - objCntIdxZ - 0.5) * dz;
	const T highrPartV = lowerPartV + (ObjZIdx_End - ObjZIdx_Start) * dz;

	const T lowerPartDet = (detCntIdxV + 0.5) * detStpZ;
	const T upperPartDet = DNV * detStpZ - lowerPartDet;

	//The source position
	const T sourLPos = lowerPartV - upperPartDet;
	const T sourHPos = highrPartV + lowerPartDet;

	prjIdx_Start = thrust::upper_bound(zPos.begin(),zPos.end(),sourLPos) - zPos.begin() - 1;
	prjIdx_End = thrust::upper_bound(zPos.begin(),zPos.end(),sourHPos) - zPos.begin() + 2;
	prjIdx_Start = (prjIdx_Start < 0) ? 0 : prjIdx_Start;
	prjIdx_Start = (prjIdx_Start > PN)? PN: prjIdx_Start;

	prjIdx_End = (prjIdx_End < 0) ? 0 : prjIdx_End;
	prjIdx_End = (prjIdx_End > PN) ? PN : prjIdx_End;
}


////////////////////////////////////////////////////////////////////////////////////
// The volume is also stored in Z, X, Y order
// Not tested yet.
template<typename T>
void combineVolume(
	T* vol, // The volume to be combined
	const int XN, const int YN, const int ZN,
	T** subVol, // All sub volumes
	const int* SZN, // Number of slices for each subVolume
	const int subVolNum) // Number of sub volumes
{
	int kk = 0;
	for (size_t yIdx = 0; yIdx != YN; ++yIdx)
	{
		for (size_t xIdx = 0; xIdx != XN; ++xIdx)
		{
			kk = 0;
			for (size_t volIdx = 0; volIdx != subVolNum; ++volIdx)
			{
				for (size_t zIdx = 0; zIdx != SZN[volIdx]; ++zIdx)
				{
					vol[(yIdx * XN + xIdx) * ZN + kk] = subVol[volIdx][(yIdx * XN + xIdx) * SZN[volIdx] + zIdx];
					kk = kk + 1;
				}
			}
		}
	}
}



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
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    const auto processorCount = std::thread::hardware_concurrency();

	// Divide the volume into three parts
    int* SLNn = new int[nDevices];
    float** shvol = new float*[nDevices];
    float** shprj = new float*[nDevices];
    float** shang = new float*[nDevices];
    for(int n = 0; n < nDevices; ++n) {
        if(n == nDevices-1) {
            SLNn[n] = SLN - startIdx[n];
        } else {
            SLNn[n] = startIdx[n+1] - startIdx[n];
        }
        shvol[n] = new float[XN * YN * SLNn[n]];
        shprj[n] = new float[DNU * SLNn[n] * PN];
        shang[n] = new float[PN * SLNn[n]];
    }
	
	
	for(int pIdx = 0; pIdx != PN; ++pIdx) {
		for(int sIdx = 0; sIdx != SLN; ++sIdx) {
            for(int n = 0; n < nDevices; ++n) {
                if(n == nDevices-1) {
                    if(sIdx >= startIdx[n] && sIdx < SLN) {
                        shang[n][pIdx * SLNn[n] + (sIdx - startIdx[n])] = hangs[pIdx * SLN + sIdx];
                    }

                } else {
                    if(sIdx >= startIdx[n] && sIdx < startIdx[n+1]) {
                        shang[n][pIdx * SLNn[n] + (sIdx - startIdx[n])] = hangs[pIdx * SLN + sIdx];
                    }
                }
            }
		}	
	}

	//Split the projection
	omp_set_num_threads(processorCount);
#pragma omp parallel for
	for(int i = 0; i < DNU * PN; ++i) {
		for(int v = 0; v != SLN; ++v) {
            for(int n = 0; n < nDevices; ++n) {
                if(n == nDevices-1) {
                    if(v >= startIdx[n] && v < SLN) {
                        shprj[n][i * SLNn[n] + (v - startIdx[n])] = hprj[i * SLN + v];
                    }
                } else {
                    if(v >= startIdx[n] && v < startIdx[n+1]) {
                        shprj[n][i * SLNn[n] + (v - startIdx[n])] = hprj[i * SLN + v];
                    }
                }
            }
		}
	}

    cudaDeviceSynchronize();
	omp_set_num_threads(nDevices);
#pragma omp parallel for
	for(int i = 0; i < nDevices; ++i)
	{
		DD2Back_gpu(x0, y0, DNU, xds, yds, detCntIdx,
			imgXCenter, imgYCenter, shang[i],
			PN, XN, YN, SLNn[i], shvol[i], shprj[i], dx,mask, i);
	}

	//Gather all

	omp_set_num_threads(processorCount);
#pragma omp parallel for
	for(int i = 0; i < XN * YN; ++i) {
		for(int v = 0; v != SLN; ++v) {
            float val = 0;
            
            for(int n = 0; n < nDevices; ++n) {
                if(n == nDevices - 1) {
                    if(v >= startIdx[n] && v < SLN) {
                        val = shvol[n][i * SLNn[n] + (v - startIdx[n])];
                    }
                } else {
                    if(v >= startIdx[n] && v < startIdx[n+1]) {
                        val = shvol[n][i * SLNn[n] + (v - startIdx[n])];
                    }
                }
            }
			hvol[i * SLN + v] = val;
		}
    }
    
    for(int n = 0; n < nDevices; ++n) {
        delete[] shprj[n];
        delete[] shvol[n];
    }
	delete[] shprj;
	delete[] shvol;
	delete[] SLNn;

    //cudaDeviceReset();
    //cudaDeviceSynchronize();
}


void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    float x0 = *((float*)mxGetData(prhs[0]));
    float y0 = *((float*)mxGetData(prhs[1]));
    int DNU = *((int*)mxGetData(prhs[2]));
    float* xds = (float*)mxGetPr(prhs[3]);
    float* yds = (float*)mxGetPr(prhs[4]);
    float detCntIdx = *((float*)mxGetData(prhs[5]));
    float imgXCenter = *((float*)mxGetData(prhs[6]));
    float imgYCenter = *((float*)mxGetData(prhs[7]));
    float* hangs = (float*)mxGetPr(prhs[8]);
    int PN = *((int*)mxGetData(prhs[9]));
    int XN = *((int*)mxGetData(prhs[10]));
    int YN = *((int*)mxGetData(prhs[11]));
    int SLN = *((int*)mxGetData(prhs[12]));
    float* hprj = (float*)mxGetPr(prhs[13]);
    float dx = *((float*)mxGetData(prhs[14]));
    byte* mask = (byte*)mxGetPr(prhs[15]);
    int* startIdx = (int*)mxGetPr(prhs[16]);
       
    const mwSize dims[] = {SLN, XN, YN};
    plhs[0] = mxCreateNumericArray(3,dims,mxSINGLE_CLASS,mxREAL);
    float* hvol = (float*)mxGetPr(plhs[0]);
    
    DD2Back_3gpus(x0, y0, DNU, xds, yds,detCntIdx, imgXCenter, imgYCenter,
		hangs, PN, XN, YN, SLN, hvol, hprj, dx, mask, startIdx);
    
}


