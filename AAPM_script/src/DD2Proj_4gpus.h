/*
 * File Name: DD2_GPU_proj.h
 * Description: Multi-slice Fan-beam projection in BL-DD or none BL-DD models
 * Created on: Apr. 4, 2016
 * Author : Rui Liu
 */

#ifndef DD2_GPU_PROJ_H_
#define DD2_GPU_PROJ_H_

typedef unsigned char byte;

extern "C"
void DD2Proj_gpu(
		float x0, float y0,
		int DNU,
		float* xds, float* yds,
		float imgXCenter, float imgYCenter,
		float* hangs, int PN,
		int XN, int YN, int SLN,
		float* hvol, float* hprj,
		float dx,
		byte* mask, int gpunum);


extern "C"
void DD2Proj_3gpus(
		float x0, float y0,
		int DNU,
		float* xds, float* yds,
		float imgXCenter, float imgYCenter,
		float* hangs, int PN,
		int XN, int YN, int SLN,
		float* hvol, float* hprj,
		float dx, byte* mask, int* startIdx);

extern "C"
void DD2Proj_4gpus(
		float x0, float y0,
		int DNU,
		float* xds, float* yds,
		float imgXCenter, float imgYCenter,
		float* hangs, int PN,
		int XN, int YN, int SLN,
		float* hvol, float* hprj,
		float dx, byte* mask, int* startIdx);

#endif /* DD3_GPU_PROJ_H_ */
