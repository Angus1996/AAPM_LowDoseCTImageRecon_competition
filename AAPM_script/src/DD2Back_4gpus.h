/*
 * DD2_GPU_back.h
 *
 *  Created on: Apr 4, 2016
 *      Author: liurui
 */

#ifndef DD2_GPU_BACK_H_
#define DD2_GPU_BACK_H_

typedef unsigned char byte;

extern "C"
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
	byte* mask, int gpunum);

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
	byte* mask, int* startIdx);


extern "C"
void DD2Back_4gpus(
	float x0, float y0,
	int DNU,
	float* xds, float* yds,
	float detCntIdx,
	float imgXCenter, float imgYCenter,
	float* hangs, int PN,
	int XN, int YN, int SLN,
	float* hvol, float* hprj,
	float dx,
	byte* mask, int* startIdx);
#endif /* DD2_GPU_BACK_H_ */
