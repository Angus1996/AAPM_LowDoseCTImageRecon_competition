/*
 * Wake Forest Health Sciences & University of Massachusetts Lowell
 * Organization: 
 *  Wake Forest Health Sciences
 *
 * DD2_multislice_mex.cpp
 * Matlab mex gateway routine for the GPU based multi-slice distance-driven
 * fan-beam projector
 *
 * author: Rui Liu (Wake Forest Health Sciences)
 * date: 2016.04.06
 * version: 1.0
 */

#include "mex.h"
#include "matrix.h"
#include <cstring>
#include <iostream>

#include "DD2Back_4gpus.h"
typedef unsigned char byte;

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
