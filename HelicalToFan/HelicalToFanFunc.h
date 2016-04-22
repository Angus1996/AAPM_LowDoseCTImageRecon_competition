/*
 * HelicalToFan.h
 *
 *  Created on: Apr 21, 2016
 *      Author: liurui
 */

#ifndef HELICALTOFAN_H_
#define HELICALTOFAN_H_

extern "C"
void HelicalToFan(
		float* Proj,  					// rebinned projection; in order (Channel, View Index, Slice Index)  TODO: need permute after call by MATLAB
		float* Views, 					// rebinned views (View Index, Slice Index) TODO: need permute after call by MATLAB
		float* proj,  		// raw projection data In order : (Height Index, Channel Index, View Index(Total View))
		float* zPos,  		// sampling position
		const int SLN,                  // slice number
		const float SD, 				// source to detector distance
		const float SO,					// source to iso-center distance
		const float BVAngle,        	// The begin view
		const int DetWidth,         	// number of detector columns
		const int DetHeight,        	// number of detector rows
		const float PerDetW,        	// Detector cell size along channel direction
		const float PerDetH,        	// Detector cell size along bench moving direction
		const int DefTimes,         	// Number of views per rotation
		const float DetCenterW,     	// Detector Center Index
		const float SpiralPitchFactor 	// Pitch defined in SIEMENS
		);


#endif /* HELICALTOFAN_H_ */
