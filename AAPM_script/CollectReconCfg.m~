% Wake Forest Health Sciences
% author: Rui Liu
% Date: Apr. 7, 2016
% The routine that we gather the reconstruction information from the
% projection data
% Input:
%   FileName    : The projection data provided by AAPM in 'dicom'.
%
% Output:
%   cfg         : The necessary information for projection/backprojection
% -----------------------------------------------------------------------
function [cfg] = CollectReconCfg(FileName)

%% Read the information of the data
info = dicominfo(FileName);

%% HU Calibration Factor. A calibration factor \mu' for the conversion of
% measured linear attenuation coefficients \mu to CT number (mm^-1); CT
% number =1000 * (\mu - \mu') / \mu';
cfg.HUCalibrationFactor = cov_uint8_to_double(info.Unknown_0018_0061); 

%% The scan field of view
cfg.DataCollectionDiameter = info.DataCollectionDiameter;

%% The pitch
cfg.SpiralPitchFactor = info.SpiralPitchFactor; %0.6 in L067 data


%% The number of detector rows
cfg.NumberofDetectorRows = double(info.Private_7029_1010(2)) * 2^8 +... 
                           double(info.Private_7029_1010(1));

%% The number of detector columns
cfg.NumberofDetectorColumns = double(info.Private_7029_1011(2)) * 2^8 +...
                              double(info.Private_7029_1011(1));
                          
%% Detector Element Transverse Spacing
cfg.DetectorElementTransverseSpacing = cov_4uint8_to_float(info.Private_7029_1002);

%% Detector Element Axial Spacing
cfg.DetectorElementAxialSpacing = cov_4uint8_to_float(info.Private_7029_1006);

%% Detector Shape (char type)
cfg.DetectorType = char(info.Private_7029_100b)';

%% Detector Focal Center Angular Position, \phi_0 the azimuthal angles of
% the detector's focal center (rad)
cfg.DetectorFocalCenterAngularPosition = cov_4uint8_to_float(info.Private_7031_1001);

%% Detector Focal Center Axial Position, z_0 the z location of the detector's focal center
% the in-plane distances
% between the detector's focal center and the isocenter (mm)
cfg.DetectorFocalCenterAxialPosition = cov_4uint8_to_float(info.Private_7031_1002);

%% Detector Focal Center Radial Distance, \rho_0, the in plane distances
% between the detector's focal center and the isocenter
cfg.DetectorFocalCenterRadialDistance = cov_4uint8_to_float(info.Private_7031_1003);

%% Detector Central Element: (Column X, Row Y), the index of the detector
% element aligning with the isocenter and the detector's focal center
[cfg.DetectorCentralElement.X, cfg.DetectorCentralElement.Y] = cov_8uint8_to_2floats(info.Private_7031_1033);


%% Constant Radial Distance, d_0, the distance between the detector's focal
% center and the detector element specified in Tag(7031,1033) (mm)
% The index start with 1, therefore, when we call the GPU function, we 
% should minus 1, here, the number is not minused.
cfg.ConstantRadialDistance = cov_4uint8_to_float(info.Private_7031_1031); 

%% Source Angular Position Shift
% \delta\phi, the \phi offset from the focal sport to the detector's focal
% center (rad) % Maybe not used
cfg.SourceAngularPositionShift = cov_4uint8_to_float(info.Private_7033_100b);

%% Source Axial Position shift
% \delte z, the z offset fro mthe focal spot to the detector's focal center
% (mm). MAYBE NOT USED BUT NOT SURE
cfg.SourceAxialPositionShift = cov_4uint8_to_float(info.Private_7033_100c); % in L067 info, it is 3.3613e-4

%% Source Radial Distance Shift
% \delta\rho, the \rho offset from the focal spot to the detector's focal
% center (mm). MAYBE NOT USED BUT NOT SURE.
cfg.SourceRadialDistanceShift = cov_4uint8_to_float(info.Private_7033_100d);


%% Number of source angular steps.
% The number of projections per complete rotation
cfg.NumberofSourceAngularSteps = double(info.Private_7033_1013(2)) * 2^8 ...
                                +double(info.Private_7033_1013(1));

%% Photon Statistics
% An array describing the spatial distribution of photons along the
% direction of the detector columns, from column1 to column M ( neglecting
% the variation across detector rows). Each element of the array
% corresponds to a detector column.
cfg.PhotonStatistics = cov_uint8_to_PhoStat(info.Private_7033_1065, int32(cfg.NumberofDetectorColumns));

%% GE Pitch definition
cfg.SpiralPitchGE =  cfg.SpiralPitchFactor * cfg.NumberofDetectorRows;

%% Projection data rescale data
cfg.RescaleIntercept = info.RescaleIntercept;
cfg.RescaleSlope = info.RescaleSlope;
