%-------------------------------------------------------------------------
% Wake Forest Health Sciences
% Date: Apr, 8, 2016
% Routine: reconscript
% Authors:
%
%   Rui Liu (Wake Forest Health)
% Organization:
% Wake Forest Health Sciences & University of Massachusetts Lowell
%
% Aim:
%   The reconstruction script initially defined
%
% Input/Output:
%--------------------------------------------------------------------------
clear;
% compileFiles;
haveTheFile = 1;
if ~haveTheFile
    %% Read the projection
    [proj, cfg] = readProj('../full_DICOM-CT-PD');
    
    %% Read the image information
    cfgRecon = CollectImageCfg('../L067_FD_1_1.CT.0001.0001.2015.12.22.18.09.40.840353.358074219.IMA');
else
    load('Proj.mat');
end

%% Define the number of slices
SLN = 512;

%% Define according to Dr. Yu's suggestion
cfg.DetectorCentralElement.X = cfg.NumberofDetectorColumns + 1 - cfg.DetectorCentralElement.X;

% %% The reconstruction configuration
conf = ConvertToReconConf(cfg, cfgRecon, SLN);
conf.recon.dx = 1;
conf.recon.XN = 512;
conf.recon.YN = 512;


%% Define the Z positions
% % h = cfg.SpiralPitchFactor * cfg.DetectorElementAxialSpacing * cfg.NumberofDetectorRows * cfg.DetectorFocalCenterRadialDistance / cfg.ConstantRadialDistance;
%% Define the Z positions (According to Yu's suggestions)
h = cfg.SpiralPitchFactor * cfg.DetectorElementAxialSpacing * cfg.NumberofDetectorRows;

deltaZ = h / cfg.NumberofSourceAngularSteps;
TotalView = cfg.NumOfDataViews;

zbegin = 67; % Define the range
zend = 514.2;

zPos = linspace(zbegin, zend, SLN) - cfg.DetectorFocalCenterAxialPosition; % Specific for L067.

%% Rebinning the projection with baodong's parameter
%[Proj, Views] = HelicalToFan_naive(proj, cfg, zPos);
[Proj, Views] = HelicalToFan_routine(proj,cfg,zPos);


%% OS-SART
mask = zeros(conf.recon.XN,conf.recon.YN);
for ii = 1 : conf.recon.XN
    for jj = 1 : conf.recon.YN
        if sqrt(((double(ii) - 0.5 - double(conf.recon.XN) / 2) / (double(conf.recon.XN) / 2))^2 +...
                ((double(jj) - 0.5 - double(conf.recon.YN) / 2) / (double(conf.recon.YN) / 2))^2) < 1.3
            mask(ii,jj) = 1;
        end
           
    end
end
mask = uint8(mask);

useFISTA = 1;
initImg = single(zeros(conf.SLN,conf.recon.XN,conf.recon.YN));
numOfOS_series = [40, 20, 10, 8, 6, 4, 1];
numOfIter_series = [1, 4, 4, 4, 5, 6, 7];
tic;
for ii = 1 : 1
    reconImg = OSSART_AAPM(Proj, Views, initImg, conf, numOfIter_series(ii), numOfOS_series(ii), mask, useFISTA);    
    initImg = reconImg;
end
toc;

