clear;
compileCOVfuncs; %% Compile the CPP file

%% Read the projection  (NOTE: PLEASE CHANGE THE PATH MANUALLY)
[proj, cfg] = readProj('/home/ruiliu/Desktop/CTRecon_old/full_DICOM-CT-PD');

%% Read the image information (NOTE: PLEASE CHANGE THE PATH MANUALLY)
cfgRecon = CollectImageCfg('Info.IMA');

% load('testReconstruction.mat');
%% Define the number of slices
SLN = 512;

%% The reconstruction configuration
conf = ConvertToReconConf(cfg, cfgRecon, SLN);
conf.recon.dx = 1; % It is suggested being 0.674, I just do the test for the feasibility of OS-SART here.
%% Define the Z positions
h = cfg.SpiralPitchFactor * cfg.DetectorElementAxialSpacing * cfg.NumberofDetectorRows * cfg.DetectorFocalCenterRadialDistance / cfg.ConstantRadialDistance;
deltaZ = h / cfg.NumberofSourceAngularSteps;

TotalView = cfg.NumOfDataViews;
zPos = linspace(h, deltaZ * TotalView - h, SLN);

%% Rebinning the projection with baodong's parameter
[Proj, Views] = HelicalToFan_routine(proj, cfg, zPos);

%% mask
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

numOfIter = 16;
numOfOS = 5;
useFISTA = 1;
initImg = single(zeros(conf.SLN,conf.recon.XN,conf.recon.YN));
reconImg = OSSART_AAPM(Proj, Views, initImg, conf, numOfIter, numOfOS, mask, useFISTA);
