function [cfg] = CollectImageCfg(FileName)%, SliceNum)

cfgRecon = dicominfo(FileName);

cfg.Width = cfgRecon.Width;
cfg.Height = cfgRecon.Height;
cfg.BitDepth = cfgRecon.BitDepth;
cfg.SliceThickness = cfgRecon.SliceThickness;
cfg.KVP = cfgRecon.KVP;
cfg.DataCollectionDiameter = cfgRecon.DataCollectionDiameter;
cfg.ReconstructionDiameter = cfgRecon.ReconstructionDiameter;
cfg.SingleCollimationWidth = cfgRecon.SingleCollimationWidth;
cfg.TotalCollimationWidth = cfgRecon.TotalCollimationWidth;
cfg.TableSpeed = cfgRecon.TableSpeed;
cfg.TableFeedPerRotation = cfgRecon.TableFeedPerRotation;
cfg.SpiralPitchFactor = cfgRecon.SpiralPitchFactor;
cfg.DataCollectionCenterPatient = cfgRecon.DataCollectionCenterPatient;
cfg.ReconstructionTargetCenterPatient = cfgRecon.ReconstructionTargetCenterPatient;
cfg.Rows = cfgRecon.Rows;
cfg.Columns = cfgRecon.Columns;
cfg.PixelSpacing = cfgRecon.PixelSpacing;
cfg.SmallestImagePixelValue = cfgRecon.SmallestImagePixelValue;
cfg.LargestImagePixelValue = cfgRecon.LargestImagePixelValue;
cfg.WindowCenter = cfgRecon.WindowCenter;
cfg.WindowWidth = cfgRecon.WindowWidth;
cfg.RescaleIntercept = cfgRecon.RescaleIntercept;
cfg.RescaleSlope = cfgRecon.RescaleSlope;
%cfg.SliceNum = SliceNum;
cfg.PixelSize = cfgRecon.ReconstructionDiameter / cfgRecon.Width;

end




