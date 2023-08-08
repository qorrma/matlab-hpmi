function [Cx, Sx, Ax] = FeatsExtr(cmap, mask)
AverageContrast = mean(cmap(mask==1));

% Extract foreground's axis
s = regionprops(mask,'MajorAxisLength', 'MinorAxisLength', 'Centroid','Orientation');

M = pixels2length(s.MajorAxisLength);
m = pixels2length(s.MinorAxisLength);

% Centroid = s.Centroid;
% Orientation = s.Orientation;

Cx = gray2cd(AverageContrast);
Sx = m * M * pi;
Ax = M;
end

function cd = gray2cd(gray)
max_cd = 300;
cd = (max_cd * gray) / 255;
end

function length = pixels2length(num_pixels)
ppi = 139.8757;
ppcm = ppi / 2.54;
length = num_pixels / ppcm;
end