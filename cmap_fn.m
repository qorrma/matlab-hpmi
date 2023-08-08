function [cmap, nmap] = cmap_fn(mura, base)

cmap = abs(mura-base);
cmap = cmap./base;

MaxVal = max(cmap(:));
MinVal = min(cmap(:));

nmap = (cmap-MinVal)./(MaxVal-MinVal);
% fprintf('[Contrast map] MinVal: %.04f, MaxVal: %.04f\n', MinVal, MaxVal);
end
