function y = AdaptiveThreshold(x, mask, p_sz)

% Adaptive thresholding
pad = padarray((x-min(x(:)))./(max(x(:))-min(x(:))), [p_sz p_sz], 'symmetric');
th = zeros(size(pad));

for i=1+p_sz:1:size(pad,1)-p_sz
    for j=1+p_sz:1:size(pad,2)-p_sz
        kernel = pad(i-p_sz:i+p_sz, j-p_sz:j+p_sz);
        if mean(kernel(:))>0.15; th(i,j)=1; else; th(i,j)=0; end
    end
end

FG = th(1+p_sz:end-p_sz, 1+p_sz:end-p_sz).*mask; 
W = ones(3, 3);
FG = imdilate(FG, W);

[k_fg, n] = bwlabel(FG);
k_arr = zeros(n,1);
for i = 1:1:n
    ne = k_fg==i;
    nn = sum(ne(:));
    k_arr(i,1) = nn;
end

[~, idx] = max(k_arr);
y = k_fg==idx;
end