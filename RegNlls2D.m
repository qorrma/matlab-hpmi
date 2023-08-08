function y = RegNlls2D(x, verbose)
[h,w] = size(x);
LIDX = 1:numel(x);
[row, col] = ind2sub(size(x), LIDX);
L = zeros(size(LIDX));

for i=1:1:numel(x)
    L(i) = x(row(i), col(i));
end

p = row-(h/2)-1;
q = col-(w/2)-1;
curve = fit([p', q'], L', 'poly22');
output = curve([p', q']);
y = reshape(output, [h, w]);

AvgVal = mean(y(:));
MinVal = min(y(:));
MaxVal = max(y(:));

if verbose == 1
    fprintf('[Base estimation] MinVal: %.04f, MaxVal: %.04f, AvgVal: %.04f\n', MinVal, MaxVal, AvgVal);
end

end