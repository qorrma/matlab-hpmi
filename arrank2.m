function output = arrank2(array, dim, flag)
[h, w] = size(array);
if strcmp(flag,'descend')==1
    if dim == 1
        output = zeros(size(array));
        for i=1:1:h
            [~, p] = sort(array(i,:), 'descend');
            rank =1:w;
            output(i, p) = rank;
        end
    else
        output = zeros(size(array));
        for i=1:1:w
            [~, p] = sort(array(:,i), 'descend');
            rank =1:h;
            output(i, p) = rank;
        end
    end
else
    if dim == 1
        output = zeros(size(array));
        for i=1:1:h
            [~, p] = sort(array(i,:), 'ascend');
            rank =1:w;
            output(i, p) = rank;
        end
    else
        output = zeros(size(array));
        for i=1:1:w
            [~, p] = sort(array(:,i), 'ascend');
            rank =1:h;
            output(i, p) = rank;
        end
    end
end