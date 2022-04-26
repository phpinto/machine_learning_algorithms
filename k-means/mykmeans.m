function [ class, centroid ] = mykmeans( pixels, K )
%
% Your goal of this assignment is implementing your own K-means.
%
% Input:
%     pixels: data set. Each row contains one data point. For image
%     dataset, it contains 3 columns, each column corresponding to Red,
%     Green, and Blue component.
%
%     K: the number of desired clusters. Too high value of K may result in
%     empty cluster error. Then, you need to reduce it.
%
% Output:
%     class: the class assignment of each data point in pixels. The
%     assignment should be 1, 2, 3, etc. For K = 5, for example, each cell
%     of class should be either 1, 2, 3, 4, or 5. The output should be a
%     column vector with size(pixels, 1) elements.
%
%     centroid: the location of K centroids in your result. With images,
%     each centroid corresponds to the representative color of each
%     cluster. The output should be a matrix with size(pixels, 1) rows and
%     3 columns. The range of values should be [0, 255].
%     
%
% You may run the following line, then you can see what should be done.
% For submission, you need to code your own implementation without using
% the kmeans matlab function directly. That is, you need to comment it out.

%	[class, centroid] = kmeans(pixels, K);
    class = zeros(length(pixels), 1);
    centroid = zeros(K,3);
    
    % Random initizliation of Centroids
    
    for i = 1 : K
       rand_k = rand([1,3]) * 255;
       centroid(i,:) = rand_k;
    end
    
    old_centroid = -1.*centroid;
    
    % E-M K-means Algorithm implementation:
    tic
    loop = 0;
    while (~isequal(centroid,old_centroid))
        old_centroid = centroid;
        % Maximization Step:
        for j = 1:length(pixels)
            cluster_center = which_k(pixels(j,:)',centroid');
            class(j) = cluster_center;
        end
      
        % Expectation Step:
        for k = 1 : K
            cluster_point_locations = find(class == k);
            if isempty(cluster_point_locations)
                centroid(k,:) = rand([1,3]) * 255;         
            else
                cluster_points = pixels(cluster_point_locations,:);
                average_point = mean(cluster_points);
                centroid(k,:) = average_point;
            end
        end
        loop = loop + 1;
    end
    toc
    time_per_iteration = toc/loop;
    disp('Running time:')
    disp(toc)
    disp('Total number of iterations:')
    disp(loop)
    disp('Time per iteration:')
    disp(time_per_iteration)
end
 

%Function to determine to which cluster center a point belongs to using
%eucledian distance:

function k_index = which_k(point, k_array)
    [~,cols] = size(k_array);
    point_array = zeros(3,cols);
    for col = 1:cols
        point_array(:,col) = point;
    end 
    distance = point_array - k_array;
    distance = sum(distance.^2);
    distance = sqrt(distance);

    k_index = find(distance == min(distance));
    k_index = k_index(1);
end