function [ class, centroid ] = mykmedoids( pixels, K )
%
% Your goal of this assignment is implementing your own K-medoids.
% Please refer to the instructions carefully, and we encourage you to
% consult with other resources about this algorithm on the web.
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


    class = zeros(length(pixels), 1);
    centroid = zeros(K,3);
    
    % Random initizliation of Centroids
    
    for i = 1 : K
       rand_k = round(rand(1) * length(pixels));
       centroid(i,:) = pixels(rand_k,:);
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
                centroid(k,:) = pixels(round(rand(1) * length(pixels)));         
            else
                cluster_points = pixels(cluster_point_locations,:);
                average_point = mean(cluster_points);
                dist_vector = zeros(1,length(cluster_points));
                for p = 1 : length(cluster_points)
                    dist_vector(p) = sqrt(sum((average_point - cluster_points(p,:)) .^ 2));
                    %dist_vector(p) = norm((average_point - cluster_points(p,:)),1);
                end
                min_index = find(dist_vector == min(dist_vector));
                centroid(k,:) = cluster_points(min_index(1),:);
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
    %distance = sum(abs(point_array - k_array));
    
    k_index = find(distance == min(distance));
    k_index = k_index(1);
end

