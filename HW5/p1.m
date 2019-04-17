clear;

% rng(600);


data=randi([0,100], 1000, 2);

num_clusters = 3;

% data = [];
% for i = 1:num_clusters
%     randint = randi([1,1000],1);
%     theta = linspace(0., 2*pi);
%     x = sin(theta)*randint;
%     y = cos(theta)*randint;
%     data = [data;[x' y']];
% end


[clusters, centroid] = kMeans(data, num_clusters);
figure;
hold on;
scatter(data(:,1),data(:,2),[],clusters,'filled')
scatter(centroid(:, 1), centroid(:, 2), 500, 'x', 'linewidth', 3)


% Spectral K-Means
% cite: https://towardsdatascience.com/spectral-clustering-for-beginners-d08b7d25b4d8
N = size(data,1);
adj = squareform(pdist(data));
deg = N .* eye(N);

lap = deg - adj;
[V,~] = eig(lap);

[clusters, centroid] = kMeans(V(:, 1:num_clusters), num_clusters);
figure;
scatter(data(:,1),data(:,2),[],clusters,'filled')


function [clusters, centroid] = kMeans(data, k)

    threshold = 0.00001;
    
    N = size(data,1);  
    
    % Randomly picked points for intial centroid
    perm = randperm(N);
    centroid = data(perm(1:k), :);
    
    % While loop goes here 
    while 1
        clusters = zeros(N, 1);
        % Looping through all the samples
        for idx = 1 : N
            dists = zeros(1, k);

            % Finding which centroid the point belongs to
            for cent = 1:k
                point = centroid(cent, :);
                dists(cent) = sum(abs(data(idx,:) - point).^2);
            end

            [~, i] = min(dists);
            clusters(idx) = i;
        end
        
       % Assign new centroids
       value = zeros(k, size(data, 2));
       count = zeros(k, 1);
       new_centroid = zeros(size(centroid));
       
       for idx = 1:N
           c = clusters(idx);
           value(c, :) = value(c, :) + data(idx, :);
           count(c) = count(c) + 1;
       end

       for cent=1:k
          new_centroid(cent, :) = value(cent, :) / count(cent);
       end
       
       error = mean(abs(centroid - new_centroid));
       if error < threshold
           break
       end
       centroid = new_centroid;
    end
end