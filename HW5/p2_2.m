clear;

% Importing data
load('USPS.mat');

p_vals = [10, 50, 100, 200];
len = length(p_vals);
error = zeros(len,1);

figure;
for i= 1:len
    
    p = p_vals(i);
    
    % Create data from PCA residuals
    [residuals,reconstructed] = pcares(A, p);
    
    % Error computation
    diff = A - reconstructed;
    error(i) = norm(diff,'fro'); % Frobenius norm to calc error
    
    % Reshape the images
    img1 = reshape(reconstructed(1,:), 16, 16);
    img2 = reshape(reconstructed(2,:), 16, 16);
    
    % Plot images
    subplot(2, len ,len+i);
    imshow(img1');
    
    subplot(2, len, i);
    imshow(img2');
    
    title(sprintf('p = %d \n Error: %f', p, error(i)));
end

% Error Plot
figure;
plot(p_vals, error, 'o-', 'linewidth', 2);
xlabel('PCA components (Values)');
ylabel('Reconstruction Error');