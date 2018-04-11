close all;
clear all;
clc;

test_data_name = '../../data/test/test_data_cranial.mat';

resultPath = '../../result/';
result_name = [resultPath, 'result_itertive_decomposition.mat'];

disp('Loading test data...');
load(test_data_name);

%%
mu_ = mu_bone_high*mu_tissue_low - mu_tissue_high*mu_bone_low;
a = mu_tissue_low / mu_;
b = -mu_tissue_high / mu_;
c = -mu_bone_low / mu_;
d = mu_bone_high / mu_;

%%
pcgmaxi = 500;
pcgtol = 1e-12;
beta1 = 2e-6; %1;
beta2 = beta1*7;

A = [a b; c d];
A = inv(A);
A = kron(A,speye(512*512));
At = A';

[h, w, slice] = size(I_L);

I_bone = zeros([h, w, slice], 'single');
I_tissue = zeros([h, w, slice], 'single');

img = zeros(h, 2*w);
img_d = zeros(h, 2*w);
for i = 1:slice
    disp(['Decomposing image ', num2str(i), '/', num2str(slice)]);
    img(:, 1:end/2) = I_H(:,:,i);
    img(:, (end/2 + 1):end) = I_L(:,:,i);
    img_d = [a*img(:,1:end/2) + b*img(:,end/2+1:end) c*img(:,1:end/2) + d*img(:,end/2+1:end)];
    x = img_d(:);
    data = A'*img(:);
    ratio = max(data(:));
    data = data / ratio;
    weight = get_weight(img, h, w);
    [x_1, flag, relres, iter, rv] = pcg(@AXfunc_pwls,data,pcgtol,pcgmaxi,[], ...
            [],x,A,At,weight,beta1,beta2);        
    x_1 = reshape(x_1, [h, 2*w])*ratio;
    
    I_bone(:,:,i) = x_1(:, 1:end/2);
    I_tissue(:,:,i) = x_1(:, (end/2 + 1):end);
end

I_bone(I_bone < 0.00001) = 0;
I_tissue(I_tissue < 0.00001) = 0;

figure(1), imshow(I_bone, []);
figure(2), imshow(I_tissue, []);

disp(['Saving result data... ', result_name]);
save(result_name, 'I_bone', 'I_tissue');




