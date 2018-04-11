clear all;
close all;
clc;

test_data_name = '../../data/test/test_data_cranial.mat';

resultPath = '../../result/';
result_name = [resultPath, 'result_matrix_inversion.mat'];

disp('Loading test data...');
load(test_data_name);

%%
mu_ = mu_bone_high*mu_tissue_low - mu_tissue_high*mu_bone_low;
yita_11 = mu_tissue_low / mu_;
yita_12 = -mu_tissue_high / mu_;
yita_21 = -mu_bone_low / mu_;
yita_22 = mu_bone_high / mu_;

%%
[h, w, slice] = size(I_L);
I_bone = zeros(h, w, 'single');
I_tissue = zeros(h, w, 'single');

for i = 1:slice
    I_bone(:,:,i) = yita_11*I_H(:,:,i) + yita_12*I_L(:,:,i);
    I_tissue(:,:,i) = yita_21*I_H(:,:,i) + yita_22*I_L(:,:,i);
end

I_bone(I_bone < 0.00001) = 0;
I_tissue(I_tissue < 0.00001) = 0;

figure(1), imshow(I_bone, []);
figure(2), imshow(I_tissue, []);

disp(['Saving result data... ', result_name]);
save(result_name, 'I_bone', 'I_tissue');


