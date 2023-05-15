I1 = imread('Original_Image/butterfly_x4_GT.png');%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Crop1 = I1(150:200,200:100,:); %first number is heightLast number is width%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
subplot(211);imshow(I1);
axis on
subplot(212);imshow(Crop1);
axis on 
imwrite(Crop1,'Patches_Created/butterfly_GT_PATCH_x4.png');%SAVE THE MODEL PATCH IMAGE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

I2 = imread('Original_Image/butterfly_x4_bicubic.png'); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Crop2 = I2(150:200,200:100,:); %first number is heightLast number is width%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
subplot(211);imshow(I2);
axis on
subplot(212);imshow(Crop2)
axis on
imwrite(Crop2,'Patches_Created/butterfly_bicubic.png');%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CALCULATE THE PSNR AND SSIM
GT = imread('Patches_Created/butterfly_GT_PATCH_x4.png'); 
ModelImg = imread('Patches_Created/butterfly_bicubic.png');%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
psnr = compute_psnr(GT,ModelImg);
fprintf('PSNR for =   %f dB\n', psnr);
[ssimval, ssimmap] = ssim(GT,ModelImg);  
fprintf('The SSIM value is %0.3f.\n',ssimval);