%Aplus, Bicubic, DRCN, DRRN, FSRCNN, LapSRN, MSLapSRN, RFL, SCN, SelfExSR, SRCNN, VDSR
I = imread('Original_Image/YumeiroCooking_x3_SRCNN.png'); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Crop = I(1000:1100,600:700,:); %first number is heightLast number is width
figure;
subplot(211);imshow(I);
axis on
subplot(212);imshow(Crop)
axis on
imwrite(Crop,'Patches_Created/YumeiroCooking_x3_SRCNN.png');%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CALCULATE THE PSNR AND SSIM
GT = imread('Patches_Created/YumeiroCooking_GT.png'); 
ModelImg = imread('Patches_Created/YumeiroCooking_x3_SRCNN.png');%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
psnr = compute_psnr(GT,ModelImg);
fprintf('The PSNR value is %0.2f.\n', psnr);
[ssimval, ssimmap] = ssim(GT,ModelImg);  
fprintf('The SSIM value is %0.4f.\n',ssimval);