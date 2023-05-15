I = imread('Original_Image/YumeiroCooking_GT.png');
Crop = I(1000:1100,600:700,:); %first number is heightLast number is width
figure;
subplot(211);imshow(I);
axis on
subplot(212);imshow(Crop);
axis on 
imwrite(Crop,'Patches_Created/YumeiroCooking_GT.png');%SAVE THE MODEL PATCH IMAGE%%%%%%%%%
