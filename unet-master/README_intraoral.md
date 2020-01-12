## Train
样本集设置在data/intraoralArea/train目录下，image文件夹保存样本初始图片，label文件夹保存样本分割后图片。
使用main.py中的train()来进行样本训练，当前epoch为300，可酌情修改。
训练完成后，模型会保存在unet_intraoralArea.hdf5文件中。
同时默认会对data/intraoralArea/test文件夹中的测试图片进行导入分割，结果同样保存在data/intraoralArea/test文件夹。

## Test
运行intraoral_area_division.py，会自动打开摄像头，加载unet_intraoralArea.hdf5模型，
实时进行图像分割并显示分割结果。输出栏会输出当前FPS信息。

## TODO
1. 目前训练集使用的是牙齿分割的样本数据。需要修改为口内区域分割样本。
2. 当前使用的是unet网络结构（分割单张256*256图片需要1.8s左右，目标耗时：0.04s），
尝试使用FCN-32s，FCN-16s，FCN-8s、VGGNet、SegNet、DeconvNet等网络结构进行训练，比较训练结果。
3. 若#2的几种网络性能都不达标，考虑简化网络结构，
并缩小输入网络的图片像素（即只对人脸区域进行分割，其他区域直接填充为黑色），以减少计算时间，提高FPS。