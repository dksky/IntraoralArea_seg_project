## Train
对应文件：train.py

样本集：设置在data/intraoralArea/train目录下，image文件夹保存样本初始图片，label文件夹保存样本分割后图片。

运行方式：运行train.py即可进行样本训练。训练完成后，模型会保存在intraoralArea.hdf5文件中。

相关参数：
* 图片尺寸：当前训练设定的输入图片尺寸为64*64
* 模型类别：当前训练设定的模型类别为unet（可输入的模型类别列表可查看models.all_models.py文件）。
* 训练epoch：当前训练epoch为1000，可酌情修改。

同时默认会对data/intraoralArea/test文件夹中的测试图片进行导入分割，结果同样保存在data/intraoralArea/test文件夹。

## Test
相关文件：predict.py

默认读取的模型文件：intraoralArea.hdf5

**进行测试时，需保证predict.py中的参数设定与训练时参数设定一致，否则会导致分割失败。**

### 对测试图片进行分割测试
对应文件：predict.py

测试样本集：设置在data/intraoralArea/test目录下。

运行方式：运行predict.py即可对测试图片进行分割测试，分割结果自动保存在data/intraoralArea/test目录下，并以后缀“_predict”命名。

### 读取摄像头实时分割测试
对应文件：intraoral_area_division.py

运行方式：运行intraoral_area_division.py即可。
运行intraoral_area_division.py，会自动打开摄像头，加载intraoralArea.hdf5模型，实时进行图像分割并显示分割结果。输出栏会输出当前FPS信息。

本地测试，在输入图片尺寸为64*64，模型为unet的情况下，fps可以达到20左右(单张耗时：50ms)。

## TODO
1. 目前训练集使用的是牙齿分割的样本数据。需要修改为口内区域分割样本。
2. 目前已能够使用不同网络结构进行训练（FCN-32s, FCN-16s，FCN-8s，VGGNet，SegNet等）。
    但还需要比较不同网络结构的差异，以及输出结果的差异，以选择最适合的网络结构。
3. 在缩小输入网络的图片像素，牺牲图片精准度的情况下，目前已经基本能够达到25fps的性能标准。
    但目前仍是整张图片进行分割，需要结合人脸识别模型，实现只针对人脸区域进行实时分割。
    
## Reference
1. https://divamgupta.com/image-segmentation/2019/06/06/deep-learning-semantic-segmentation-keras.html
2. https://github.com/divamgupta/image-segmentation-keras
3. https://blog.csdn.net/g11d111/article/details/78068413
4. https://github.com/zhixuhao/unet
