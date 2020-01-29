from model import *
from data import *
from keras.callbacks import ModelCheckpoint

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

n_classes = 2
#设定输入图片的width，height。设定使用的模型类别，设定分类的个数。
model = get_model(width = 64, height = 64, model_name = 'unet', n_classes = n_classes)

def train():
    data_gen_args = dict(rotation_range=0.2,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=0.05,
                        zoom_range=0.05,
                        horizontal_flip=True,
                        fill_mode='nearest')

    img_target_size = (model.input_width, model.input_height)
    mask_target_size = (model.output_width, model.output_height)
    myGene = trainGenerator(2,'data/intraoralArea/train','image','label',data_gen_args, n_classes=n_classes,target_size = img_target_size, mask_target_size = mask_target_size, save_to_dir = None)

    model_checkpoint = ModelCheckpoint('intraoralArea.hdf5', monitor='loss',verbose=1, save_best_only=True)
    model.fit_generator(myGene,steps_per_epoch=1000,epochs=1,callbacks=[model_checkpoint])

if __name__=='__main__':
    train()