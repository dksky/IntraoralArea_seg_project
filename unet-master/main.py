from model import *
from data import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import load_model
import time

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train():
    data_gen_args = dict(rotation_range=0.2,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=0.05,
                        zoom_range=0.05,
                        horizontal_flip=True,
                        fill_mode='nearest')
    myGene = trainGenerator(2,'data/intraoralArea/train','image','label',data_gen_args,save_to_dir = None)

    model = unet()
    model_checkpoint = ModelCheckpoint('unet_intraoralArea.hdf5', monitor='loss',verbose=1, save_best_only=True)
    model.fit_generator(myGene,steps_per_epoch=100,epochs=1,callbacks=[model_checkpoint])

    testGene = testGenerator("data/intraoralArea/test", 3)
    results = model.predict_generator(testGene,3,verbose=1)
    saveResult("data/intraoralArea/test",results)

def predict():
    model = load_model("unet_intraoralArea.hdf5")
    testGene = testGenerator("data/intraoralArea/test", 3)
    results = model.predict_generator(testGene, 3, verbose=1)
    saveResult("data/intraoralArea/test", results)

def getPredictValueByImg(img, model):
    img_w = img.shape[0]
    img_h = img.shape[1]
    #model = load_model("unet_intraoralArea.hdf5")
    t1 = time.time()
    testGene = testGeneratorForImg(img)
    t2 = time.time()
    results = model.predict_generator(testGene, 1, verbose=1)
    t3 = time.time()
    print("t2-t1=" + str(t2 - t1) + "t3-t2=" + str(t3 - t2))
    return results[0]

if __name__=='__main__':
    #train()
    predict()