#!/usr/bin/env python3
from train import Trainer
HEIGHT = 64
WIDTH= 64
CHANNEL = 3
LATENT_SPACE_SIZE = 100
EPOCHS = 500000
BATCH = 32
CHECKPOINT = 10
MODEL_TYPE = "DCGAN"
PATH = "/content/GAN-testing/DCGAN/data/facadeRGB.npy"

trainer = Trainer(height=HEIGHT,width=WIDTH,channels=CHANNEL,latent_size=LATENT_SPACE_SIZE,epochs =EPOCHS,batch=BATCH,checkpoint=CHECKPOINT,model_type=MODEL_TYPE,data_path=PATH)
if MODEL_TYPE == 'simple':
    trainer.train()
elif MODEL_TYPE == "DCGAN":
    trainer.dc_train()