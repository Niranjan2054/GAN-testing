#!/usr/bin/env python3
from gan import GAN
from generator import Generator
from discriminator import Discriminator
from keras.datasets import mnist
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

class Trainer: 
    def __init__(self,width=28,height=28,channels=1,latent_size=100,epochs=50000,batch=32,checkpoint=50,model_type="DCGAN",data_path=''): 
        self.W =width
        self.H = height 
        self.C = channels
        self.EPOCHS = epochs
        self.BATCH = batch 
        self.CHECKPOINT = checkpoint 
        self.model_type = model_type 
        self.LATENT_SPACE_SIZE = latent_size 

        self.generator = Generator(width=self.W,height=self.H,channels=self.C,latent_size=self.LATENT_SPACE_SIZE,model_type=self.model_type)

        self.discriminator = Discriminator(width=self.W,height=self.H,channels=self.C,model_type=self.model_type)
        self.gan_model = GAN(self.discriminator.Discriminator,self.generator.Generator)
        
        if self.model_type=="simple":
            self.load_MNIST()
        elif self.model_type=="DCGAN":
            self.load_npy(data_path)

    def load_npy(self,npy_path,amount_of_data=0.25):
        self.X_train = np.load(npy_path)
        self.X_train = self.X_train[:int(amount_of_data*float(len(self.X_train)))]
        self.X_train = (self.X_train.astype(np.float32)-127.5)/127.5
        self.X_train = np.expand_dims(self.X_train,axis=3)
        return

    def load_MNIST(self,model_type=3):
        allowed_types = [*range(-1,10)]
        if self.model_type not in allowed_types:
            print("Error: Only Integer Values from -1 to 9 are allowed")
        
        (self.X_train,self.Y_train),(_,_) = mnist.load_data()

        if self.model_type !=-1: 
            self.X_train = self.X_train[np.where(self.Y_train==int(self.model_type))[0]]
        
        vector = np.vectorize(np.float)
        self.X_train = (vector(self.X_train) - 127.5)/127.5
        self.X_train = np.expand_dims(self.X_train,axis=3)
        return 
    
    def train(self):
        for e in range(self.EPOCHS):
            #load real images 
            count_real_images = int(self.BATCH/2) 
            starting_index = randint(0,len(self.X_train)-count_real_images)

            real_images_raw = self.X_train[starting_index:starting_index+count_real_images]
            x_real_images = real_images_raw.reshape(count_real_images,self.W,self.H,self.C)
            y_real_labels = np.ones([count_real_images,1])

            #load generated Images 
            latent_space_samples = self.sample_latent_space(count_real_images)
            x_generated_images = self.generator.Generator.predict(latent_space_samples) 
            y_generated_labels = np.zeros([count_real_images,1])
            
            #combine to train on the discriminator 
            x_batch = np.concatenate([x_real_images,x_generated_images])
            y_batch = np.concatenate([y_real_labels,y_generated_labels])

            #Train the discriminator 
            discriminator_loss = self.discriminator.Discriminator.train_on_batch(x_batch,y_batch)[0]

            #Generate Noise 
            x_latent_space_samples = self.sample_latent_space(self.BATCH)
            y_generated_labels = np.ones([self.BATCH,1 ])
            
            generated_loss = self.gan.gan_model.train_on_batch(x_latent_space_samples,y_generated_labels)

            print('Epoch: '+str(e)+', [Discriminator :: Loss : '+str(discriminator_loss)+' ], [Generator :: Loss : '+str(generated_loss)+' ]')
            if  e % self.CHECKPOINT ==0 : 
                self.plot_checkpoint(e)
    

    def dc_train(self):
        generated_loss=0
        for e in range(self.EPOCHS):
            b = 0 
            X_train_temp = deepcopy(self.X_train)
            while len(X_train_temp)>self.BATCH:
                b=b+1 
                # prepare the training data for discriminator
                if self.flipCoin():
                    count_real_images = int(self.BATCH)
                    starting_index = randint(0,len(X_train_temp)-count_real_images)
                    real_images_raw = X_train_temp[starting_index:starting_index+count_real_images]

                    #Delete the images used until we have left none 
                    X_train_temp = np.delete(X_train_temp,range(starting_index,starting_index+count_real_images),0)
                    x_batch = real_images_raw.reshape(count_real_images,self.W,self.H,self.C)

                    y_batch = np.ones([count_real_images,1])
                else: 
                    latent_space_samples = self.sample_latent_space(self.BATCH)
                    x_batch = self.generator.Generator.predict(latent_space_samples)
                    y_batch = np.zeros([self.BATCH,1])
                
                # Train Discriminator with this batch 
                discriminator_loss = self.discriminator.Discriminator.train_on_batch(x_batch,y_batch)[0]

                if self.flipCoin(chance=0.9):
                    y_generated_labels = np.ones([self.BATCH,1])
                else:
                    y_generated_labels = np.zeros([self.BATCH,1])
                    x_latent_space_samples = self.sample_latent_space(self.BATCH)
                    generated_loss = self.gan_model.gan_model.train_on_batch(x_latent_space_samples,y_generated_labels)
                print('Batch: '+str(b)+', [Discriminator :: Loss : '+str(discriminator_loss)+' ], [Generator :: Loss : '+str(generated_loss)+' ]')
                if  b % self.CHECKPOINT == 0:
                    label = str(e)+"_"+str(b)
                    self.plot_checkpoint(label)
            
            print('Epoch: '+str(e)+', [Discriminator :: Loss : '+str(discriminator_loss)+' ], [Generator :: Loss : '+str(generated_loss)+' ]')
            if  e % self.CHECKPOINT ==0 : 
                self.plot_checkpoint(e)

    def sample_latent_space(self,instances):
        return np.random.normal(0,1,(instances,self.LATENT_SPACE_SIZE))

    def plot_checkpoint(self,e):
        filename = "data/sample_"+str(e)+".png"
        
        noise = self.sample_latent_space(16) 
        images = self.generator.Generator.predict(noise)
        plt.figure(figsize = (10,10))
        for i in range(images.shape[0]):
            plt.subplot(4,4,i+1)
            image = images[i,:,:,:]
            image = np.reshape(image,[self.H,self.W,self.C])
            plt.imshow(image,cmap='gray')
            plt.axis('off')
        plt.tight_layout() 
        plt.savefig(filename)
        plt.close('all')
    
    def flipCoin(self,chance=0.5):
        return np.random.binomial(1,chance)