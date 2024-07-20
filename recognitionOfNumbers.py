import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical, plot_model

import matplotlib.pyplot as plt
import numpy as np
import random

import warnings
from warnings import filterwarnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=FutureWarning)
warnings.filterwarnings("ignore",category=UserWarning)
filterwarnings('ignore')


# mnist veri setinin yüklenmesi
# x -> pixelleri gösterir
# y -> outputlar/target/label
(x_train,y_train),(x_test,y_test) = mnist.load_data()
print("Eğitim seti boyutu:", x_train.shape,y_train.shape) #60000 tane görsel var 28x28 pixellik
print("Test seti boyutu:", x_test.shape, y_test.shape) #10000 tane görsel var

num_labels = len(np.unique(y_train)) # 0,1,2,3,4,5,6,7,8,9 -> 10 adet
print(num_labels)

# veri setinden örnekler gösterilmesi
plt.figure(figsize=(10,10))
plt.imshow(x_train[0], cmap='gray') #train setindeki ilk gözlem gelir
#plt.show()
plt.imshow(x_train[59000], cmap='pink') # 59000. gözlem gelir
#plt.show()

def visualize_img(data):
    plt.figure(figsize=(10,10))
    for n in range(10):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(x_train[n], cmap='gray')
        plt.axis('off')
    plt.show()

#visualize_img(x_train)

# RGB(0-255) kodları kullanılır
# 255,255,255 = beyaz
# 0,0,0 = siyah

print(x_train[2])
print(x_train[2].shape)
print(x_train[2][10,10])
print(x_train[2].mean())
print(x_train[2].sum())
print(x_train[2][14:20,10:20])
print(x_train[2][14:20,10:20].mean())



def pixel_visualize(img):
    fig = plt.figure(figsize=(12,12))
    ax= fig.add_subplot(111)
    ax.imshow(img,cmap='gray')
    width, height = img.shape
    
    treshold = img.max() / 2.5
    
    for x in range(width):
        for y in range(height):
            ax. annotate(str(round(img[x][y],2)), xy=(y,x),
                    color='white' if img[x][y]<treshold else 'black')
    plt.show()
            
#pixel_visualize(x_train[2])


# ENCODING : bağımlı değişkene yani outputa/target/sonuuca/labela uygularız
print(y_train[0:5]) # 0 dan 5 e kadar olan görsellerde hangi sayılar olduğunu verdi

y_train = to_categorical(y_train) # 1 ve 0 larla ifade edilen şekle döndürüldü 
y_test = to_categorical(y_test)

print(y_train)
print('*************')
print(y_test)

# RESHAPING : 28x28 lik train set için 60000 görsel var. Her pixelde bulunan 0-255 arasında aldığı değeri elde etmek için reshape yapılır.
image_size = x_train.shape[1]
print(image_size)
print(f"x_train boyutu: {x_train.shape}")
print(f"x_test boyutu: {x_test.shape}")

x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)
print(f"x_train boyutu: {x_train.shape}")
print(f"x_test boyutu: {x_test.shape}")

# STANDARDIZATION : renk değerleri 0-255 arasında bunu standartlaştırıp 0-1 arasında olmasını sağlıycaz
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# MODELING(MODELLEME) : 
model = tf.keras.Sequential([
    Flatten(input_shape = (28,28,1)),
    Dense(units=128, activation='relu', name='layer1'),
    Dense(units=num_labels, activation='softmax',name='output_layer')])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),"accuracy"])

model.summary()
#model.fit(x_train,y_train, epochs=5, batch_size=128,validation_data=(x_test,y_test))
        #iterationlar(epoch) arttıkça başarı oranı arttı
        # val_score ve train_score artıyor, loss düşer


# MODEL BAŞARISINI DEĞERLENDİRME(EVALUATION) :
history = model.fit(x_train,y_train, epochs=5, batch_size=128,validation_data=(x_test,y_test))

# Accuracy graph 
plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], color='b', label='Training Accuracy')
plt.plot(history.history['val_accuracy'],color='r', label='Validation Accuracy')
plt.legend(loc='lower right')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.ylim([min(plt.ylim()),1])
plt.title('Eğitim ve test başarım grafiği', fontsize=16)

# Loss graph
plt.subplot(1,2,2)
plt.plot(history.history['loss'], color='b', label='Training Loss')
plt.plot(history.history['val_loss'],color='r', label='Validation Loss')
plt.legend(loc='upper right')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.ylim(0,max(plt.ylim()))
plt.title('Eğitim ve test kayıp grafiği', fontsize=16)
plt.show()

loss,precision, recall, acc = model.evaluate(x_test,y_test, verbose=False)
print("\nTest Accuracy: %.1f%%" %(100.0 * acc))
print("\nTest Loss: %.1f%%" %(100.0 * loss))
print("\nTest Precision: %.1f%%" %(100.0 * precision)) #tahmin ettiklerimizin başarısı
print("\nTest Recall: %.1f%%" %(100.0 * recall)) #doğru tahmin ettiklerim



# Modelin kaydedilmesi ve tahmin için kullanılması
model.save('mnist_model.h5')
random = random.randint(0,x_test.shape[0])
print(random)

test_image = x_test[random]
print(y_test[random])
plt.imshow(test_image.reshape(28,28), cmap='gray')
plt.show()

test_data = x_test[random].reshape(1,28,28,1)
probability = model.predict(test_data)
print(probability)

predicted_classes = np.argmax(probability)
print(predicted_classes)

print(f"Tahmin edilen sınıf: {predicted_classes} \n")
print(f"Tahmin edilen sınıfın olasılık değeri: {(np.max(probability, axis=-1))[0]} \n")
print(f"Diğer sınıfların olasılık değerleri: \n {probability}")










