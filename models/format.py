import keras

model = keras.models.load_model("APP-host\models\imageclassifier.h5")
model.save("imageclassifier.keras")