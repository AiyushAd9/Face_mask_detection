from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Model load karo
model = load_model('mask_detector_mobilenetv2.keras')  # ya jo bhi file name hai

# Test data prepare karo (same preprocessing like training)
test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    'dataset/test',  # ya test path jo bhi ho
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary"
)

# Evaluation
loss, accuracy = model.evaluate(test_data)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
