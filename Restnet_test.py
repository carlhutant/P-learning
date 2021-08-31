import warnings

warnings.filterwarnings('ignore')

from tensorflow.keras.applications.resnet import ResNet101, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ## Import data
### change the dataset here###
dataset = 'AWA2'
##############################

batch_size = 16
train_dir = './data/{}/IMG/train'.format(dataset)
val_dir = './data/{}/IMG/val'.format(dataset)
IMG_SHAPE = 224
epochs = 15
seen_class_num = 40

# ## Fine tune or Retrain ResNet101
base_model = ResNet101(weights='imagenet', include_top=False)

# # lock the model
# for layer in base_model.layers:
#     layer.trainable = False

# add a global averge pollinf layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# add a dense
x = Dense(1024, activation='relu')(x)

# add a classifier
predictions = Dense(seen_class_num, activation='softmax')(x)

# Constructure
model = Model(inputs=base_model.input, outputs=predictions)

image_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_gen = image_gen.flow_from_directory(
    batch_size=24,
    directory="./data/AWA2/test_draw",
    color_mode="rgb",
    target_size=(224, 224),
    class_mode='sparse',
    shuffle=False,
    seed=42
)
model.load_weights("./model/AWA2/FineTuneResNet101_extend_with_head.h5")
data, label = train_gen.next()
ans = model.predict(data)

print(ans.argmax())
# for i in range(ans.shape[0]):
#     print(ans[i].argmax())
a = 0
