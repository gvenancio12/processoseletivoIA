import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

#insira seu código aqui
# 1. Carregamento do dataset via Keras
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Normalização dos pixels
# As imagens originais possuem valores de pixel que vão de 0 a 255. 
# Dividimos por 255.0 para convertê-los para o intervalo entre 0 e 1, o que facilita o treinamento da rede.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 3. Ajuste de formato (Reshape) para as camadas Conv2D
# As CNNs esperam que as imagens tenham o formato (altura, largura, canais de cor).
# O MNIST tem imagens de 28x28 pixels em tons de cinza. Precisamos adicionar explicitamente o canal de cor "1".
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 4. Construção do Modelo (CNN)
print("Construindo a arquitetura do modelo...")
model = keras.Sequential([
    # Primeira camada convolucional e de pooling
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    # Segunda camada convolucional e de pooling
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Camada de achatamento (transforma a matriz 2D em um vetor 1D)
    layers.Flatten(),
    
    # Camadas Densas (Fully Connected) para classificação
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax') # 10 neurônios de saída (dígitos de 0 a 9)
])

# 5. Compilação do Modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 6. Treinamento do Modelo
print("Iniciando o treinamento...")
# epochs=5 significa que o modelo vai "ver" o dataset inteiro 5 vezes
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# 7. Avaliação e Exibição da Acurácia Final
print("\nAvaliando o modelo com os dados de teste...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"*** Acurácia final no conjunto de teste: {test_acc * 100:.2f}% ***\n")

# 8. Salvamento do Modelo em formato .h5
model.save('model.h5')
print("Modelo salvo com sucesso como 'model.h5'!")