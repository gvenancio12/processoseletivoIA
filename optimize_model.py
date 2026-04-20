import tensorflow as tf
import os

try:
    # 1. Carregar o modelo treinado da Etapa 1
    print("Carregando o modelo original: ")
    if not os.path.exists('model.h5'):
        raise FileNotFoundError("Arquivo 'model.h5' não encontrado.")
    
    model = tf.keras.models.load_model('model.h5')
    print("Modelo carregado com sucesso.")

    # 2. Inicializar o conversor TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # 3. Aplicar otimização: Dynamic Range Quantization
    print("Aplicando otimização: ")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # 4. Executar a conversão
    print("Iniciando conversão: ")
    tflite_model = converter.convert()
    print("Conversão bem-sucedida.")

    # 5. Salvar o modelo final otimizado
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)

    print("Conversão concluída! Modelo otimizado salvo como 'model.tflite'")

except Exception as e:
    print(f"Erro durante a otimização: {e}")
    import traceback
    traceback.print_exc()