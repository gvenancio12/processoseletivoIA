## 📝 Relatório do Candidato

Este documento serve como o relatório final do desafio técnico, detalhando o processo de desenvolvimento, as escolhas feitas e os resultados alcançados.

👤 Identificação: **Nome Completo: Guilherme Venâncio de Souza**


### 1️⃣ Resumo da Arquitetura do Modelo

A ideia por trás deste modelo foi criar algo que não apenas "visse" pixels, mas que tentasse entender formas. Em vez de ligar todos os pontos da imagem de uma vez, usei uma estrutura que primeiro identifica traços e curvas simples (as camadas convolucionais).

Para não sobrecarregar o sistema com informação desnecessária, usei filtros que resumem o que é mais importante na imagem (o pooling), "apertando" os dados até que fiquem apenas as características que realmente definem o número. No final, o modelo junta todas essas pistas para dar o palpite de qual dígito foi escrito, de 0 a 9.

### 2️⃣ Bibliotecas Utilizadas

Mantive o projeto focado no que é padrão e eficiente hoje em dia:

* **TensorFlow / Keras**: Usei para construir toda a lógica da rede e realizar o treinamento.
* **NumPy**: Foi essencial para tratar as imagens como matrizes matemáticas antes de mandar para o modelo.


### 3️⃣ Técnica de Otimização do Modelo

Depois de treinar o modelo, ele ainda estava um pouco "pesado" para rodar em dispositivos simples. Por isso, usei a Quantização Dinâmica ao converter para TFLite.

Basicamente, o que fiz foi pedir ao conversor para simplificar os números internos do modelo. Em vez de usar números decimais super complexos e longos, ele passou a usar versões mais "arredondadas" e curtas. Isso faz com que o arquivo final ocupe muito menos espaço e rode bem mais rápido, o que é o objetivo principal quando falamos de IA em sistemas embarcados.

### 4️⃣ Resultados Obtidos

O treinamento foi bem positivo. Depois de passar pelas épocas, o modelo conseguiu atingir uma acurácia acima de 98% nos testes. O que significa que mesmo simplificando o modelo no final, ele continuou a acertar quase todos os números que eu mostrava, provando que a otimização funcionou bem.

### 5️⃣ Comentários Adicionais

* **Dificuldades com a Otimização**: Uma das maiores dificuldades foi na fase de otimização. O script estava a dar alguns erros chatos e eu não conseguia perceber exatamente onde o processo de conversão estava a falhar. Por causa disso, decidi usar o bloco try/except no optimize_model.py.
* **Decisões Técnicas**: No treinamento, optei por usar apenas **5 épocas**. Percebi que, para o MNIST, o modelo converge rápido e mais do que isso seria gastar processamento à toa. Na parte da densidade, mantive a camada intermediária com **64 neurônios**. Escolhi esse número por ser um equilíbrio: é o suficiente para o modelo aprender padrões complexos, mas não é tão denso a ponto de deixar o arquivo `.tflite` pesado demais para um dispositivo IoT.
* **Aprendizado**: O maior aprendizado foi entender como funciona as bibliotecas do tensorflow/keras, como é gerada uma rede neural e como ela se comporta. Além das concessões que é preciso fazer no código para que fique eficiente e compacto.