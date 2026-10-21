# Estimação de Pose Humana sob Oclusão de Câmera

Projeto acadêmico desenvolvido para a disciplina de **Inteligência Artificial II** do 6º período do curso de Ciência da Computação da Universidade do Estado do Rio Grande do Norte (UERN).

## Sobre o Projeto

Este repositório contém a implementação e a validação de diferentes arquiteturas preditivas clássicas e de *Deep Learning* aplicadas à visão computacional. O objetivo central é solucionar o problema de perda temporal de rastreamento (*tracking*) quando um alvo humano em movimento irregular entra em uma zona de oclusão (ex: atrás de um pilar obscuro). 

A abordagem utiliza o modelo **YOLOv11** para a extração inicial do esqueleto humano (17 keypoints) frame a frame, passando o histórico de trajetória para os algoritmos preditivos assumirem a estimação de pose de forma autorregressiva durante a ausência de dados do sensor visual.

## Algoritmos Implementados e Avaliados

A pesquisa estabelece um comparativo de erro (Erro Quadrático Médio - MSE) em pixels entre os seguintes métodos:

*   **Filtro de Kalman Estendido (EKF):** Modelo cinemático baseado na física do movimento, com linearização via matrizes jacobianas.
*   **Long Short-Term Memory (LSTM):** Rede neural recorrente para captura de dependências de longo prazo em séries temporais.
*   **Gated Recurrent Unit (GRU):** Arquitetura recorrente otimizada para maior eficiência e velocidade computacional.
*   **Temporal Transformers:** Modelo baseado inteiramente no mecanismo de *Self-Attention* global, focado em ciclos longos de movimento (ex: marcha humana).
*   **Temporal Graph Neural Networks (T-GNN):** O estado da arte do projeto. Modela o corpo humano combinando Redes em Grafos (para as restrições físicas espaciais das articulações) e recorrência (para a evolução do movimento no tempo).

## Tecnologias e Dependências

*   **Linguagem:** Python 3
*   **Visão Computacional:** OpenCV, Ultralytics (YOLO)
*   **Deep Learning:** PyTorch
*   **Filtros Bayesianos:** FilterPy
*   **Computação Científica:** NumPy, Scikit-learn, Matplotlib

## Como Executar (Pipeline de Execução)

O projeto foi estruturado em uma esteira de dados (pipeline) que separa a extração, o treinamento e a inferência.

### 1. Instalação
Clone o repositório e instale as bibliotecas necessárias:
```bash
git clone [https://github.com/vinxciusg/IA2-Estimacao-Oclusao.git](https://github.com/vinxciusg/IA2-Estimacao-Oclusao.git)
cd IA2-Estimacao-Oclusao
pip install ultralytics torch torchvision opencv-python matplotlib numpy scikit-learn filterpy
```

### 2. Extração de Poses
O YOLO processará o vídeo e extrairá o Ground Truth (coordenadas do esqueleto) para gerar o arquivo poses.npy:
```bash
python extrair_poses.py
```

### 3. Treinamento das Redes
Os modelos recorrentes e de atenção aprenderão os padrões de movimento do vídeo antes da oclusão:
```bash
python treinar_avancado.py
```

### 4. Estimação e Oclusão Simulada
Aplica o "Pilar Obscuro" no vídeo, executa a inferência autorregressiva às cegas, gera o vídeo final e plota o gráfico comparativo de erros:
```bash
python estimacao_avancada.py
```

# Equipe Desenvolvedora
- Vinicius Samuel Gomes Miranda
- Estela Viana Santos
- Pâmela Wallesca Fernandes Alves
