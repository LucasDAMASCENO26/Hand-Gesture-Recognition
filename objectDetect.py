import cv2
import numpy as np
import random

# Função para gerar cores aleatórias
def get_random_color():
    return [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

classNames = ['pessoa', 'bicicleta', 'carro', 'motocicleta', 'avião', 'ônibus', 'trem', 'caminhão',
              'barco', 'semáforo', 'hidrante', 'placa de rua', 'placa de pare', 'parquímetro', 'banco',
              'pássaro', 'gato', 'cachorro', 'cavalo', 'ovelha', 'vaca', 'elefante', 'urso', 'zebra',
              'girafa', 'chapéu', 'mochila', 'guarda-chuva', 'sapato', 'óculos', 'bolsa', 'gravata',
              'mala', 'frisbee', 'esquis', 'snowboard', 'bola de esporte', 'pipa', 'taco de beisebol',
              'luva de beisebol', 'skate', 'prancha de surfe', 'raquete de tênis', 'garrafa', 'prato', 
              'taça de vinho', 'copo', 'garfo', 'faca', 'colher', 'tigela', 'banana', 'maçã', 'sanduíche',
              'laranja', 'brócolis', 'cenoura', 'cachorro-quente', 'pizza', 'rosquinha', 'bolo',
              'cadeira', 'sofá', 'planta em vaso', 'cama', 'espelho', 'mesa de jantar',
              'janela', 'escrivaninha', 'vaso sanitário', 'porta', 'tv', 'laptop',
              'mouse', 'controle remoto', 'teclado', 'celular', 'micro-ondas', 'forno',
              'torradeira', 'pia', 'geladeira', 'liquidificador', 'livro', 'relógio',
              'vaso', 'tesoura', 'urso de pelúcia', 'secador de cabelo', 'escova de dentes',
              'escova de cabelo']

# Dicionário para armazenar cores para cada classe
class_colors = {i: get_random_color() for i in range(len(classNames))}

cap = cv2.VideoCapture(0)
thres = 0.5
nms_thres = 0.2

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

if not cap.isOpened():
    print("Erro ao abrir a câmera.")
    exit()

while True:
    ret, video = cap.read()

    if not ret:
        print("Erro ao capturar vídeo.")
        break

    classIds, confis, bbox = net.detect(video, confThreshold=thres)

    if len(classIds) == 0:
        print("Nenhuma classe detectada.")
        continue

    bbox = list(bbox)
    confis = list(np.array(confis).reshape(1, -1)[0])
    confis = list(map(float, confis))

    indices = cv2.dnn.NMSBoxes(bbox, confis, thres, nms_thres)

    if len(indices) > 0:
        for idx in indices:
            if isinstance(idx, np.ndarray):
                idx = idx[0]
            box = bbox[idx]
            x, y, w, h = box[0], box[1], box[2], box[3]
            classId = int(classIds[idx]) - 1  # Convert classId to int and subtract 1
            color = class_colors[classId]  # Obtém a cor para a classe
            label = f"{classNames[classId]}: {confis[idx]*100:.2f}%"  # Adiciona a probabilidade ao label
            cv2.rectangle(video, (x, y), (x + w, y + h), color, 2)
            cv2.putText(video, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    else:
        print("Nenhum índice retornado pelo NMSBoxes.")

    cv2.imshow("objectDetect", video)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
