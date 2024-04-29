import cv2
import pandas as pd
import face_recognition
import os
import time


def carregar_foto_diretorio(dir, metadata_file):
    '''
    entrada:
        dir = diretório das imagens para treinamento
        metadata_file = arquivo onde tem as informações relevantes de cada imagem

    saída:
        imagens = encoding da primeira face da imagem
        labels = lista com nomes da imagem
        metadata = lista com os metadados da imagem (idade, sexo)
    '''
    imagens = list()
    labels = list()
    metadata = list()

    metadata_df = pd.read_excel(metadata_file)
    metadata_dict = metadata_df.set_index('Unique ID').T.to_dict('list')
    for arquivo in os.listdir(dir):
        if arquivo.endswith('.jpg') or arquivo.endswith(
                '.png') or arquivo.endswith('.jpeg'):
            imagem = face_recognition.load_image_file(
                os.path.join(dir, arquivo))
            encoding = face_recognition.face_encodings(
                imagem)[0]   # Primeira face da foto
            imagens.append(encoding)
            labels.append(arquivo.split('.')[0])  # Nome sem o .jpg/.png
            identificador = arquivo.split('.')[0]
            if identificador in metadata_dict:
                metadata.append(metadata_dict[identificador])
            else:
                metadata.append(
                    ['Desconhecido', 'Desconhecido', 'Desconhecido'])

    return imagens, labels, metadata


dir_dados_de_treino = 'Dados de Treinamento'
arquivo_metadados = 'Dados de Treinamento/Dados.xlsx'

'''
    encodes_conhecidos = guarda os valores de encode de faces conhecidos
        são features extraidas do training set
    frame_conhecido_nomes = nome das faces conhecidas
    metadata = informações adicionais relativas aos nomes/imagens (no caso
        são idade e sexo)

'''

encodes_conhecidos, frame_conhecido_nomes, metadata = carregar_foto_diretorio(
    dir_dados_de_treino, arquivo_metadados)


captura_de_camera = cv2.VideoCapture(0)
fps: int = 0
t = time.time()
while True:
    ret, frame = captura_de_camera.read()
    fps += 1
    frame = cv2.flip(frame, 1, 0)
    face = face_recognition.face_locations(frame)
    encodings = face_recognition.face_encodings(frame, face)
    for encode in encodings:
        matches = face_recognition.compare_faces(
            encodes_conhecidos, encode, tolerance=0.4)
        nome = 'Desconhecido'
        nome_completo = 'Desconhecido'
        idade = 'Desconhecido'
        sexo = 'Desconhecido'
        if True in matches:
            indice_match = matches.index(True)
            nome = frame_conhecido_nomes[indice_match]
            nome_completo, idade, sexo = metadata[indice_match]
        #     cv2.putText(
        #         frame, 'Identidade Verificada, acesso permitido', (50, 50),
        #         cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
        # else:
        #     cv2.putText(
        #         frame, 'Desconhecido, acesso negado', (50, 50),
        #         cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

        topo, direita, baixo, esquerda = face_recognition.face_locations(
            frame)[0]
        cv2.rectangle(frame, (esquerda, topo),
                      (direita, baixo), (0, 0, 2555))
        cv2.putText(frame, f'Nome:{nome_completo}',
                    (esquerda, baixo+20), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 1)
        cv2.putText(frame, f'Idade:{idade}',
                    (esquerda, baixo+40), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 1)
        cv2.putText(frame, f'Sexo:{sexo}',
                    (esquerda, baixo+60), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 1)
    cv2.imshow('Webcam Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Masking com 0xFF
        break
tfinal = time.time()
print(fps / (tfinal-t))
print(f'frames = {fps}, tempo = {tfinal-t}')
captura_de_camera.release()
cv2.destroyAllWindows()
