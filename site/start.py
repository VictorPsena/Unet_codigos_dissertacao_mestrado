import os
import sys
import io
import base64
from pathlib import Path
import numpy as np
from PIL import Image

import tensorflow as tf
from flask import Flask
from flask import render_template
from flask import request
from keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODELS_DIR = PROJECT_ROOT / 'modelos'
TMP_UPLOAD_DIR = BASE_DIR / 'uploads_tmp'
TMP_UPLOAD_DIR.mkdir(exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from funcs import banco
from redes.fuzzy_layer import get_fuzzy_custom_objects



App = Flask(__name__, template_folder='templates')

imagens_em_memoria = []
modelos_cache = {}
FUZZY_CUSTOM_OBJECTS = get_fuzzy_custom_objects()

""" setting DICE coefficient to evaluate the model """
def dice_coefficient(y_true, y_pred, smooth=1.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Flatten per-sample to compute Dice over all pixels
    y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    union = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1)
    dice_coeff = (2.0 * intersection + smooth) / (union + smooth)
    return dice_coeff

def listar_modelos_keras():
    if not MODELS_DIR.exists():
        return []
    return sorted([f.name for f in MODELS_DIR.iterdir() if f.suffix == '.keras'])


def carregar_modelo(nome_modelo):
    if nome_modelo in modelos_cache:
        return modelos_cache[nome_modelo]

    caminho_modelo = MODELS_DIR / nome_modelo
    custom_objects = {
        'dice_coefficient': dice_coefficient,
        **FUZZY_CUSTOM_OBJECTS,
    }
    modelo = load_model(caminho_modelo, custom_objects=custom_objects)
    modelos_cache[nome_modelo] = modelo
    return modelo


def limpar_uploads_temporarios():
    for arquivo in TMP_UPLOAD_DIR.iterdir():
        if arquivo.is_file():
            arquivo.unlink()


def inferir_shape_entrada(modelo):
    input_shape = modelo.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if len(input_shape) != 4:
        raise ValueError(f'Formato de entrada não suportado: {input_shape}')

    _, altura, largura, canais = input_shape
    altura = int(altura) if altura is not None else 200
    largura = int(largura) if largura is not None else 200
    canais = int(canais) if canais is not None else 3

    return altura, largura, canais


def ajustar_canais(x, canais_esperados):
    if x.ndim != 4:
        raise ValueError(f'Entrada esperada com 4 dimensões, recebido: {x.shape}')

    canais_atuais = x.shape[-1]
    if canais_atuais == canais_esperados:
        return x

    if canais_atuais == 1 and canais_esperados == 3:
        return np.repeat(x, 3, axis=-1)

    if canais_atuais == 3 and canais_esperados == 1:
        return np.mean(x, axis=-1, keepdims=True)

    raise ValueError(
        f'Não foi possível ajustar canais: atuais={canais_atuais}, esperados={canais_esperados}'
    )

############################## 
# Mensagem para exibir o resultado da predição de forma resumida 
##############################
def resumir_predicao(predicao):
    p = np.asarray(predicao).squeeze()

    if p.ndim == 0:
        valor = float(np.clip(p, 0.0, 1.0))
        imagem_array = np.full((128, 128), valor, dtype=np.float32)
    elif p.ndim == 1:
        if p.size == 1:
            valor = float(np.clip(p[0], 0.0, 1.0))
            imagem_array = np.full((128, 128), valor, dtype=np.float32)
        else:
            valores = p.astype(np.float32)
            vmin = float(np.min(valores))
            vmax = float(np.max(valores))
            escala = (vmax - vmin) if (vmax - vmin) > 1e-8 else 1.0
            valores = (valores - vmin) / escala

            altura = 120
            largura = max(160, valores.size * 28)
            imagem_array = np.zeros((altura, largura), dtype=np.float32)
            barra = max(8, largura // valores.size)

            for i, v in enumerate(valores):
                x0 = i * barra
                x1 = min(largura, (i + 1) * barra - 2)
                y0 = int(altura - (v * (altura - 1)))
                imagem_array[y0:altura, x0:x1] = 1.0
    else:
        if p.ndim == 3:
            if p.shape[-1] == 1:
                imagem_array = p[..., 0]
            else:
                imagem_array = np.mean(p, axis=-1)
        else:
            imagem_array = p

        imagem_array = imagem_array.astype(np.float32)
        vmin = float(np.min(imagem_array))
        vmax = float(np.max(imagem_array))
        escala = (vmax - vmin) if (vmax - vmin) > 1e-8 else 1.0
        imagem_array = (imagem_array - vmin) / escala

    imagem_uint8 = np.clip(imagem_array * 255.0, 0, 255).astype(np.uint8)
    imagem_pil = Image.fromarray(imagem_uint8, mode='L')
    imagem_pil = imagem_pil.resize((256, 256), Image.NEAREST)

    buffer = io.BytesIO()
    imagem_pil.save(buffer, format='PNG')
    imagem_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{imagem_base64}'
    

@App.route('/start/<name>', methods=['GET', 'POST'])
@App.route('/start/', methods=['GET', 'POST'])
def start(name=None):
    global imagens_em_memoria

    modelos_disponiveis = listar_modelos_keras()
    resultados = []
    mensagem = None
    modelo_selecionado = request.form.get('modelo', '')

    if request.method == 'POST':
        arquivos = request.files.getlist('imagens')
        arquivos_validos = [a for a in arquivos if a and a.filename]

        if not modelo_selecionado:
            mensagem = 'Selecione um modelo .keras.'
        elif modelo_selecionado not in modelos_disponiveis:
            mensagem = 'Modelo selecionado não foi encontrado.'
        elif not arquivos_validos:
            mensagem = 'Selecione ao menos uma imagem para processar.'
        else:
            try:
                modelo = carregar_modelo(modelo_selecionado)
                altura, largura, canais = inferir_shape_entrada(modelo)
                grayscale = canais == 1

                limpar_uploads_temporarios()
                imagens_em_memoria = []
                nomes_salvos = []

                for indice, arquivo in enumerate(arquivos_validos):
                    conteudo = arquivo.read()
                    if not conteudo:
                        continue

                    imagens_em_memoria.append(
                        {
                            'nome': arquivo.filename,
                            'bytes': conteudo,
                        }
                    )

                    nome_salvo = f'{indice:04d}_{Path(arquivo.filename).name}'
                    caminho_saida = TMP_UPLOAD_DIR / nome_salvo
                    with open(caminho_saida, 'wb') as f:
                        f.write(conteudo)
                    nomes_salvos.append(nome_salvo)

                if not nomes_salvos:
                    mensagem = 'Não foi possível ler os arquivos enviados.'
                else:
                    x = banco(
                        str(TMP_UPLOAD_DIR),
                        resolution=(largura, altura),
                        grayscale=grayscale,
                        keep_channel_dim=True,
                        file_list=nomes_salvos,
                    )
                    x = ajustar_canais(x, canais)

                    predicoes = modelo.predict(x, verbose=0)
                    for i, item in enumerate(imagens_em_memoria):
                        resultados.append(
                            {
                                'arquivo': item['nome'],
                                'imagem_predicao': resumir_predicao(predicoes[i]),
                            }
                        )

                    mensagem = (
                        f'{len(resultados)} imagem(ns) processada(s) com o modelo "{modelo_selecionado}".'
                    )
            except Exception as e:
                mensagem = f'Erro no processamento/predição: {e}'

    return render_template(
        'main.html',
        name=name,
        mensagem=mensagem,
        resultados=resultados,
        modelos=modelos_disponiveis,
        modelo_selecionado=modelo_selecionado,
    )
