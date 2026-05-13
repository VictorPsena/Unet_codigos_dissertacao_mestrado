
import os
import numpy as np
import cv2
import pytesseract
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

from PIL import Image, ImageEnhance, ImageOps



pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
##################################################################################################################
def _adicionar_manchas_pretas(
    img_u8,
    qtd=(1, 4),
    raio_rel=(0.03, 0.10),
    intensidade=(0.45, 0.95),
    blur_ksize=11,
    y_range=None,
    x_range=None,
    manter_dentro=True,
):
    h, w = img_u8.shape[:2]
    menor = min(h, w)

    y0, y1 = (0, h - 1) if y_range is None else (
        max(0, int(y_range[0])),
        min(h - 1, int(y_range[1]))
    )
    x0, x1 = (0, w - 1) if x_range is None else (
        max(0, int(x_range[0])),
        min(w - 1, int(x_range[1]))
    )

    if y0 > y1 or x0 > x1:
        raise ValueError("Faixa x_range/y_range inválida")

    mask = np.zeros((h, w), dtype=np.float32)
    n = np.random.randint(qtd[0], qtd[1] + 1)

    for _ in range(n):
        r = int(np.random.uniform(raio_rel[0], raio_rel[1]) * menor)
        r = max(2, r)

        rx = max(2, int(r * np.random.uniform(0.7, 1.4)))
        ry = max(2, int(r * np.random.uniform(0.7, 1.4)))
        ang = float(np.random.uniform(0, 180))

        if manter_dentro:
            cx_min = max(x0 + rx, 0)
            cx_max = min(x1 - rx, w - 1)
            cy_min = max(y0 + ry, 0)
            cy_max = min(y1 - ry, h - 1)
        else:
            cx_min, cx_max = x0, x1
            cy_min, cy_max = y0, y1

        if cx_min > cx_max or cy_min > cy_max:
            continue

        cx = np.random.randint(cx_min, cx_max + 1)
        cy = np.random.randint(cy_min, cy_max + 1)

        cv2.ellipse(mask, (cx, cy), (rx, ry), ang, 0, 360, 1.0, -1)

    if blur_ksize is not None and blur_ksize > 1:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        mask = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)

    alpha = float(np.random.uniform(intensidade[0], intensidade[1]))
    out = img_u8.astype(np.float32) * (1.0 - alpha * np.clip(mask, 0.0, 1.0))
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out
##################################################################################################################
def banco(path, resolution=(200, 200), grayscale=False, keep_channel_dim=True, nomes=False, file_list=None):
    lista = []
    nomes_list = []

    # Importante: os.listdir não garante ordem; ordenar evita x/y desalinharem.
    files = list(file_list) if file_list is not None else sorted(os.listdir(path=path))

    for file in files:
        caminho = os.path.join(path, file)
        img = Image.open(caminho)
        img = img.resize(resolution)

        if nomes:
            nomes_list.append(file)

        if grayscale:
            img = img.convert('L')

        img_array = np.array(img, dtype=np.float32) / 255.0

        # Para CNNs normalmente é melhor manter (H, W, 1)
        if grayscale and keep_channel_dim:
            img_array = np.expand_dims(img_array, axis=-1)

        lista.append(img_array)

    lista = np.array(lista)

    if nomes:
        return lista, nomes_list
    return lista
##################################################################################################################
def OitentaVinte(x,y, shuffle=True, partition=0.8):
    dados = list(zip(x,y))
    n = len(dados)
    if shuffle:
        np.random.seed(None)
        np.random.shuffle(dados)
    split_index = int(partition * n)
    x_train, y_train = zip(*dados[:split_index])
    x_test, y_test = zip(*dados[split_index:])
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
##################################################################################################################
def corte(entry_data, exit_data, corte = [0.18, 0.03, 0.85, 0.54]):
    os.makedirs(exit_data, exist_ok=True)
    for nome_arquivo in os.listdir(entry_data):
        caminho = os.path.join(entry_data, nome_arquivo)
        img = Image.open(caminho)
        largura, altura = img.size
        esquerda = largura * corte[0]
        superior = altura * corte[1]
        direita = largura * corte[2]
        inferior = altura * corte[3]
        img_cortada = img.crop((esquerda, superior, direita, inferior))
        destino = os.path.join(exit_data, nome_arquivo)
        img_cortada.save(destino)
##################################################################################################################
def altera_nome(entrada_arquivos, saida_arquivos, extensao=".BMP", encontrar='_'):
    os.makedirs(saida_arquivos, exist_ok=True)
    custom_config = r'--psm 6 -c tessedit_char_whitelist=0123456789'
    for nome_arquivo in os.listdir(entrada_arquivos):
        caminho = os.path.join(entrada_arquivos, nome_arquivo)
        img = cv2.imread(caminho)

        # Defina as coordenadas da região (y1:y2, x1:x2)
        y1, y2 = 350, 377   
        x1, x2 = 164, 250
        roi = img[y1:y2, x1:x2]
        #cria um retângulo caso necessário visualizar a área de recorte
        # img_rec = img.copy()
        # cv2.rectangle(img_rec, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.imshow('Imagem com Retângulo', img_rec)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        texto1 = pytesseract.image_to_string(roi, config=custom_config).strip()
        texto2 = pytesseract.image_to_string(img)
        if encontrar in texto2:
            print(f'Selecionado: {nome_arquivo}')
            nome_limpo = "".join(c for c in texto1 if c.isalnum() or c in (' ', '_', '-')).rstrip()
            novo_nome = f"{nome_limpo}{extensao}"
            destino = os.path.join(saida_arquivos, novo_nome)
            cont = 1
            while os.path.exists(destino):
                novo_nome = f"{nome_limpo}_{cont}{extensao}"
                destino = os.path.join(saida_arquivos, novo_nome)
                cont += 1
            cv2.imwrite(destino, img)
            print(f'Renomeado para: {novo_nome}')
##################################################################################################################
def modificacoes(
    entrada,
    saida,
    original=True,
    brilho=None,
    flip=False,
    escura=None,
    rotacao=None,
):
    os.makedirs(saida, exist_ok= True)
    nomes = os.listdir(entrada)
    for i, nome in enumerate(nomes):
        entry = os.path.join(entrada, nome)
        img = Image.open(entry)
        nome, ext = os.path.splitext(nome)
        img_original = img.copy()

        # original
        if original:
            nome_origi = f'{nome}_origi_{i}{ext}'
            exitt = os.path.join(saida, nome_origi)
            img.save(exitt)

        

        # brilho
        if brilho is not None:
            luz = ImageEnhance.Brightness(img_original)
            img = luz.enhance(brilho)
            new_nome = f'{nome}_luz_{i}{ext}'
            exitt = os.path.join(saida, new_nome)
            img.save(exitt)


        # Redução de brilho
        if escura is not None:
            img_dark = ImageEnhance.Brightness(img_original).enhance(escura)
            new_nome = f'{nome}_dark_{i}{ext}'
            exitt = os.path.join(saida, new_nome)
            img_dark.save(exitt)


        # Rotaciona levemente (graus). Aceita:
        # - rotacao = 5 (fixo)
        # - rotacao = (-5, 5) (aleatório uniforme)
        if rotacao is not None:
            if isinstance(rotacao, (tuple, list)) and len(rotacao) == 2:
                angulo = float(np.random.uniform(rotacao[0], rotacao[1]))
            else:
                angulo = float(rotacao)

            # Deixa o nome de arquivo "seguro" (sem ponto/menos)
            ang_str = f"{angulo:.1f}".replace("-", "m").replace(".", "p")

            try:
                img_rot = img_original.rotate(
                    angulo,
                    resample=Image.BICUBIC,
                    expand=False,
                    fillcolor=0,
                )
            except TypeError:
                # Pillow antigo pode não suportar fillcolor
                img_rot = img_original.rotate(
                    angulo,
                    resample=Image.BICUBIC,
                    expand=False,
                )

            new_nome = f'{nome}_rot_{ang_str}_{i}{ext}'
            exitt = os.path.join(saida, new_nome)
            img_rot.save(exitt)

        
        # Reflete
        if flip:
            img = ImageOps.mirror(img_original)
            nome_flip = f'{nome}_flip_{i}{ext}'
            exitt = os.path.join(saida, nome_flip)
            img.save(exitt)
##################################################################################################################
def tvt(x, y, shuffle=True, partition=0.8, nomes=None):
    """Separa dados em treino/teste ou treino/validação/teste.

    Parâmetros
    - x, y: listas/arrays com o mesmo comprimento.
    - shuffle: embaralha antes de separar.
    - partition:
        - float (ex.: 0.8): fração para treino; restante vai para teste.
          Retorna: x_train, y_train, x_test, y_test (compatível com o código antigo).
        - tuple/list:
            - (train, val): teste vira o restante (1 - train - val)
            - (train, val, test): se não somar 1, o restante é adicionado ao teste.
          Retorna: x_train, y_train, x_val, y_val, x_test, y_test.
    """
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    nomes_arr = None
    if nomes is not None:
        nomes_arr = np.asarray(nomes)
    if len(x_arr) != len(y_arr):
        raise ValueError(f"x e y precisam ter o mesmo tamanho, mas têm {len(x_arr)} e {len(y_arr)}")
    if nomes_arr is not None and len(nomes_arr) != len(x_arr):
        raise ValueError(
            f"nomes precisa ter o mesmo tamanho de x/y, mas tem {len(nomes_arr)} e x tem {len(x_arr)}"
        )

    n = len(x_arr)
    if n == 0:
        if isinstance(partition, (tuple, list, np.ndarray)):
            if nomes_arr is not None:
                return x_arr, y_arr, nomes_arr, x_arr, y_arr, nomes_arr, x_arr, y_arr, nomes_arr
            return x_arr, y_arr, x_arr, y_arr, x_arr, y_arr
        if nomes_arr is not None:
            return x_arr, y_arr, nomes_arr, x_arr, y_arr, nomes_arr
        return x_arr, y_arr, x_arr, y_arr

    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng()
        rng.shuffle(idx)

    if isinstance(partition, (tuple, list, np.ndarray)):
        parts = list(partition)
        if len(parts) == 2:
            train_p, val_p = float(parts[0]), float(parts[1])
            test_p = 1.0 - train_p - val_p
        elif len(parts) == 3:
            train_p, val_p, test_p = float(parts[0]), float(parts[1]), float(parts[2])
            total = train_p + val_p + test_p
            if total < 1.0:
                test_p += (1.0 - total)
        else:
            raise ValueError("partition como tupla/lista deve ter 2 ou 3 valores: (train, val) ou (train, val, test)")

        if train_p < 0 or val_p < 0 or test_p < 0:
            raise ValueError("As frações em partition não podem ser negativas")
        if (train_p + val_p + test_p) > 1.0 + 1e-9:
            raise ValueError("As frações em partition não podem somar mais que 1")

        train_end = int(train_p * n)
        val_end = train_end + int(val_p * n)

        idx_train = idx[:train_end]
        idx_val = idx[train_end:val_end]
        idx_test = idx[val_end:]
        
        if nomes_arr is not None:
            return(
                x_arr[idx_train], y_arr[idx_train], nomes_arr[idx_train],
                x_arr[idx_val], y_arr[idx_val], nomes_arr[idx_val],
                x_arr[idx_test], y_arr[idx_test], nomes_arr[idx_test],
            )
        else:
            return (
                x_arr[idx_train], y_arr[idx_train],
                x_arr[idx_val], y_arr[idx_val],
                x_arr[idx_test], y_arr[idx_test],
                
            )

    split_index = int(float(partition) * n)
    idx_train = idx[:split_index]
    idx_test = idx[split_index:]
    if nomes_arr is not None:
        return (
            x_arr[idx_train], y_arr[idx_train], nomes_arr[idx_train],
            x_arr[idx_test], y_arr[idx_test], nomes_arr[idx_test],
        )
    else:
        return x_arr[idx_train], y_arr[idx_train], x_arr[idx_test], y_arr[idx_test]
##################################################################################################################
def modificacoes1(
    entrada,
    nomes,
    original=True,
    brilho=None,
    escura=None,
    rotacao=None,
    nitidez=None,
    manchas_pretas=None,
):
    dict_img_arr = {}
    for i, img in enumerate(entrada):
        nome, ext = os.path.splitext(nomes[i])
        # original
        if original:
            nome_origi = f'{nome}_origi_{i}{ext}'
            dict_img_arr[nome_origi] = img

        

        # brilho
        if brilho is not None:
            if img.ndim == 3 and img.shape[-1] == 1:
                img = img.squeeze(-1)
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255)
                img = np.clip(img, 0, 255).astype(np.uint8)
            img_luz = Image.fromarray(img, mode = 'L')
            luz = ImageEnhance.Brightness(img_luz)
            img_luz = luz.enhance(brilho)
            img_luz = np.array(img_luz, dtype=np.float32) / 255.0
            new_nome = f'{nome}_luz_{i}{ext}'
            dict_img_arr[new_nome] = img_luz    
            


        # Redução de brilho
        if escura is not None:
            if img.ndim == 3 and img.shape[-1] == 1:
                img = img.squeeze(-1)
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255)
                img = np.clip(img, 0, 255).astype(np.uint8)
            img_escura = Image.fromarray(img, mode = 'L')
            luz = ImageEnhance.Brightness(img_escura).enhance(escura)
            img_escura = np.array(luz, dtype=np.float32) / 255.0
            new_nome = f'{nome}_dark_{i}{ext}'
            dict_img_arr[new_nome] = img_escura

        if rotacao is not None:
            if img.ndim == 3 and img.shape[-1] == 1:
                img = img.squeeze(-1)
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255)
                img = np.clip(img, 0, 255).astype(np.uint8)
            img_rot = Image.fromarray(img, mode='L')
            
            if isinstance(rotacao, (tuple, list)) and len(rotacao) == 2:
                angulo = int(np.random.uniform(rotacao[0], rotacao[1]))
            else:
                angulo = int(rotacao)

            # Deixa o nome de arquivo "seguro" (sem ponto/menos)
            ang_str = f"{angulo}".replace("-", "m").replace(".", "p")

            try:
                img_rot = img_rot.rotate(
                    angulo,
                    resample=Image.BICUBIC,
                    expand=False,
                    fillcolor=0,
                )
                img_rot = np.array(img_rot, dtype=np.float32) / 255.0

            except TypeError:
                # Pillow antigo pode não suportar fillcolor
                img_rot = img_rot.rotate(
                    angulo,
                    resample=Image.BICUBIC,
                    expand=False,
                )
                img_rot = np.array(img_rot, dtype=np.float32) / 255.0

            new_nome = f'{nome}_rot_{ang_str}_{i}{ext}'
            dict_img_arr[new_nome] = img_rot
        
        if nitidez is not None:
            if img.ndim == 3 and img.shape[-1] == 1:
                img = img.squeeze(-1)
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255)
                img = np.clip(img, 0, 255).astype(np.uint8)
            img_nitida = Image.fromarray(img, mode='L')
            if isinstance(nitidez, (tuple, list)) and len(nitidez) == 2:
                nitidez_val = float(np.random.uniform(nitidez[0], nitidez[1]))
            else:
                nitidez_val = float(nitidez)
            enhancer = ImageEnhance.Sharpness(img_nitida)
            img_nitida = enhancer.enhance(nitidez_val) # 1.0 é original, <1.0 é mais borrada, >1.0 é mais nítida
            img_nitida = np.array(img_nitida, dtype=np.float32) / 255.0
            new_nome = f'{nome}_ndz_{i}{ext}'
            dict_img_arr[new_nome] = img_nitida
        
        # Manchas pretas (artefato tipo sombra/dropout local)
        if manchas_pretas is not None:
            if img.ndim == 3 and img.shape[-1] == 1:
                img_base = img.squeeze(-1)
            else:
                img_base = img.copy()

            if img_base.dtype != np.uint8:
                if img_base.max() <= 1.0:
                    img_base = img_base * 255.0
                img_base = np.clip(img_base, 0, 255).astype(np.uint8)

            cfg = {
                "qtd": (1, 4),
                "raio_rel": (0.03, 0.10),
                "intensidade": (0.45, 0.95),
                "blur_ksize": 11,
                "y_range": (0,150),
                "x_range": (0,150),
                "manter_dentro": True,
            }
            if isinstance(manchas_pretas, dict):
                cfg.update(manchas_pretas)

            img_mp = _adicionar_manchas_pretas(
                img_base,
                qtd=cfg["qtd"],
                raio_rel=cfg["raio_rel"],
                intensidade=cfg["intensidade"],
                blur_ksize=cfg["blur_ksize"],
                y_range=cfg["y_range"],
                x_range=cfg["x_range"],
                manter_dentro=cfg["manter_dentro"],
            )

            img_mp = img_mp.astype(np.float32) / 255.0
            new_nome = f"{nome}_mp_{i}{ext}"
            dict_img_arr[new_nome] = img_mp

    
    x_train = list(dict_img_arr.values())
    x_train = [
        img[..., np.newaxis] if img.ndim == 2 else img for img in x_train
    ]   

    return dict_img_arr, x_train
##################################################################################################################
def modificacoes2(dict_dados, nomes_train):
    label = []
    nome = []
    lista_nomes_repetidos = []
    for i in dict_dados.keys():
        cont = 0
        for j in nomes_train:
            if j[:6] == i[:6]:
                if 'rot' in j:
                    parts = j.split('_rot_')
                    if len(parts) > 1:
                        ang_part = parts[1].rsplit('_', 1)[0]  # pega "m15" ou "15"
                        if ang_part.startswith('m'):
                            rotacao = -int(ang_part[1:].replace('p', '.'))
                        else:
                            rotacao = int(ang_part.replace('p', '.'))
                    img = dict_dados[i]
                    if img.ndim == 3 and img.shape[-1] == 1:
                        img = img.squeeze(-1)
                    if img.dtype != np.uint8:
                        if img.max() <= 1.0:
                            img = (img * 255)
                        img = np.clip(img, 0, 255).astype(np.uint8)
                    img_rot = Image.fromarray(img, mode='L')
                    img_rot = img_rot.rotate(
                        rotacao,
                        resample=Image.BICUBIC,
                        expand=False,
                        fillcolor=0,
                    )
                    img_rot = np.array(img_rot, dtype=np.float32) / 255.0
                    label.append(img_rot[..., np.newaxis]
                     if img_rot.ndim == 2 else img_rot)
                    nome.append(i)
                else:
                    label.append(dict_dados[i])
                    nome.append(i)


                
                

            
    return label, nome
##################################################################################################################
def AreaAOL(
    predict,
    orig_size=(403, 333), # dimensão original
    threshold=0.5,
    px_per_cm=24.05,
    save_path='resultados_area_aol.xlsx',
 ):
    orig_h, orig_w = orig_size  # (H, W)
    cm2_per_px2 = 1.0 / (px_per_cm ** 2)
    # Predição (probabilidade)
    y_pred = np.squeeze(predict)
    # Binariza
    y_pred_bin = (y_pred >= threshold).astype(np.uint8)
    # Garante 2D
    if y_pred_bin.ndim == 3:
        y_pred_bin = np.squeeze(y_pred_bin)
    # Resize para dimensão original (OpenCV usa (W,H))
    pred_orig = cv2.resize(y_pred_bin, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST_EXACT)
    # Áreas (px²) e (cm²)
    area_pred_px = int(pred_orig.sum())
    area_pred_cm2 = area_pred_px * cm2_per_px2

    result = {
        'orig_size': (orig_h, orig_w),
        'threshold': float(threshold),
        'px_per_cm': float(px_per_cm),
        'area_pred_cm2': float(f"{area_pred_cm2:.2f}"),
    }

    df = pd.DataFrame([result])

    if save_path:
        return df.to_excel(save_path, index=False)

    return df
##################################################################################################################
def MedidaEGL(
    predict,
    orig_size=(403, 333),
    threshold=0.5,
    px_per_cm=24.05,
    save_path='resultados_medida_egl.xlsx',
 ):
    orig_h, orig_w = orig_size  # (H, W)
    cm_per_px = 1.0 / px_per_cm 

    y_pred = np.squeeze(predict)
    # Binariza
    y_pred_bin = (y_pred >= threshold).astype(np.uint8)
    # Se a máscara veio como 0..255, normaliza implicitamente via threshold
    # Garante 2D
    if y_pred_bin.ndim == 3:
        y_pred_bin = np.squeeze(y_pred_bin)
    # Resize para dimensão original (OpenCV usa (W,H))
    pred_orig = cv2.resize(y_pred_bin, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST_EXACT)
    # Diametro vertical (de cima para baixo) na dimensão original

    def _media_vertical(mask: np.ndarray) -> float:
        lengths = []
        rows, cols = mask.shape

        for col in range(cols):
            current_len = 0
            for row in range(rows):
                if mask[row, col] == 1:
                    current_len += 1
                else:
                    if current_len > 0:
                        lengths.append(current_len)
                    current_len = 0
            if current_len > 0:
                lengths.append(current_len)

        return np.mean(lengths) if lengths else 0.0
  
    diam_pred_px = _media_vertical(pred_orig)
    diam_pred_cm = diam_pred_px * cm_per_px



    

    result = {
        'orig_size': (orig_h, orig_w),
        'threshold': float(threshold),
        'px_per_cm': float(px_per_cm),
        'diam_pred_cm': float(diam_pred_cm),
    }
    
    df = pd.DataFrame([result])

    if save_path:
        return df.to_excel(save_path, index=False)

    return df

##################################################################################################################
