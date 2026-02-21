
import os
import numpy as np
import cv2
import pytesseract
from PIL import Image, ImageEnhance, ImageOps


pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

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