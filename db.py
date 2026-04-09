import funcs
import os
from PIL import Image
""" Brilho na imagem"""
# pasta_img = 'banco1/imgAOL'
# pasta_destino ='banco1/imgBRILHO'
# os.makedirs(pasta_destino, exist_ok=True)
# for img in os.listdir(pasta_img):
#     caminho = os.path.join(pasta_img, img)
#     img = Image.open(caminho)
#     img_brilho = funcs.brilho(img, fator= 2)
#     nome_arquivo = os.path.basename(caminho)
#     caminho_saida = os.path.join(pasta_destino, nome_arquivo)
#     img_brilho.save(caminho_saida)



""" Seleciona e Renomeia """
pasta_imagens ='banco1/imgGERAIS/24_11_21_NelorePGP-manha'
pasta_alvo = 'banco1/imgAOL/24_11_21_NelorePGP-manha'
# funcs.altera_nome(pasta_imagens, pasta_alvo, extensao=".BMP", encontrar='_')

"""Corte na imagem"""
pasta_alvo ='banco2/Nelore2/EGG/imgs1'
pasta_saida ='banco2/Nelore2/EGG/imgs'
funcs.corte(pasta_alvo, pasta_saida, corte = [0.15, 0.06, 0.67, 0.90])

""" Aplica transformações nas imagens """
pasta_saida1 = 'banco1/imgAUGMENTATION/24_11_21_NelorePGP-manha'
# funcs.dataAugmentation(pasta_saida, pasta_saida1, num_augmented_images=5)
# funcs.modificacoes(pasta_saida, pasta_saida1, flip=True, escura=0.5)

""" Banco de teste """
teste1_img = 'banco1/imgGERAIS/04_11_20_NeloreFinalPG'
teste2_img = 'banco1/teste/imgAOL'
# funcs.altera_nome(teste1_img,teste2_img,extensao=".BMP", encontrar='_')
teste3_img = 'banco1/teste/imgCORTE'
# funcs.corte(teste2_img, teste3_img, corte = [0.18, 0.1, 0.94, 0.54])


""" Banco de teste EGG """
testeegg = 'testeEGG'
testeeggexit = 'testeEGG'
# funcs.corte(testeegg, testeeggexit, corte=[0.18, 0.04, 0.95, 0.20])


""" Novo Banco de Dados """
entry_teste ='banco1/imgGERAIS/'
exit_teste  ='banco1/imgAOL/25_11_21-PGPCaracu'
# funcs.altera_nome(entry_teste, exit_teste, extensao=".BMP", encontrar='_')