import tkinter as tk
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog, messagebox
from PIL import Image
import numpy as np

from pathlib import Path
from funcs import comparar_area_predicao_egg

from keras.models import load_model


janela = tk.Tk()
janela.title("AoGl")
janela.geometry("900x800")

intro = tk.Label(janela, text = "Bem-vindo ao AoGl")
intro.pack(pady=20)

#### Seleciona o modelo ####
PASTA_MODELOS = Path("/home/victors/Documents/Codigos_Mestrado/modelos/")

modelos_disponiveis = [f.name for f in PASTA_MODELOS.glob("*.keras")]
modelo_selecionado = tk.StringVar(janela)
modelo_selecionado.set(modelos_disponiveis[0] if modelos_disponiveis else "Nenhum modelo disponível")

dropdown_modelos = tk.OptionMenu(janela, modelo_selecionado, *modelos_disponiveis)
dropdown_modelos.pack(pady=10)
#### ################# ####

#### Seleciona o Arquivo ####
arquivos_selecionados = []
def escolher_arquivos():
    global arquivos_selecionados
    caminhos = filedialog.askopenfilenames(
        title="Selecione 1 ou mais arquivos",
        initialdir=str(PASTA_MODELOS.parent),
        filetypes=[
            ("Imagens", "*.png *.jpg *.jpeg"),
            ("Todos os arquivos", "*.*"),
        ],
    )
    arquivos_selecionados = list(caminhos)
    arquivos_texto.delete("1.0", tk.END)

    for caminho in caminhos:
        arquivos_texto.insert(tk.END, f"Arquivos selecionados: {caminho}\n")




botao_arquivos = tk.Button(janela, text="Escolher Arquivo(s)", command=escolher_arquivos)
botao_arquivos.pack(pady=8)


arquivos_texto = tk.Text(janela,  height=4, width=60)
arquivos_texto.insert(tk.END, "Arquivos:\n")
arquivos_texto.pack(pady=10)


plot_container = tk.Frame(janela)
plot_container.pack(fill="both", expand=True, padx=10, pady=10)

plot_canvas = tk.Canvas(plot_container)
scrollbar_y = tk.Scrollbar(plot_container, orient="vertical", command=plot_canvas.yview)
plot_canvas.configure(yscrollcommand=scrollbar_y.set)

scrollbar_y.pack(side="right", fill="y")
plot_canvas.pack(side="left", fill="both", expand=True)

frame_plot = tk.Frame(plot_canvas)
plot_window = plot_canvas.create_window((0, 0), window=frame_plot, anchor="nw")

status_var = tk.StringVar(value="Selecione modelo e arquivos para processar.")
status_label = tk.Label(janela, textvariable=status_var)
status_label.pack(pady=6)


def _atualizar_scrollregion(event=None):
    plot_canvas.configure(scrollregion=plot_canvas.bbox("all"))


def _ajustar_largura_frame(event):
    plot_canvas.itemconfigure(plot_window, width=event.width)


def _mousewheel(event):
    if event.delta:
        plot_canvas.yview_scroll(int(-event.delta / 120), "units")
    elif event.num == 4:
        plot_canvas.yview_scroll(-1, "units")
    elif event.num == 5:
        plot_canvas.yview_scroll(1, "units")


frame_plot.bind("<Configure>", _atualizar_scrollregion)
plot_canvas.bind("<Configure>", _ajustar_largura_frame)
plot_canvas.bind_all("<MouseWheel>", _mousewheel)
plot_canvas.bind_all("<Button-4>", _mousewheel)
plot_canvas.bind_all("<Button-5>", _mousewheel)


def mostrar_figuras_no_tk(figs):
    for w in frame_plot.winfo_children():
        w.destroy()

    for fig in figs:
        canvas = FigureCanvasTkAgg(fig, master=frame_plot)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x", expand=True, pady=8)

    janela.update_idletasks()
    _atualizar_scrollregion()
    plot_canvas.yview_moveto(0)


def _inferir_shape_entrada(modelo):
    input_shape = modelo.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    _, altura, largura, canais = input_shape
    altura = int(altura) if altura is not None else 200
    largura = int(largura) if largura is not None else 200
    canais = int(canais) if canais is not None else 3
    return altura, largura, canais


def _carregar_imagens(paths, largura, altura, canais):
    imagens = []
    grayscale = canais == 1

    for caminho in paths:
        img = Image.open(caminho)

        esquerda = largura * 0.15
        superior = altura * 0.06
        direita = largura * 0.67
        inferior = altura * 0.90
        img = img.crop((esquerda, superior, direita, inferior))
        
        img = img.resize((largura, altura))


        if grayscale:
            img = img.convert('L')
        else:
            img = img.convert('RGB')

        arr = np.asarray(img, dtype=np.float32) / 255.0
        if grayscale:
            arr = np.expand_dims(arr, axis=-1)

        imagens.append(arr)

    return np.asarray(imagens, dtype=np.float32)


def processar_e_plotar():
    if not modelos_disponiveis:
        messagebox.showerror("Erro", "Nenhum modelo disponível na pasta de modelos.")
        return

    if not arquivos_selecionados:
        messagebox.showwarning("Aviso", "Selecione pelo menos um arquivo.")
        return

    caminho_modelo = PASTA_MODELOS / modelo_selecionado.get()
    if not caminho_modelo.exists():
        messagebox.showerror("Erro", f"Modelo não encontrado: {caminho_modelo}")
        return

    try:
        status_var.set("Carregando modelo...")
        janela.update_idletasks()
        model = load_model(caminho_modelo, compile=False)

        altura, largura, canais = _inferir_shape_entrada(model)
        print(f"Modelo espera entrada com shape (altura={altura}, largura={largura}, canais={canais})")
        x_imgs = _carregar_imagens(arquivos_selecionados, largura, altura, canais)

        nomes_dados = [Path(p).name for p in arquivos_selecionados]

        figs = []
        resultados = []
        for idx in range(len(x_imgs)):
            result, fig1, fig2 = comparar_area_predicao_egg(
                idx=idx,
                model=model,
                x_imgs=x_imgs,
                nomes_dados=nomes_dados,
                plot=True,
            )
            resultados.append(result)
            if fig1 is not None:
                figs.append(fig1)
            if fig2 is not None:
                figs.append(fig2)

        mostrar_figuras_no_tk(figs)
        status_var.set(f"Processado(s) {len(resultados)} arquivo(s).")
    except Exception as e:
        messagebox.showerror("Erro no processamento", str(e))
        status_var.set("Falha no processamento.")


botao_processar = tk.Button(janela, text="Processar e Plotar", command=processar_e_plotar)
botao_processar.pack(pady=8)


#### ################# ####

#### Abrindo o/s arquivo/s selecionado/s ####




janela.mainloop()