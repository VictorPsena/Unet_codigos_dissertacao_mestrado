import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image
import numpy as np
import pandas as pd

from pathlib import Path
# from funcs import banco

from keras.models import load_model

from redes.fuzzy_layer import get_fuzzy_custom_objects



BASE_DIR = Path(__file__).resolve().parent
PASTA_MODELOS = BASE_DIR / "modelos_finais" / "App"
CUSTOM_OBJECTS_FUZZY = get_fuzzy_custom_objects()

COR_FUNDO = "#f4f1ea"
COR_SUPERFICIE = "#fffdf8"
COR_DESTAQUE = "#2d6a4f"
COR_DESTAQUE_HOVER = "#40916c"
COR_TEXTO = "#1f2a1f"
COR_TEXTO_SUAVE = "#6b705c"
COR_BORDA = "#ddd6c8"
COR_INFO = "#e8f3eb"
COR_SUCESSO = "#d8f3dc"
COR_AVISO = "#faedcd"
COR_ERRO = "#f4d6d6"
FONTE_BASE = "DejaVu Sans"
MEDIDAS_DISPONIVEIS = ("AOL", "EGL")
ARQUIVOS_PADRAO_MEDIDA = {
    "AOL": "resultados_area_aol.xlsx",
    "EGL": "resultados_medida_egl.xlsx",
}

arquivos_selecionados = []

janela = tk.Tk()
janela.title("Medidas")
janela.geometry("1200x920")
janela.minsize(1100, 720)
janela.configure(bg=COR_FUNDO)


def configurar_estilos():
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Shell.TFrame", background=COR_FUNDO)
    style.configure("Card.TFrame", background=COR_SUPERFICIE)
    style.configure(
        "Title.TLabel",
        background=COR_FUNDO,
        foreground=COR_TEXTO,
        font=(FONTE_BASE, 24, "bold"),
    )
    style.configure(
        "Subtitle.TLabel",
        background=COR_FUNDO,
        foreground=COR_TEXTO_SUAVE,
        font=(FONTE_BASE, 11),
    )
    style.configure(
        "SectionTitle.TLabel",
        background=COR_SUPERFICIE,
        foreground=COR_TEXTO,
        font=(FONTE_BASE, 12, "bold"),
    )
    style.configure(
        "Body.TLabel",
        background=COR_SUPERFICIE,
        foreground=COR_TEXTO_SUAVE,
        font=(FONTE_BASE, 10),
    )
    style.configure(
        "Badge.TLabel",
        background="#e7f1ea",
        foreground=COR_DESTAQUE,
        font=(FONTE_BASE, 10, "bold"),
        padding=(12, 6),
    )
    style.configure(
        "Accent.TButton",
        background=COR_DESTAQUE,
        foreground="#ffffff",
        font=(FONTE_BASE, 10, "bold"),
        padding=(18, 10),
        borderwidth=0,
        relief="flat",
    )
    style.map(
        "Accent.TButton",
        background=[("active", COR_DESTAQUE_HOVER), ("disabled", "#b7c6bb")],
        foreground=[("disabled", "#f7f4ee")],
    )
    style.configure(
        "Secondary.TButton",
        background=COR_SUPERFICIE,
        foreground=COR_TEXTO,
        font=(FONTE_BASE, 10),
        padding=(14, 10),
        borderwidth=1,
        relief="solid",
    )
    style.map(
        "Secondary.TButton",
        background=[("active", "#f5f1e8"), ("disabled", "#f4efe7")],
        foreground=[("disabled", "#9a9f97")],
    )
    style.configure(
        "Modern.Horizontal.TProgressbar",
        background=COR_DESTAQUE,
        troughcolor="#e8e1d4",
        lightcolor=COR_DESTAQUE,
        darkcolor=COR_DESTAQUE,
        bordercolor="#e8e1d4",
    )
    style.configure(
        "TCombobox",
        padding=8,
        fieldbackground="#ffffff",
        background="#ffffff",
        foreground=COR_TEXTO,
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", "#ffffff")],
        selectbackground=[("readonly", COR_DESTAQUE_HOVER)],
        selectforeground=[("readonly", "#ffffff")],
    )


configurar_estilos()

modelos_disponiveis = [f.name for f in PASTA_MODELOS.glob("*.keras")]
modelo_padrao = modelos_disponiveis[0] if modelos_disponiveis else "Nenhum modelo disponível"
modelo_selecionado = tk.StringVar(value=modelo_padrao)
status_var = tk.StringVar(value="Selecione um modelo e as imagens para começar.")
resumo_arquivos_var = tk.StringVar(value="Nenhuma imagem selecionada.")
detalhe_arquivo_var = tk.StringVar(value="Os nomes dos arquivos selecionados aparecerão aqui.")
resumo_resultados_var = tk.StringVar(value="As visualizações serão exibidas nesta área após o processamento.")
badge_modelos_var = tk.StringVar(value=f"{len(modelos_disponiveis)} modelo(s) disponível(is)")
badge_arquivos_var = tk.StringVar(value="0 imagem(ns) selecionada(s)")
medida_selecionada = tk.StringVar(value=MEDIDAS_DISPONIVEIS[0])


def definir_status(mensagem, tipo="info"):
    paleta = {
        "info": (COR_INFO, COR_DESTAQUE),
        "success": (COR_SUCESSO, "#1b4332"),
        "warning": (COR_AVISO, "#9c6644"),
        "error": (COR_ERRO, "#9d0208"),
    }
    fundo, texto = paleta.get(tipo, paleta["info"])
    status_var.set(mensagem)
    status_box.configure(bg=fundo, highlightbackground=fundo)
    status_label.configure(bg=fundo, fg=texto)


def atualizar_estado_acoes(processando=False):
    if processando:
        botao_arquivos.state(["disabled"])
        botao_limpar.state(["disabled"])
        botao_processar.state(["disabled"])
        return

    if arquivos_selecionados:
        botao_limpar.state(["!disabled"])
    else:
        botao_limpar.state(["disabled"])

    if modelos_disponiveis and arquivos_selecionados:
        botao_processar.state(["!disabled"])
    else:
        botao_processar.state(["disabled"])

    botao_arquivos.state(["!disabled"])


def atualizar_detalhe_arquivo(event=None):
    selecao = lista_arquivos.curselection()
    if not selecao:
        detalhe_arquivo_var.set("Os nomes dos arquivos selecionados aparecerão aqui.")
        return

    detalhe_arquivo_var.set(arquivos_selecionados[selecao[0]])


def atualizar_lista_arquivos():
    lista_arquivos.delete(0, tk.END)

    for caminho in arquivos_selecionados:
        lista_arquivos.insert(tk.END, Path(caminho).name)

    quantidade = len(arquivos_selecionados)
    resumo_arquivos_var.set(
        f"{quantidade} arquivo(s) pronto(s) para processamento." if quantidade else "Nenhuma imagem selecionada."
    )
    badge_arquivos_var.set(f"{quantidade} imagem(ns) selecionada(s)")
    atualizar_detalhe_arquivo()
    atualizar_estado_acoes()


def escolher_arquivos():
    global arquivos_selecionados

    caminhos = filedialog.askopenfilenames(
        title="Selecione uma ou mais imagens",
        initialdir=str(BASE_DIR),
        filetypes=[
            ("Imagens", "*.png *.jpg *.jpeg"),
            ("Todos os arquivos", "*.*"),
        ],
    )

    if not caminhos:
        definir_status("Seleção de arquivos cancelada.", "warning")
        return

    arquivos_selecionados = list(caminhos)
    atualizar_lista_arquivos()
    limpar_figuras("Carregamento concluído. Clique em 'Processar imagens' para gerar as visualizações.")
    definir_status(f"{len(arquivos_selecionados)} imagem(ns) selecionada(s).", "info")


def limpar_arquivos():
    global arquivos_selecionados

    arquivos_selecionados = []
    atualizar_lista_arquivos()
    limpar_figuras("As visualizações serão exibidas aqui após um novo processamento.")
    definir_status("Seleção removida. Escolha novas imagens para continuar.", "info")


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
    return "break"


def limpar_figuras(mensagem=None):
    for widget in frame_plot.winfo_children():
        widget.destroy()

    cartao_vazio = tk.Frame(
        frame_plot,
        bg=COR_SUPERFICIE,
        highlightbackground=COR_BORDA,
        highlightthickness=1,
        padx=20,
        pady=20,
    )
    cartao_vazio.pack(fill="both", expand=True, padx=8, pady=8)

    tk.Label(
        cartao_vazio,
        text="Nenhuma visualização carregada",
        bg=COR_SUPERFICIE,
        fg=COR_TEXTO,
        font=(FONTE_BASE, 14, "bold"),
        anchor="w",
    ).pack(anchor="w")
    tk.Label(
        cartao_vazio,
        text=mensagem or "As figuras e gráficos aparecerão aqui.",
        bg=COR_SUPERFICIE,
        fg=COR_TEXTO_SUAVE,
        font=(FONTE_BASE, 10),
        justify="left",
        wraplength=760,
        anchor="w",
    ).pack(anchor="w", pady=(8, 0))

    resumo_resultados_var.set(mensagem or "As visualizações serão exibidas nesta área após o processamento.")
    janela.update_idletasks()
    _atualizar_scrollregion()
    plot_canvas.yview_moveto(0)


def mostrar_figuras_no_tk(figs):
    for widget in frame_plot.winfo_children():
        widget.destroy()

    if not figs:
        limpar_figuras("O processamento terminou, mas não retornou figuras para exibição.")
        return

    for indice, fig in enumerate(figs, start=1):
        cartao_figura = tk.Frame(
            frame_plot,
            bg="#ffffff",
            highlightbackground=COR_BORDA,
            highlightthickness=1,
            padx=12,
            pady=12,
        )
        cartao_figura.pack(fill="x", expand=True, padx=8, pady=8)

        tk.Label(
            cartao_figura,
            text=f"Visualização {indice}",
            bg="#ffffff",
            fg=COR_TEXTO,
            font=(FONTE_BASE, 11, "bold"),
            anchor="w",
        ).pack(anchor="w", pady=(0, 10))

        canvas = FigureCanvasTkAgg(fig, master=cartao_figura)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill="x", expand=True)
        canvas_widget.bind("<MouseWheel>", _mousewheel)
        canvas_widget.bind("<Button-4>", _mousewheel)
        canvas_widget.bind("<Button-5>", _mousewheel)

    resumo_resultados_var.set(f"{len(figs)} visualização(ões) exibida(s) na área principal.")
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

def _normalizar_saida_modelo(predicao):
    if isinstance(predicao, (list, tuple)):
        predicao = predicao[0]

    predicao = np.asarray(predicao)
    predicao = np.squeeze(predicao)

    if predicao.ndim == 3:
        if predicao.shape[-1] > 1:
            predicao = np.argmax(predicao, axis=-1)
        else:
            predicao = predicao[..., 0]

    return predicao


def _montar_entrada_modelo(modelo, imagem):
    imagem_batch = np.expand_dims(imagem, axis=0)
    input_shape = modelo.input_shape

    if not isinstance(input_shape, list):
        return imagem_batch

    entradas = [imagem_batch]
    for shape in input_shape[1:]:
        dims = [1 if dim is None else int(dim) for dim in shape[1:]]

        if len(dims) == 1 and dims[0] == 2:
            entradas.append(np.array([[0.0, 1.0]], dtype=np.float32))
        else:
            entradas.append(np.zeros((1, *dims), dtype=np.float32))

    return tuple(entradas)


def _resolver_funcao_medida(nome_funcao):
    from funcs import AreaAOL, MedidaEGL

    funcoes = {
        "AOL": AreaAOL,
        "EGL": MedidaEGL,
    }

    try:
        return funcoes[nome_funcao]
    except KeyError as exc:
        raise ValueError(f"Função de medida inválida: {nome_funcao}") from exc


def _salvar_relatorio_medidas(nome_funcao, predicoes, nomes_arquivos, nome_modelo):
    if not predicoes:
        return None

    caminho_saida = filedialog.asksaveasfilename(
        title="Salvar relatório de medidas",
        initialdir=str(BASE_DIR),
        initialfile=ARQUIVOS_PADRAO_MEDIDA.get(nome_funcao, "resultados_medidas.xlsx"),
        defaultextension=".xlsx",
        filetypes=[("Planilha Excel", "*.xlsx")],
    )

    if not caminho_saida:
        return None

    funcao_medida = _resolver_funcao_medida(nome_funcao)
    registros = []

    for nome_arquivo, predicao in zip(nomes_arquivos, predicoes):
        resultado_df = funcao_medida(predicao, save_path=None)
        if resultado_df is None or resultado_df.empty:
            continue

        registro = {
            "arquivo": nome_arquivo,
            "modelo": nome_modelo,
            "funcao_medida": nome_funcao,
        }
        registro.update(resultado_df.iloc[0].to_dict())
        registros.append(registro)

    if not registros:
        raise ValueError("Nenhuma medida pôde ser calculada para as predições geradas.")

    pd.DataFrame(registros).to_excel(caminho_saida, index=False)
    return Path(caminho_saida)


def _criar_figura_resultado(imagem, predicao, titulo):
    imagem_plot = np.squeeze(imagem)
    if imagem_plot.ndim == 3 and imagem_plot.shape[-1] == 1:
        imagem_plot = imagem_plot[..., 0]

    mapa = _normalizar_saida_modelo(predicao)

    if np.issubdtype(mapa.dtype, np.floating):
        overlay = np.ma.masked_where(mapa < 0.5, mapa)
    else:
        overlay = np.ma.masked_where(mapa == 0, mapa)

    fig = Figure(figsize=(9, 4.2), dpi=100)
    eixo_imagem = fig.add_subplot(1, 2, 1)
    eixo_overlay = fig.add_subplot(1, 2, 2)

    eixo_imagem.imshow(imagem_plot, cmap="gray" if imagem_plot.ndim == 2 else None)
    eixo_imagem.set_title("Imagem")
    eixo_imagem.axis("off")

    eixo_overlay.imshow(imagem_plot, cmap="gray" if imagem_plot.ndim == 2 else None)
    eixo_overlay.imshow(overlay, cmap="Reds", alpha=0.35)
    eixo_overlay.set_title("Predição sobreposta")
    eixo_overlay.axis("off")

    fig.suptitle(titulo)
    fig.tight_layout()
    return fig

def _carregar_imagens(paths, largura, altura, canais):
    imagens = []
    grayscale = canais == 1

    for caminho in paths:
        img = Image.open(caminho)

        esquerda = largura * 0.19
        superior = altura * 0.188
        direita = largura * 1.8
        inferior = altura *1.7

        img = img.crop((esquerda, superior, direita, inferior))
        img = img.resize((largura, altura))

        if grayscale:
            img = img.convert("L")
        else:
            img = img.convert("RGB")

        arr = np.asarray(img, dtype=np.float32) / 255.0
        if grayscale:
            arr = np.expand_dims(arr, axis=-1)

        imagens.append(arr)

    return np.asarray(imagens, dtype=np.float32)


def processar_e_plotar():
    if not modelos_disponiveis:
        messagebox.showerror("Erro", "Nenhum modelo disponível na pasta de modelos.")
        definir_status("Nenhum modelo .keras foi encontrado para processamento.", "error")
        return

    if not arquivos_selecionados:
        messagebox.showwarning("Aviso", "Selecione pelo menos um arquivo.")
        definir_status("Selecione pelo menos uma imagem antes de processar.", "warning")
        return

    caminho_modelo = PASTA_MODELOS / modelo_selecionado.get()
    if not caminho_modelo.exists():
        messagebox.showerror("Erro", f"Modelo não encontrado: {caminho_modelo}")
        definir_status("O modelo escolhido não foi encontrado no diretório esperado.", "error")
        return

    try:
        atualizar_estado_acoes(processando=True)
        barra_progresso.start(12)
        resumo_resultados_var.set("Carregando modelo e preparando imagens...")
        definir_status("Processamento em andamento. Aguarde alguns instantes.", "info")
        janela.update_idletasks()

        model = load_model(caminho_modelo, compile=False, custom_objects=CUSTOM_OBJECTS_FUZZY)
        altura, largura, canais = _inferir_shape_entrada(model)
        x_imgs = _carregar_imagens(arquivos_selecionados, largura, altura, canais)

        figs = []
        predicoes = []
        nomes_dados = [Path(p).name for p in arquivos_selecionados]

        for idx, imagem in enumerate(x_imgs):
            x_input = _montar_entrada_modelo(model, imagem)
            predicao = model.predict(x_input, verbose=0)
            predicoes.append(predicao)
            figs.append(_criar_figura_resultado(imagem, predicao, nomes_dados[idx]))
            

        mostrar_figuras_no_tk(figs)
        caminho_relatorio = _salvar_relatorio_medidas(
            medida_selecionada.get(),
            predicoes,
            nomes_dados,
            caminho_modelo.name,
        )

        if caminho_relatorio:
            resumo_resultados_var.set(
                f"{len(figs)} visualização(ões) exibida(s) e relatório salvo em '{caminho_relatorio.name}'."
            )
            definir_status(
                f"Modelo '{caminho_modelo.name}' processado e relatório salvo com {len(predicoes)} medida(s).",
                "success",
            )
        else:
            resumo_resultados_var.set(
                f"{len(figs)} visualização(ões) exibida(s). O salvamento do relatório foi cancelado."
            )
            definir_status(
                "Processamento concluído, mas o relatório .xlsx não foi salvo.",
                "warning",
            )

        if not figs:
            resumo_resultados_var.set(
                f"Processamento concluído para {len(x_imgs)} imagem(ns), mas sem figuras geradas pela rotina atual."
            )
    except Exception as erro:
        messagebox.showerror("Erro no processamento", str(erro))
        definir_status("Falha no processamento. Revise o modelo e os arquivos selecionados.", "error")
        resumo_resultados_var.set("O processamento falhou antes da geração das visualizações.")
    finally:
        barra_progresso.stop()
        atualizar_estado_acoes()


main_container = ttk.Frame(janela, style="Shell.TFrame", padding=24)
main_container.grid(row=0, column=0, sticky="nsew")
janela.grid_rowconfigure(0, weight=1)
janela.grid_columnconfigure(0, weight=1)

main_container.grid_rowconfigure(1, weight=1)
main_container.grid_columnconfigure(0, weight=0)
main_container.grid_columnconfigure(1, weight=1)

header_frame = ttk.Frame(main_container, style="Shell.TFrame")
header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 18))
header_frame.grid_columnconfigure(0, weight=1)

ttk.Label(header_frame, text="AOGL", style="Title.TLabel").grid(row=0, column=0, sticky="w")
ttk.Label(
    header_frame,
    text="Painel de Visualização",
    style="Subtitle.TLabel",
    wraplength=760,
    justify="left",
).grid(row=1, column=0, sticky="w", pady=(6, 0))

header_badges = ttk.Frame(header_frame, style="Shell.TFrame")
header_badges.grid(row=0, column=1, rowspan=2, sticky="e")
ttk.Label(header_badges, textvariable=badge_modelos_var, style="Badge.TLabel").grid(row=0, column=0, padx=(0, 8))
ttk.Label(header_badges, textvariable=badge_arquivos_var, style="Badge.TLabel").grid(row=0, column=1)

painel_controles = ttk.Frame(main_container, style="Shell.TFrame")
painel_controles.grid(row=1, column=0, sticky="nsew", padx=(0, 18))
painel_controles.grid_rowconfigure(1, weight=1)

cartao_modelo = ttk.Frame(painel_controles, style="Card.TFrame", padding=18)
cartao_modelo.grid(row=0, column=0, sticky="ew", pady=(0, 16))
cartao_modelo.grid_columnconfigure(0, weight=1)
ttk.Label(cartao_modelo, text="1. Modelo e medidas", style="SectionTitle.TLabel").grid(row=0, column=0, sticky="w")
ttk.Label(
    cartao_modelo,
    text="Escolha a Rede Neural.",
    style="Body.TLabel",
    wraplength=300,
    justify="left",
).grid(row=1, column=0, sticky="w", pady=(6, 14))

combo_estado = "readonly" if modelos_disponiveis else "disabled"
combo_modelos = ttk.Combobox(
    cartao_modelo,
    textvariable=modelo_selecionado,
    values=modelos_disponiveis,
    state=combo_estado,
    width=34,
)
combo_modelos.grid(row=2, column=0, sticky="ew")

ttk.Label(
    cartao_modelo,
    text="Escolha a medida a ser calculada.",
    style="Body.TLabel",
).grid(row=3, column=0, sticky="w", pady=(14, 6))

combo_medidas = ttk.Combobox(
    cartao_modelo,
    textvariable=medida_selecionada,
    values=MEDIDAS_DISPONIVEIS,
    state="readonly",
    width=34,
)
combo_medidas.grid(row=4, column=0, sticky="ew")

ttk.Label(
    cartao_modelo,
    text="Ao final do processamento, o aplicativo pedirá onde salvar o arquivo com as medidas.",
    style="Body.TLabel",
    wraplength=300,
    justify="left",
).grid(row=5, column=0, sticky="w", pady=(10, 0))

cartao_arquivos = ttk.Frame(painel_controles, style="Card.TFrame", padding=18)
cartao_arquivos.grid(row=1, column=0, sticky="nsew", pady=(0, 16))
cartao_arquivos.grid_rowconfigure(3, weight=1)
cartao_arquivos.grid_columnconfigure(0, weight=1)

ttk.Label(cartao_arquivos, text="2. Imagens", style="SectionTitle.TLabel").grid(row=0, column=0, sticky="w")
ttk.Label(
    cartao_arquivos,
    textvariable=resumo_arquivos_var,
    style="Body.TLabel",
    wraplength=300,
    justify="left",
).grid(row=1, column=0, sticky="w", pady=(6, 12))

acoes_arquivos = ttk.Frame(cartao_arquivos, style="Card.TFrame")
acoes_arquivos.grid(row=2, column=0, sticky="ew", pady=(0, 12))
botao_arquivos = ttk.Button(
    acoes_arquivos,
    text="Escolher imagens",
    command=escolher_arquivos,
    style="Secondary.TButton",
)
botao_arquivos.grid(row=0, column=0, sticky="ew", padx=(0, 8))
botao_limpar = ttk.Button(
    acoes_arquivos,
    text="Limpar",
    command=limpar_arquivos,
    style="Secondary.TButton",
)
botao_limpar.grid(row=0, column=1, sticky="ew")
acoes_arquivos.grid_columnconfigure(0, weight=1)
acoes_arquivos.grid_columnconfigure(1, weight=1)

lista_container = tk.Frame(
    cartao_arquivos,
    bg="#ffffff",
    highlightbackground=COR_BORDA,
    highlightthickness=1,
)
lista_container.grid(row=3, column=0, sticky="nsew")
lista_container.grid_rowconfigure(0, weight=1)
lista_container.grid_columnconfigure(0, weight=1)

lista_arquivos = tk.Listbox(
    lista_container,
    bg="#ffffff",
    fg=COR_TEXTO,
    bd=0,
    highlightthickness=0,
    activestyle="none",
    font=(FONTE_BASE, 10),
    selectbackground="#cde5d3",
    selectforeground=COR_TEXTO,
)
lista_arquivos.grid(row=0, column=0, sticky="nsew")
lista_arquivos.bind("<<ListboxSelect>>", atualizar_detalhe_arquivo)

scroll_arquivos = ttk.Scrollbar(lista_container, orient="vertical", command=lista_arquivos.yview)
scroll_arquivos.grid(row=0, column=1, sticky="ns")
lista_arquivos.configure(yscrollcommand=scroll_arquivos.set)

ttk.Label(
    cartao_arquivos,
    textvariable=detalhe_arquivo_var,
    style="Body.TLabel",
    wraplength=300,
    justify="left",
).grid(row=4, column=0, sticky="ew", pady=(12, 0))

cartao_acoes = ttk.Frame(painel_controles, style="Card.TFrame", padding=18)
cartao_acoes.grid(row=2, column=0, sticky="ew")
cartao_acoes.grid_columnconfigure(0, weight=1)

ttk.Label(cartao_acoes, text="3. Processamento", style="SectionTitle.TLabel").grid(row=0, column=0, sticky="w")
ttk.Label(
    cartao_acoes,
    text="Processamento após seleção do modelo, função de medida e das imagens.",
    style="Body.TLabel",
    wraplength=300,
    justify="left",
).grid(row=1, column=0, sticky="w", pady=(6, 14))

status_box = tk.Frame(cartao_acoes, bg=COR_INFO, highlightbackground=COR_INFO, highlightthickness=1, padx=12, pady=12)
status_box.grid(row=2, column=0, sticky="ew")
status_label = tk.Label(
    status_box,
    textvariable=status_var,
    bg=COR_INFO,
    fg=COR_DESTAQUE,
    font=(FONTE_BASE, 10, "bold"),
    justify="left",
    anchor="w",
    wraplength=300,
)
status_label.pack(fill="x")

barra_progresso = ttk.Progressbar(
    cartao_acoes,
    mode="indeterminate",
    style="Modern.Horizontal.TProgressbar",
)
barra_progresso.grid(row=3, column=0, sticky="ew", pady=(14, 14))

botao_processar = ttk.Button(
    cartao_acoes,
    text="Processar imagens",
    command=processar_e_plotar,
    style="Accent.TButton",
)
botao_processar.grid(row=4, column=0, sticky="ew")

painel_resultados = ttk.Frame(main_container, style="Card.TFrame", padding=18)
painel_resultados.grid(row=1, column=1, sticky="nsew")
painel_resultados.grid_rowconfigure(2, weight=1)
painel_resultados.grid_columnconfigure(0, weight=1)

ttk.Label(painel_resultados, text="Resultados e visualizações", style="SectionTitle.TLabel").grid(row=0, column=0, sticky="w")
ttk.Label(
    painel_resultados,
    textvariable=resumo_resultados_var,
    style="Body.TLabel",
    wraplength=760,
    justify="left",
).grid(row=1, column=0, sticky="w", pady=(6, 14))

plot_wrapper = tk.Frame(
    painel_resultados,
    bg=COR_SUPERFICIE,
    highlightbackground=COR_BORDA,
    highlightthickness=1,
)
plot_wrapper.grid(row=2, column=0, sticky="nsew")
plot_wrapper.grid_rowconfigure(0, weight=1)
plot_wrapper.grid_columnconfigure(0, weight=1)

plot_canvas = tk.Canvas(plot_wrapper, bg=COR_SUPERFICIE, bd=0, highlightthickness=0)
plot_canvas.grid(row=0, column=0, sticky="nsew")

scrollbar_y = ttk.Scrollbar(plot_wrapper, orient="vertical", command=plot_canvas.yview)
scrollbar_y.grid(row=0, column=1, sticky="ns")
plot_canvas.configure(yscrollcommand=scrollbar_y.set)

frame_plot = tk.Frame(plot_canvas, bg=COR_SUPERFICIE)
plot_window = plot_canvas.create_window((0, 0), window=frame_plot, anchor="nw")

frame_plot.bind("<Configure>", _atualizar_scrollregion)
plot_canvas.bind("<Configure>", _ajustar_largura_frame)
plot_canvas.bind("<MouseWheel>", _mousewheel)
plot_canvas.bind("<Button-4>", _mousewheel)
plot_canvas.bind("<Button-5>", _mousewheel)
frame_plot.bind("<MouseWheel>", _mousewheel)
frame_plot.bind("<Button-4>", _mousewheel)
frame_plot.bind("<Button-5>", _mousewheel)

atualizar_lista_arquivos()
limpar_figuras()

if not modelos_disponiveis:
    definir_status("Nenhum modelo .keras foi encontrado na pasta configurada.", "warning")

janela.mainloop()