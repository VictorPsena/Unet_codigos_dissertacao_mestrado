"""
FilterInspectorCallback — visualiza os filtros convolucionais da U-Net
ao final de cada época.

O que faz:
  1. Salva grade de imagens dos filtros de cada camada Conv2D
  2. Conta filtros "mortos" (quase todos zeros / só preto)
  3. Gera um relatório .txt por época com estatísticas
  4. Opcional: salva histograma dos pesos por camada

Estrutura de pastas gerada:
  filtros/
    epoca_001/
      enc1_conv1.png
      enc1_conv2.png
      ...
      relatorio.txt
    epoca_002/
      ...
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib
# matplotlib.use("Agg")  # sem GUI — funciona em servidor
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime


class FilterInspectorCallback(tf.keras.callbacks.Callback):
    """
    Callback que inspeciona filtros convolucionais ao final de cada época.

    Parâmetros
    ----------
    output_dir   : pasta raiz onde salvar as imagens
    layers_watch : lista de nomes de camadas para inspecionar.
                   Se None, inspeciona TODAS as Conv2D do modelo.
    save_every   : salva a cada N épocas (padrão: 1)
    max_filters  : máximo de filtros a exibir por camada (padrão: 64)
    dead_threshold: filtro é "morto" se std dos pesos < esse valor
    hist_weights : se True, salva histograma dos pesos por camada
    """

    def __init__(
        self,
        output_dir="filtros",
        layers_watch=None,
        save_every=1,
        max_filters=64,
        dead_threshold=1e-3,
        hist_weights=True,
    ):
        super().__init__()
        self.output_dir    = output_dir
        self.layers_watch  = layers_watch
        self.save_every    = save_every
        self.max_filters   = max_filters
        self.dead_threshold = dead_threshold
        self.hist_weights  = hist_weights

    # ------------------------------------------------------------------
    # UTILITÁRIOS
    # ------------------------------------------------------------------

    def _get_conv_layers(self):
        """Retorna lista de camadas Conv2D a inspecionar."""
        layers = []
        for layer in self.model.layers:
            # Camadas Conv2D diretas
            if isinstance(layer, tf.keras.layers.Conv2D):
                if self.layers_watch is None or layer.name in self.layers_watch:
                    layers.append(layer)
            # Camadas dentro de sub-módulos (ex: FuzzyBottleneckModule)
            elif hasattr(layer, 'layers'):
                for sublayer in layer.layers:
                    if isinstance(sublayer, tf.keras.layers.Conv2D):
                        if self.layers_watch is None or sublayer.name in self.layers_watch:
                            layers.append(sublayer)
        return layers

    def _normalize_filter(self, f):
        """Normaliza filtro para [0, 1] para visualização."""
        f_min, f_max = f.min(), f.max()
        if f_max - f_min < 1e-8:
            return np.zeros_like(f)
        return (f - f_min) / (f_max - f_min)

    def _count_dead_filters(self, weights):
        """
        Conta filtros mortos: aqueles cujo desvio padrão é
        menor que dead_threshold (pesos quase todos iguais = filtro inativo).

        weights shape: (H, W, C_in, C_out)
        Cada filtro é weights[:, :, :, i]
        """
        n_filters = weights.shape[-1]
        dead = 0
        dead_indices = []
        for i in range(n_filters):
            filt = weights[:, :, :, i]
            if filt.std() < self.dead_threshold:
                dead += 1
                dead_indices.append(i)
        return dead, dead_indices

    # ------------------------------------------------------------------
    # VISUALIZAÇÃO DOS FILTROS
    # ------------------------------------------------------------------

    def _save_filter_grid(self, layer, epoch_dir, epoch):
        """
        Salva uma grade com todos os filtros da camada.
        Cada célula = um filtro (média dos canais de entrada).
        """
        weights = layer.get_weights()
        if not weights:
            return None, 0, []

        W = weights[0]  # (kH, kW, C_in, C_out)
        n_filters = W.shape[-1]
        n_show    = min(n_filters, self.max_filters)

        # Grade: tenta deixar próximo de quadrado
        n_cols = min(8, n_show)
        n_rows = int(np.ceil(n_show / n_cols))

        fig = plt.figure(figsize=(n_cols * 1.5, n_rows * 1.5 + 1))
        fig.suptitle(
            f"Camada: {layer.name}  |  Época {epoch+1}\n"
            f"Shape filtros: {W.shape}  |  Mostrando {n_show}/{n_filters}",
            fontsize=9, y=0.98
        )

        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                               hspace=0.05, wspace=0.05)

        dead, dead_indices = self._count_dead_filters(W)

        for i in range(n_show):
            ax = fig.add_subplot(gs[i // n_cols, i % n_cols])

            # Média sobre canais de entrada → imagem 2D
            filt = W[:, :, :, i].mean(axis=-1)   # (kH, kW)
            filt = self._normalize_filter(filt)

            # Filtros mortos ficam com borda vermelha
            is_dead = i in dead_indices
            ax.imshow(filt, cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])

            if is_dead:
                for spine in ax.spines.values():
                    spine.set_edgecolor("red")
                    spine.set_linewidth(2)

        # Legenda
        from matplotlib.patches import Patch
        legend = [Patch(facecolor="red",  label=f"Mortos: {dead}/{n_filters}"),
                  Patch(facecolor="navy", label="Ativos")]
        fig.legend(handles=legend, loc="lower center",
                   ncol=2, fontsize=8, bbox_to_anchor=(0.5, 0.0))

        fname = os.path.join(epoch_dir, f"{layer.name}.png")
        plt.savefig(fname, dpi=100, bbox_inches="tight")
        plt.close(fig)

        return W, dead, dead_indices

    # ------------------------------------------------------------------
    # HISTOGRAMA DE PESOS
    # ------------------------------------------------------------------

    def _save_weight_histogram(self, layer, epoch_dir):
        """Salva histograma da distribuição dos pesos da camada."""
        weights = layer.get_weights()
        if not weights:
            return

        W = weights[0].flatten()

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(W, bins=60, color="steelblue", edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="red", linestyle="--", linewidth=1, label="zero")
        ax.set_title(f"Pesos: {layer.name}", fontsize=9)
        ax.set_xlabel("Valor do peso")
        ax.set_ylabel("Frequência")
        ax.legend(fontsize=8)

        # Estatísticas no gráfico
        stats = f"μ={W.mean():.4f}  σ={W.std():.4f}\nmin={W.min():.4f}  max={W.max():.4f}"
        ax.text(0.98, 0.95, stats, transform=ax.transAxes,
                fontsize=7, va="top", ha="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        fname = os.path.join(epoch_dir, f"{layer.name}_hist.png")
        plt.savefig(fname, dpi=100, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # RELATÓRIO TEXTO
    # ------------------------------------------------------------------

    def _save_report(self, epoch_dir, epoch, report_data):
        """Salva relatório .txt com estatísticas de todas as camadas."""
        fname = os.path.join(epoch_dir, "relatorio.txt")
        total_mortos  = sum(d["mortos"] for d in report_data)
        total_filtros = sum(d["total"] for d in report_data)

        with open(fname, "w") as f:
            f.write(f"{'='*60}\n")
            f.write(f"RELATÓRIO DE FILTROS — Época {epoch+1}\n")
            f.write(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"RESUMO GERAL:\n")
            f.write(f"  Total de filtros:  {total_filtros}\n")
            f.write(f"  Filtros mortos:    {total_mortos} ({100*total_mortos/max(total_filtros,1):.1f}%)\n")
            f.write(f"  Filtros ativos:    {total_filtros - total_mortos}\n\n")
            f.write(f"{'='*60}\n")
            f.write(f"DETALHES POR CAMADA:\n")
            f.write(f"{'='*60}\n\n")

            for d in report_data:
                pct = 100 * d["mortos"] / max(d["total"], 1)
                alerta = " ⚠ ALTO" if pct > 30 else (" ✓" if pct == 0 else "")
                f.write(f"Camada: {d['nome']}\n")
                f.write(f"  Shape pesos:    {d['shape']}\n")
                f.write(f"  Total filtros:  {d['total']}\n")
                f.write(f"  Mortos (<std {self.dead_threshold}): {d['mortos']} ({pct:.1f}%){alerta}\n")
                f.write(f"  Média pesos:    {d['media']:.6f}\n")
                f.write(f"  Std pesos:      {d['std']:.6f}\n")
                f.write(f"  Min/Max:        {d['min']:.6f} / {d['max']:.6f}\n")
                if d["indices_mortos"]:
                    f.write(f"  Índices mortos: {d['indices_mortos'][:20]}")
                    if len(d["indices_mortos"]) > 20:
                        f.write(f" ... (+{len(d['indices_mortos'])-20} mais)")
                    f.write("\n")
                f.write("\n")

    # ------------------------------------------------------------------
    # HOOK PRINCIPAL
    # ------------------------------------------------------------------

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_every != 0:
            return

        # Cria pasta da época
        epoch_dir = os.path.join(self.output_dir, f"epoca_{epoch+1:03d}")
        os.makedirs(epoch_dir, exist_ok=True)

        conv_layers = self._get_conv_layers()
        report_data = []

        print(f"\n[FilterInspector] Época {epoch+1} — inspecionando {len(conv_layers)} camadas...")

        for layer in conv_layers:
            weights = layer.get_weights()
            if not weights:
                continue

            W = weights[0]  # (kH, kW, C_in, C_out)

            # Salva grade de filtros
            _, dead, dead_idx = self._save_filter_grid(layer, epoch_dir, epoch)

            # Salva histograma opcional
            if self.hist_weights:
                self._save_weight_histogram(layer, epoch_dir)

            # Coleta estatísticas para o relatório
            W_flat = W.flatten()
            report_data.append({
                "nome":          layer.name,
                "shape":         str(W.shape),
                "total":         W.shape[-1],
                "mortos":        dead,
                "indices_mortos": dead_idx,
                "media":         float(W_flat.mean()),
                "std":           float(W_flat.std()),
                "min":           float(W_flat.min()),
                "max":           float(W_flat.max()),
            })

        # Salva relatório
        self._save_report(epoch_dir, epoch, report_data)

        # Resumo no console
        total_m = sum(d["mortos"] for d in report_data)
        total_f = sum(d["total"]  for d in report_data)
        print(f"[FilterInspector] Salvo em '{epoch_dir}'")
        print(f"[FilterInspector] Filtros mortos: {total_m}/{total_f} "
              f"({100*total_m/max(total_f,1):.1f}%)\n")


# ==============================================================================
# EXEMPLO DE USO
# ==============================================================================

if __name__ == "__main__":

    # Inspeciona todas as Conv2D a cada época
    callback_todas = FilterInspectorCallback(
        output_dir="filtros",
        save_every=1,          # salva toda época
        max_filters=64,        # máximo de filtros por imagem
        dead_threshold=1e-3,   # filtro morto se std < 0.001
        hist_weights=True,     # salva histogramas
    )

    # Inspeciona só camadas específicas a cada 5 épocas
    callback_seletivo = FilterInspectorCallback(
        output_dir="filtros_seletivo",
        layers_watch=["enc1_conv1", "enc2_conv1", "bottleneck_conv1"],
        save_every=5,
        max_filters=32,
        dead_threshold=1e-3,
    )

    # Adiciona ao model.fit():
    # model.fit(
    #     X_train, y_train,
    #     validation_data=(X_val, y_val),
    #     epochs=50,
    #     callbacks=[
    #         callback_todas,
    #         tf.keras.callbacks.EarlyStopping(...),
    #         tf.keras.callbacks.ModelCheckpoint(...),
    #     ]
    # )

    print("Callbacks criados com sucesso!")
    print("\nComo usar:")
    print("  1. Importe: from filter_inspector import FilterInspectorCallback")
    print("  2. Instancie: cb = FilterInspectorCallback(output_dir='filtros')")
    print("  3. Adicione ao fit: model.fit(..., callbacks=[cb])")
    print("\nSaída gerada:")
    print("  filtros/")
    print("    epoca_001/")
    print("      enc1_conv1.png       ← grade de filtros (mortos = borda vermelha)")
    print("      enc1_conv1_hist.png  ← histograma dos pesos")
    print("      relatorio.txt        ← estatísticas detalhadas")
    print("    epoca_002/")
    print("      ...")