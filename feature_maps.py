"""
FeatureMapCallback — visualiza os feature maps gerados após cada
camada Conv2D ao final de cada época.

O que faz:
  1. Passa uma imagem de referência pelo modelo
  2. Captura a saída de cada camada Conv2D via modelo intermediário
  3. Salva grade de imagens dos feature maps por camada
  4. Conta mapas "mortos" (todos zeros — neurônio inativo para essa imagem)
  5. Gera relatório com estatísticas de ativação

"""

import os
import numpy as np
import tensorflow as tf
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime


class FeatureMapCallback(tf.keras.callbacks.Callback):
    """
    Visualiza feature maps de cada Conv2D ao final de cada época.

    Parâmetros
    ----------
    sample_image  : imagem de referência, shape (H, W, C) ou (1, H, W, C)
    output_dir    : pasta raiz onde salvar as imagens
    layers_watch  : lista de nomes de camadas. Se None, usa todas as Conv2D
    save_every    : salva a cada N épocas
    max_maps      : máximo de feature maps a exibir por camada
    dead_threshold: mapa é "morto" se max(|valores|) < esse threshold
    cmap          : colormap matplotlib (ex: 'viridis', 'gray', 'plasma')
    """

    def __init__(
        self,
        sample_image,
        output_dir="feature_maps",
        layers_watch=None,
        save_every=1,
        max_maps=64,
        dead_threshold=1e-4,
        cmap="viridis",
    ):
        super().__init__()

        # Aceita entrada única (imagem) ou estrutura completa (lista/dict)
        # para modelos com múltiplas entradas.
        self.sample_image  = sample_image

        self.output_dir    = output_dir
        self.layers_watch  = layers_watch
        self.save_every    = save_every
        self.max_maps      = max_maps
        self.dead_threshold = dead_threshold
        self.cmap          = cmap

        # Modelo intermediário — construído no on_train_begin
        self._activation_model = None
        self._warned_auto_fill = False

    # ------------------------------------------------------------------
    # CONSTRÓI MODELO INTERMEDIÁRIO
    # ------------------------------------------------------------------

    def on_train_begin(self, logs=None):
        """
        Cria um modelo que retorna as saídas de cada Conv2D.
        Precisa ser feito aqui porque o modelo só está disponível
        depois de model.fit() ser chamado.
        """
        self._activation_model = self._build_activation_model()
        print(f"\n[FeatureMapCallback] Monitorando {len(self._activation_model.outputs)} camadas.")

    def _get_target_layers(self):
        """Retorna camadas Conv2D a monitorar."""
        target = []
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                if self.layers_watch is None or layer.name in self.layers_watch:
                    target.append(layer)
            # Sub-módulos (ex: FuzzyBottleneckModule)
            elif hasattr(layer, 'layers'):
                for sublayer in layer.layers:
                    if isinstance(sublayer, tf.keras.layers.Conv2D):
                        if self.layers_watch is None or sublayer.name in self.layers_watch:
                            target.append(sublayer)
        return target

    def _build_activation_model(self):
        """
        Cria modelo com múltiplas saídas — uma por camada Conv2D.
        Usa a saída APÓS a ativação (o que de fato propaga para frente).
        """
        target_layers = self._get_target_layers()

        outputs = []
        names   = []
        for layer in target_layers:
            outputs.append(layer.output)
            names.append(layer.name)

        if not outputs:
            raise ValueError("Nenhuma camada Conv2D encontrada no modelo.")

        activation_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=outputs,
            name="activation_model"
        )
        self._layer_names = names
        return activation_model

    def _tensor_name(self, tensor):
        """Extrai nome estável de um KerasTensor."""
        return tensor.name.split(":")[0]

    def _zeros_for_input(self, input_tensor):
        """Cria entrada zerada compatível com o shape esperado."""
        shape = [d if d is not None else 1 for d in input_tensor.shape]
        return np.zeros(shape, dtype=np.float32)

    def _prepare_single_input(self, sample, input_tensor):
        """Converte input para float32 e ajusta batch quando necessário."""
        arr = np.asarray(sample, dtype=np.float32)
        expected_rank = len(input_tensor.shape)

        # Caso comum: sem batch explícito (ex: (H, W, C) ou (C,))
        if arr.ndim == expected_rank - 1:
            arr = np.expand_dims(arr, axis=0)

        return arr

    def _prepare_model_inputs(self):
        """
        Monta entradas no formato exigido pelo modelo.

        Regras:
          - Modelo 1 entrada: usa sample_image.
          - Modelo N entradas + sample_image único: usa sample_image na 1a
            entrada e preenche as demais com zeros.
          - sample_image list/tuple/dict: tenta casar posição/nome.
        """
        model_inputs = self._activation_model.inputs

        if len(model_inputs) == 1:
            return self._prepare_single_input(self.sample_image, model_inputs[0])

        if isinstance(self.sample_image, (list, tuple)):
            if len(self.sample_image) != len(model_inputs):
                raise ValueError(
                    "sample_image como lista/tupla deve ter o mesmo número "
                    f"de entradas do modelo ({len(model_inputs)})."
                )
            return [
                self._prepare_single_input(s, t)
                for s, t in zip(self.sample_image, model_inputs)
            ]

        if isinstance(self.sample_image, dict):
            prepared = []
            for tensor in model_inputs:
                key = self._tensor_name(tensor)
                if key not in self.sample_image:
                    raise ValueError(
                        f"Entrada '{key}' não encontrada em sample_image (dict)."
                    )
                prepared.append(self._prepare_single_input(self.sample_image[key], tensor))
            return prepared

        prepared = [self._prepare_single_input(self.sample_image, model_inputs[0])]
        for tensor in model_inputs[1:]:
            prepared.append(self._zeros_for_input(tensor))

        if not self._warned_auto_fill:
            print(
                "[FeatureMapCallback] Modelo com múltiplas entradas detectado. "
                "Entradas auxiliares serão preenchidas com zeros para "
                "gerar os feature maps da imagem de referência."
            )
            self._warned_auto_fill = True

        return prepared

    # ------------------------------------------------------------------
    # UTILITÁRIOS
    # ------------------------------------------------------------------

    def _normalize_map(self, fmap):
        """Normaliza feature map para [0, 1]."""
        vmin, vmax = fmap.min(), fmap.max()
        if vmax - vmin < 1e-8:
            return np.zeros_like(fmap)
        return (fmap - vmin) / (vmax - vmin)

    def _count_dead_maps(self, activation):
        """
        Conta feature maps mortos: aqueles onde max(|valores|) < threshold.
        Significa que para essa imagem, o filtro não ativou nada.

        activation shape: (1, H, W, C)
        """
        n = activation.shape[-1]
        dead = []
        for i in range(n):
            fmap = activation[0, :, :, i]
            if np.max(np.abs(fmap)) < self.dead_threshold:
                dead.append(i)
        return dead

    # ------------------------------------------------------------------
    # VISUALIZAÇÃO
    # ------------------------------------------------------------------

    def _save_feature_map_grid(self, activation, layer_name, epoch_dir, epoch):
        """
        Salva grade de feature maps de uma camada.

        activation shape: (1, H, W, C)
        """
        n_maps  = activation.shape[-1]
        n_show  = min(n_maps, self.max_maps)
        H, W    = activation.shape[1], activation.shape[2]

        dead_indices = self._count_dead_maps(activation)
        n_dead = len(dead_indices)

        # Grade próxima de quadrado
        n_cols = min(8, n_show)
        n_rows = int(np.ceil(n_show / n_cols))

        # Tamanho da figura proporcional ao feature map
        cell_size = max(1.0, min(2.0, 32 / max(H, W)))
        fig = plt.figure(figsize=(n_cols * cell_size + 1,
                                  n_rows * cell_size + 1.5))

        fig.suptitle(
            f"Feature Maps — {layer_name}  |  Época {epoch+1}\n"
            f"Shape: (H={H}, W={W}, C={n_maps})  |  "
            f"Mostrando {n_show}/{n_maps}  |  "
            f"Mortos: {n_dead}/{n_maps} ({100*n_dead/max(n_maps,1):.0f}%)",
            fontsize=9, y=0.99
        )

        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                               hspace=0.08, wspace=0.08)

        for i in range(n_show):
            ax = fig.add_subplot(gs[i // n_cols, i % n_cols])
            fmap = activation[0, :, :, i]          # (H, W)
            fmap_norm = self._normalize_map(fmap)

            ax.imshow(fmap_norm, cmap=self.cmap, vmin=0, vmax=1)
            ax.set_title(f"#{i}", fontsize=5, pad=1)
            ax.set_xticks([])
            ax.set_yticks([])

            # Borda vermelha = mapa morto
            is_dead = i in dead_indices
            color = "red" if is_dead else "none"
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(1.5 if is_dead else 0)

        # Legenda
        from matplotlib.patches import Patch
        legend = [
            Patch(facecolor="red",   label=f"Mortos (inativos): {n_dead}"),
            Patch(facecolor="navy",  label=f"Ativos: {n_maps - n_dead}"),
        ]
        fig.legend(handles=legend, loc="lower center",
                   ncol=2, fontsize=7, bbox_to_anchor=(0.5, 0.0))

        safe_name = layer_name.replace("/", "_")
        fname = os.path.join(epoch_dir, f"{safe_name}.png")
        plt.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close(fig)

    def _save_activation_stats(self, activation, layer_name):
        """Retorna dict com estatísticas de ativação da camada."""
        flat = activation.flatten()
        dead = self._count_dead_maps(activation)
        return {
            "nome":    layer_name,
            "shape":   str(activation.shape[1:]),   # sem batch
            "n_maps":  activation.shape[-1],
            "mortos":  len(dead),
            "media":   float(flat.mean()),
            "std":     float(flat.std()),
            "min":     float(flat.min()),
            "max":     float(flat.max()),
            "pct_zero": float((np.abs(flat) < self.dead_threshold).mean() * 100),
        }

    # ------------------------------------------------------------------
    # RELATÓRIO
    # ------------------------------------------------------------------

    def _save_report(self, epoch_dir, epoch, stats_list):
        total_maps  = sum(s["n_maps"] for s in stats_list)
        total_mortos = sum(s["mortos"] for s in stats_list)

        fname = os.path.join(epoch_dir, "relatorio.txt")
        with open(fname, "w") as f:
            f.write(f"{'='*60}\n")
            f.write(f"RELATÓRIO DE FEATURE MAPS — Época {epoch+1}\n")
            f.write(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"RESUMO:\n")
            f.write(f"  Camadas monitoradas: {len(stats_list)}\n")
            f.write(f"  Total de mapas:      {total_maps}\n")
            f.write(f"  Mapas mortos:        {total_mortos} ({100*total_mortos/max(total_maps,1):.1f}%)\n\n")
            f.write(f"{'='*60}\n")
            f.write(f"DETALHES POR CAMADA:\n")
            f.write(f"{'='*60}\n\n")

            for s in stats_list:
                pct  = 100 * s["mortos"] / max(s["n_maps"], 1)
                flag = " ⚠ ALTO" if pct > 50 else (" ✓ OK" if pct < 10 else "")
                f.write(f"Camada: {s['nome']}\n")
                f.write(f"  Shape:          {s['shape']}\n")
                f.write(f"  Total mapas:    {s['n_maps']}\n")
                f.write(f"  Mortos:         {s['mortos']} ({pct:.1f}%){flag}\n")
                f.write(f"  Média ativ.:    {s['media']:.6f}\n")
                f.write(f"  Std ativ.:      {s['std']:.6f}\n")
                f.write(f"  Min / Max:      {s['min']:.6f} / {s['max']:.6f}\n")
                f.write(f"  % valores ~0:   {s['pct_zero']:.1f}%\n\n")

    # ------------------------------------------------------------------
    # HOOK PRINCIPAL
    # ------------------------------------------------------------------

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_every != 0:
            return

        epoch_dir = os.path.join(self.output_dir, f"epoca_{epoch+1:03d}")
        os.makedirs(epoch_dir, exist_ok=True)

        # Roda a imagem de referência pelo modelo de ativação
        model_inputs = self._prepare_model_inputs()
        activations = self._activation_model(model_inputs, training=False)

        # Se só há uma camada, tf retorna tensor em vez de lista
        if not isinstance(activations, (list, tuple)):
            activations = [activations]

        activations = [a.numpy() for a in activations]

        print(f"\n[FeatureMapCallback] Época {epoch+1} — salvando feature maps...")

        stats_list = []
        for layer_name, activation in zip(self._layer_names, activations):
            self._save_feature_map_grid(activation, layer_name, epoch_dir, epoch)
            stats_list.append(self._save_activation_stats(activation, layer_name))

        self._save_report(epoch_dir, epoch, stats_list)

        total_m = sum(s["mortos"] for s in stats_list)
        total_f = sum(s["n_maps"] for s in stats_list)
        print(f"[FeatureMapCallback] Salvo em '{epoch_dir}'")
        print(f"[FeatureMapCallback] Mapas mortos: {total_m}/{total_f} "
              f"({100*total_m/max(total_f,1):.1f}%)\n")


# ==============================================================================
# EXEMPLO DE USO
# ==============================================================================

if __name__ == "__main__":

    # Use uma imagem real do seu dataset como referência
    # Aqui simulamos uma imagem de ultrassom 256×256 grayscale
    sample = np.random.rand(256, 256, 1).astype(np.float32)

    # Todas as camadas, toda época
    cb = FeatureMapCallback(
        sample_image=sample,
        output_dir="feature_maps",
        save_every=1,
        max_maps=64,
        dead_threshold=1e-4,
        cmap="viridis",       # ou "gray" para escala de cinza
    )

    # Só camadas específicas, a cada 5 épocas
    cb_seletivo = FeatureMapCallback(
        sample_image=sample,
        output_dir="feature_maps_seletivo",
        layers_watch=["enc1_conv1", "enc2_conv1", "bottleneck_conv1"],
        save_every=5,
        max_maps=32,
        cmap="plasma",
    )
