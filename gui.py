#!/usr/bin/env python3
"""
Interface gráfica para detectar_texto.py.

  - Identificar: desenha bounding boxes à volta do texto.
  - Remover: apaga o texto sem IA generativa (pixels reais de outros frames;
    só o fundo nunca visível é preenchido pelo modo escolhido).

Uso:  .venv/bin/python gui.py
"""

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk
import cv2
from PIL import Image

AQUI = Path(__file__).resolve().parent
SCRIPT = AQUI / "detectar_texto.py"

# ----------------------------------------------------------------------------- estilo (claro, escuro)
ACENTO = ("#4F46E5", "#6366F1")
ACENTO_HOVER = ("#4338CA", "#4F46E5")
FUNDO = ("#F4F5F7", "#111214")
CARTAO = ("#FFFFFF", "#1B1C1F")
BORDA = ("#E4E5E9", "#2A2B30")
TEXTO = ("#111827", "#ECEDEE")
SUAVE = ("#6B7280", "#9A9CA3")
OK = ("#059669", "#34D399")
ERRO = ("#DC2626", "#F87171")
SEGMENTO = ("#ECEDF0", "#232428")
FAMILIA = "Ubuntu Sans"
PREV_W, PREV_H = 320, 520

PREENCHIMENTOS = {
    "Suave": ("suave", "Interpolação suave a partir das cores da borda. Recomendado."),
    "Preto": ("preencher", "Não inventa nenhum pixel: a zona fica a preto para o detetor a ignorar."),
    "Inpaint": ("inpaint", "Inpainting clássico do OpenCV (Telea). Pode deixar facetas."),
    "Desfoque": ("borrar", "Só esconde o texto; pode deixar um 'fantasma'."),
}


def abrir_no_sistema(caminho):
    caminho = str(caminho)
    if sys.platform.startswith("win"):
        os.startfile(caminho)  # noqa
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", caminho],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def ler_frame(video, fracao=0.5):
    """Devolve (PIL.Image, info) de um frame a meio do vídeo, ou (None, None)."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None, None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    info = {"w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": fps, "frames": total, "dur": total / fps}
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * fracao))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None, info
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), info


def caber(img, w, h):
    r = min(w / img.width, h / img.height)
    return ctk.CTkImage(light_image=img, dark_image=img, size=(int(img.width * r), int(img.height * r)))


def fmt_tempo(s):
    s = int(s)
    return f"{s // 60}:{s % 60:02d}"


class App(ctk.CTk):
    def __init__(self):
        ctk.set_appearance_mode("system")
        super().__init__(fg_color=FUNDO)
        self.title("Identificador de texto")
        self.geometry("1040x740")
        self.minsize(960, 700)

        self.f = lambda tam, peso="normal": ctk.CTkFont(family=FAMILIA, size=tam, weight=peso)
        self.proc = None
        self.fila = queue.Queue()
        self.t0 = None
        self.saida_atual = self.json_atual = self.mascara_atual = None
        self.img_antes = self.img_depois = None

        self.v_video = ctk.StringVar()
        self.v_saida = ctk.StringVar()
        self.v_modo = ctk.StringVar(value="Identificar")
        self.v_min_conf = ctk.DoubleVar(value=0.5)
        self.v_a_cada = ctk.IntVar(value=1)
        self.v_so_detecao = ctk.BooleanVar(value=False)
        self.v_preench = ctk.StringVar(value="Suave")
        self.v_mascara = ctk.StringVar(value="Letras")
        self.v_sem_perdas = ctk.BooleanVar(value=True)
        self.v_salvar_mascara = ctk.BooleanVar(value=True)
        self.v_margem = ctk.IntVar(value=3)
        self.v_janela = ctk.IntVar(value=30)
        self.v_limiar = ctk.DoubleVar(value=6.0)
        self.v_persistir = ctk.IntVar(value=3)
        self.v_mostrar = ctk.BooleanVar(value=False)
        self.v_vista = ctk.StringVar(value="Antes")
        self.av_aberto = False

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._coluna_previa()
        self._coluna_controlos()
        self._mudou_modo()
        self.after(100, self._ler_fila)
        self.protocol("WM_DELETE_WINDOW", self._fechar)

    # ------------------------------------------------------------------ helpers de UI
    def _cartao(self, pai, **kw):
        return ctk.CTkFrame(pai, fg_color=CARTAO, corner_radius=14, border_width=1, border_color=BORDA, **kw)

    def _rotulo(self, pai, texto, tam=13, peso="normal", cor=TEXTO, **kw):
        kw.setdefault("anchor", "w")
        return ctk.CTkLabel(pai, text=texto, font=self.f(tam, peso), text_color=cor, **kw)

    def _segmentado(self, pai, valores, var, comando=None):
        return ctk.CTkSegmentedButton(
            pai, values=valores, variable=var, command=comando, font=self.f(13),
            selected_color=ACENTO, selected_hover_color=ACENTO_HOVER, height=32, corner_radius=8,
            fg_color=SEGMENTO, unselected_color=SEGMENTO, unselected_hover_color=BORDA, text_color=TEXTO)

    def _interruptor(self, pai, texto, var):
        return ctk.CTkSwitch(pai, text=texto, variable=var, font=self.f(13), text_color=TEXTO,
                             progress_color=ACENTO, switch_width=36, switch_height=18)

    def _botao_secundario(self, pai, texto, comando, **kw):
        return ctk.CTkButton(pai, text=texto, command=comando, font=self.f(12), corner_radius=8,
                             fg_color="transparent", border_width=1, border_color=BORDA,
                             text_color=TEXTO, hover_color=BORDA, **kw)

    def _slider(self, pai, titulo, var, de, ate, passos, fmt):
        linha = ctk.CTkFrame(pai, fg_color="transparent")
        topo = ctk.CTkFrame(linha, fg_color="transparent")
        topo.pack(fill="x")
        self._rotulo(topo, titulo).pack(side="left")
        valor = self._rotulo(topo, "", cor=SUAVE)
        valor.pack(side="right")

        def mexeu(_=None):
            valor.configure(text=fmt(var.get()))
        ctk.CTkSlider(linha, from_=de, to=ate, number_of_steps=passos, variable=var, command=mexeu,
                      button_color=ACENTO, button_hover_color=ACENTO_HOVER, progress_color=ACENTO,
                      height=16).pack(fill="x", pady=(6, 0))
        mexeu()
        return linha

    # ------------------------------------------------------------------ coluna esquerda: pré-visualização
    def _coluna_previa(self):
        col = ctk.CTkFrame(self, fg_color="transparent")
        col.grid(row=0, column=0, sticky="ns", padx=(24, 12), pady=24)

        self.c_previa = self._cartao(col, width=PREV_W + 24, height=PREV_H + 24)
        self.c_previa.pack()
        self.c_previa.pack_propagate(False)
        self.l_previa = ctk.CTkLabel(
            self.c_previa, text="Clique para escolher um vídeo\n\nMP4 · MOV · MKV · AVI · WEBM",
            font=self.f(14), text_color=SUAVE, cursor="hand2")
        self.l_previa.pack(expand=True, fill="both", padx=12, pady=12)
        self.l_previa.bind("<Button-1>", lambda _: self._escolher_video())

        self.l_info = self._rotulo(col, "", 12, cor=SUAVE, anchor="center", justify="center")
        self.l_info.pack(fill="x", pady=(10, 8))
        self.seg_vista = self._segmentado(col, ["Antes", "Depois"], self.v_vista, lambda _: self._mostrar_vista())


    # ------------------------------------------------------------------ coluna direita: controlos
    def _coluna_controlos(self):
        col = ctk.CTkFrame(self, fg_color="transparent")
        col.grid(row=0, column=1, sticky="nsew", padx=(12, 24), pady=24)
        col.grid_columnconfigure(0, weight=1)
        col.grid_rowconfigure(2, weight=1)

        cab = ctk.CTkFrame(col, fg_color="transparent")
        cab.grid(row=0, column=0, sticky="ew")
        self._rotulo(cab, "Identificador de texto", 24, "bold").pack(anchor="w")
        self._rotulo(cab, "Deteta ou remove texto em vídeo — sem IA generativa.", 13, cor=SUAVE).pack(anchor="w")

        self.seg_modo = self._segmentado(col, ["Identificar", "Remover"], self.v_modo, lambda _: self._mudou_modo())
        self.seg_modo.grid(row=1, column=0, sticky="ew", pady=(18, 14))

        self.opcoes = ctk.CTkScrollableFrame(col, fg_color="transparent", scrollbar_button_color=BORDA)
        self.opcoes.grid(row=2, column=0, sticky="nsew")
        self.opcoes.grid_columnconfigure(0, weight=1)
        self._opcoes_identificar()
        self._opcoes_remover()
        self._opcoes_avancadas()

        # Rodapé: saída, executar, resultados
        rod = self._cartao(col)
        rod.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        rod.grid_columnconfigure(0, weight=1)

        s = ctk.CTkFrame(rod, fg_color="transparent")
        s.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 0))
        s.grid_columnconfigure(1, weight=1)
        self._rotulo(s, "Guardar em", 12, cor=SUAVE).grid(row=0, column=0, padx=(0, 10))
        ctk.CTkEntry(s, textvariable=self.v_saida, font=self.f(12), height=30, border_color=BORDA,
                     corner_radius=8).grid(row=0, column=1, sticky="ew")
        self._botao_secundario(s, "Alterar", self._escolher_saida, width=70, height=30).grid(
            row=0, column=2, padx=(8, 0))

        e = ctk.CTkFrame(rod, fg_color="transparent")
        e.grid(row=1, column=0, sticky="ew", padx=16, pady=14)
        e.grid_columnconfigure(0, weight=1)
        self.barra = ctk.CTkProgressBar(e, progress_color=ACENTO, fg_color=BORDA, height=6, corner_radius=3)
        self.barra.set(0)
        self.barra.grid(row=0, column=0, sticky="ew", padx=(0, 14))
        self.l_estado = self._rotulo(e, "Escolha um vídeo para começar", 12, cor=SUAVE)
        self.l_estado.grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.b_iniciar = ctk.CTkButton(e, text="Iniciar", width=120, height=40, font=self.f(14, "bold"),
                                       fg_color=ACENTO, hover_color=ACENTO_HOVER, corner_radius=10,
                                       command=self._iniciar_ou_cancelar)
        self.b_iniciar.grid(row=0, column=1, rowspan=2)

        # Resultados (aparecem no fim)
        self.c_res = ctk.CTkFrame(rod, fg_color="transparent")
        self.c_res.grid_columnconfigure((0, 1, 2), weight=1, uniform="r")
        self.estat = []
        for i in range(3):
            c = ctk.CTkFrame(self.c_res, fg_color=FUNDO, corner_radius=10)
            c.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 8, 0))
            v = self._rotulo(c, "—", 20, "bold", anchor="center")
            v.pack(fill="x", pady=(10, 0))
            r = self._rotulo(c, "", 11, cor=SUAVE, anchor="center")
            r.pack(fill="x", pady=(0, 10))
            self.estat.append((v, r))
        acoes = ctk.CTkFrame(self.c_res, fg_color="transparent")
        acoes.grid(row=1, column=0, columnspan=3, sticky="w", pady=(10, 0))
        self.b_acoes = {}
        for nome, fn in (("Abrir vídeo", lambda: abrir_no_sistema(self.saida_atual)),
                         ("Máscara", lambda: abrir_no_sistema(self.mascara_atual)),
                         ("JSON", lambda: abrir_no_sistema(self.json_atual)),
                         ("Pasta", lambda: abrir_no_sistema(Path(self.saida_atual).parent)),
                         ("Registo", self._alternar_registo)):
            b = self._botao_secundario(acoes, nome, fn, height=28, width=0)
            b.pack(side="left", padx=(0, 6))
            self.b_acoes[nome] = b
        self.registo = ctk.CTkTextbox(rod, height=120, font=ctk.CTkFont(family="Ubuntu Mono", size=11),
                                      fg_color=FUNDO, corner_radius=8, wrap="word")
        self.registo.configure(state="disabled")

    def _opcoes_identificar(self):
        c = self.c_ident = self._cartao(self.opcoes)
        self._rotulo(c, "Deteção", 15, "bold").pack(anchor="w", padx=18, pady=(16, 10))
        self._slider(c, "Confiança mínima do OCR", self.v_min_conf, 0.1, 0.95, 17,
                     lambda v: f"{v:.2f}").pack(fill="x", padx=18, pady=(0, 14))
        self._slider(c, "Frames analisados", self.v_a_cada, 1, 10, 9,
                     lambda v: "todos" if int(v) == 1 else f"1 em cada {int(v)}").pack(fill="x", padx=18, pady=(0, 14))
        self._interruptor(c, "Só deteção (mais rápido, não lê o texto)", self.v_so_detecao).pack(
            anchor="w", padx=18, pady=(0, 16))

    def _opcoes_remover(self):
        c = self.c_rem = self._cartao(self.opcoes)
        self._rotulo(c, "Remoção", 15, "bold").pack(anchor="w", padx=18, pady=(16, 2))
        self._rotulo(c, "Usa primeiro pixels reais de outros frames onde o fundo está visível.", 12,
                     cor=SUAVE, wraplength=520, justify="left").pack(anchor="w", padx=18, pady=(0, 12))

        self._rotulo(c, "Fundo nunca visível", 13).pack(anchor="w", padx=18)
        self._segmentado(c, list(PREENCHIMENTOS), self.v_preench,
                         lambda _: self._desc_preench()).pack(fill="x", padx=18, pady=(6, 4))
        self.l_desc = self._rotulo(c, "", 12, cor=SUAVE, wraplength=520, justify="left")
        self.l_desc.pack(anchor="w", padx=18, pady=(0, 12))
        self._desc_preench()

        self._rotulo(c, "Máscara", 13).pack(anchor="w", padx=18)
        self._segmentado(c, ["Letras", "Caixa"], self.v_mascara).pack(fill="x", padx=18, pady=(6, 14))
        self._interruptor(c, "Saída sem perdas (recomendado para deepfakes)", self.v_sem_perdas).pack(
            anchor="w", padx=18, pady=(0, 8))
        self._interruptor(c, "Gravar vídeo da máscara", self.v_salvar_mascara).pack(anchor="w", padx=18, pady=(0, 16))

    def _opcoes_avancadas(self):
        self.c_av = self._cartao(self.opcoes)
        self.c_av.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        self.b_av = ctk.CTkButton(self.c_av, text="Avançado   ›", anchor="w", font=self.f(13), height=40,
                                  fg_color="transparent", hover_color=BORDA, text_color=TEXTO,
                                  corner_radius=12, command=self._alternar_avancado)
        self.b_av.pack(fill="x", padx=6, pady=4)
        self.av = ctk.CTkFrame(self.c_av, fg_color="transparent")
        self.av_rem = [
            self._slider(self.av, "Margem à volta das letras", self.v_margem, 0, 20, 20, lambda v: f"{int(v)} px"),
            self._slider(self.av, "Janela temporal", self.v_janela, 0, 120, 24, lambda v: f"±{int(v)} frames"),
            self._slider(self.av, "Limiar temporal (menor = mais rigoroso)", self.v_limiar, 1, 20, 38,
                         lambda v: f"{v:.1f}"),
        ]
        self.av_com = [
            self._slider(self.av, "Manter caixas quando o detetor falha", self.v_persistir, 0, 15, 15,
                         lambda v: f"{int(v)} frames"),
        ]
        self.sw_mostrar = self._interruptor(self.av, "Mostrar o vídeo enquanto processa", self.v_mostrar)

    def _arrumar_avancado(self):
        for w in self.av.winfo_children():
            w.pack_forget()
        for w in (self.av_rem if self.v_modo.get() == "Remover" else []) + self.av_com:
            w.pack(fill="x", padx=18, pady=(0, 14))
        self.sw_mostrar.pack(anchor="w", padx=18, pady=(0, 16))

    def _alternar_avancado(self):
        self.av_aberto = not self.av_aberto
        self.b_av.configure(text="Avançado   ⌄" if self.av_aberto else "Avançado   ›")
        if self.av_aberto:
            self._arrumar_avancado()
            self.av.pack(fill="x")
        else:
            self.av.pack_forget()

    def _alternar_registo(self):
        if self.registo.winfo_ismapped():
            self.registo.grid_forget()
        else:
            self.registo.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 14))

    def _desc_preench(self):
        self.l_desc.configure(text=PREENCHIMENTOS[self.v_preench.get()][1])

    def _mudou_modo(self):
        remover = self.v_modo.get() == "Remover"
        (self.c_ident if remover else self.c_rem).grid_forget()
        (self.c_rem if remover else self.c_ident).grid(row=0, column=0, sticky="ew")
        if self.av_aberto:
            self._arrumar_avancado()
        self._sugerir_saida()

    # ------------------------------------------------------------------ ficheiros e pré-visualização
    def _sugerir_saida(self):
        v = self.v_video.get()
        if v:
            p = Path(v)
            suf = "_sem_texto" if self.v_modo.get() == "Remover" else "_texto"
            self.v_saida.set(str(p.with_name(p.stem + suf + ".mp4")))

    def _escolher_video(self, caminho=None):
        if self.proc:
            return
        c = caminho or filedialog.askopenfilename(
            title="Escolher vídeo", initialdir=AQUI,
            filetypes=[("Vídeos", "*.mp4 *.mov *.mkv *.avi *.webm *.m4v"), ("Todos", "*.*")])
        if not c:
            return
        img, info = ler_frame(c)
        if img is None:
            self._estado("Não foi possível ler esse vídeo.", ERRO)
            return
        self.v_video.set(c)
        self._sugerir_saida()
        self.img_antes, self.img_depois = img, None
        self.v_vista.set("Antes")
        self.seg_vista.pack_forget()
        self._mostrar_vista()
        self.l_info.configure(text=f"{Path(c).name}\n{info['w']}×{info['h']} · {info['fps']:.0f} fps · "
                                   f"{fmt_tempo(info['dur'])}")
        self.c_res.grid_forget()
        self.registo.grid_forget()
        self.barra.set(0)
        self._estado("Pronto. Clique na imagem para trocar de vídeo.")

    def _mostrar_vista(self):
        img = self.img_depois if self.v_vista.get() == "Depois" and self.img_depois else self.img_antes
        if img is not None:
            self.l_previa.configure(image=caber(img, PREV_W, PREV_H), text="")

    def _escolher_saida(self):
        atual = Path(self.v_saida.get() or AQUI / "saida.mp4")
        c = filedialog.asksaveasfilename(title="Guardar vídeo como", initialdir=atual.parent,
                                         initialfile=atual.name, defaultextension=".mp4",
                                         filetypes=[("MP4", "*.mp4")])
        if c:
            self.v_saida.set(c)

    def _estado(self, texto, cor=SUAVE):
        self.l_estado.configure(text=texto, text_color=cor)

    def _escrever(self, texto):
        self.registo.configure(state="normal")
        self.registo.insert("end", texto)
        self.registo.see("end")
        self.registo.configure(state="disabled")

    # ------------------------------------------------------------------ execução
    def _comando(self, video, saida):
        cmd = [sys.executable, "-u", str(SCRIPT), str(video), "-o", str(saida),
               "--persistir", str(int(self.v_persistir.get()))]
        if self.v_modo.get() == "Remover":
            cmd += ["--remover", PREENCHIMENTOS[self.v_preench.get()][0],
                    "--mascara", self.v_mascara.get().lower(),
                    "--margem", str(int(self.v_margem.get())),
                    "--janela", str(int(self.v_janela.get())),
                    "--limiar-temporal", f"{self.v_limiar.get():.1f}", "--a-cada", "1"]
            if self.v_sem_perdas.get():
                cmd.append("--sem-perdas")
            if self.v_salvar_mascara.get():
                cmd.append("--salvar-mascara")
        else:
            cmd += ["--min-conf", f"{self.v_min_conf.get():.2f}", "--a-cada", str(int(self.v_a_cada.get()))]
            if self.v_so_detecao.get():
                cmd.append("--so-detecao")
        if self.v_mostrar.get():
            cmd.append("--mostrar")
        return cmd

    def _iniciar_ou_cancelar(self):
        if self.proc:
            self._cancelar()
        else:
            self._iniciar()

    def _iniciar(self):
        video = Path(self.v_video.get())
        saida = Path(self.v_saida.get().strip())
        if not video.is_file():
            self._escolher_video()
            return
        if not saida.name or saida.resolve() == video.resolve():
            self._estado("Indique um ficheiro de saída diferente do vídeo de entrada.", ERRO)
            return

        self.saida_atual = saida
        self.json_atual = saida.with_suffix(".json")
        self.mascara_atual = saida.with_name(saida.stem + "_mascara.mp4")
        self.c_res.grid_forget()
        self.registo.grid_forget()
        self.registo.configure(state="normal")
        self.registo.delete("1.0", "end")
        self.registo.configure(state="disabled")
        cmd = self._comando(video, saida)
        self._escrever("$ " + " ".join(cmd[2:]) + "\n\n")
        self.barra.set(0)
        self._estado("A carregar o modelo…")
        self.b_iniciar.configure(text="Cancelar", fg_color=BORDA, text_color=TEXTO, hover_color=SUAVE)
        self.seg_modo.configure(state="disabled")
        self.t0 = None
        self.proc = subprocess.Popen(cmd, cwd=AQUI, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     env={**os.environ, "PYTHONUNBUFFERED": "1"})
        threading.Thread(target=self._ler_processo, args=(self.proc,), daemon=True).start()

    def _ler_processo(self, proc):
        """Lê a saída do script; separa o progresso ('\\rN/T frames') do resto."""
        pedaco = b""
        while True:
            c = proc.stdout.read(1)
            if c in (b"\r", b"\n", b""):
                linha = re.sub(r"\x1b\[[0-9;]*m", "", pedaco.decode("utf-8", "replace"))
                pedaco = b""
                m = re.fullmatch(r"\s*(\d+)/(-?\d+) frames\s*", linha)
                if m:
                    self.fila.put(("progresso", int(m.group(1)), int(m.group(2))))
                elif linha.strip() and not ("[RapidOCR]" in linha and "[INFO]" in linha):
                    self.fila.put(("log", linha + "\n"))
                if not c:
                    break
            else:
                pedaco += c
        self.fila.put(("fim", proc.wait(), proc))

    def _ler_fila(self):
        try:
            while True:
                msg = self.fila.get_nowait()
                if msg[0] == "log":
                    self._escrever(msg[1])
                elif msg[0] == "progresso":
                    self._progresso(msg[1], msg[2])
                elif msg[0] == "fim" and msg[2] is self.proc:  # ignora processos já cancelados
                    self._terminou(msg[1])
        except queue.Empty:
            pass
        self.after(100, self._ler_fila)

    def _progresso(self, n, total):
        agora = time.monotonic()
        if self.t0 is None:
            self.t0 = (agora, n)
        if total <= 0:
            self._estado(f"{n} frames processados")
            return
        self.barra.set(min(n / total, 1))
        t, n0 = self.t0
        texto = f"{n} de {total} frames · {100 * n // total}%"
        if n > n0 and agora > t:
            texto += f" · falta ~{fmt_tempo((total - n) * (agora - t) / (n - n0))}"
        self._estado(texto)

    def _repor_botao(self):
        self.proc = None
        self.b_iniciar.configure(text="Iniciar", fg_color=ACENTO, hover_color=ACENTO_HOVER, text_color="#FFFFFF")
        self.seg_modo.configure(state="normal")

    def _terminou(self, codigo):
        self._repor_botao()
        if codigo != 0:
            self.barra.set(0)
            self._estado(f"O processamento falhou (código {codigo}). Veja o registo.", ERRO)
            self._mostrar_resultados(None)
            self._alternar_registo()
            return
        self.barra.set(1)
        self._estado("Concluído", OK)
        try:
            d = json.loads(Path(self.json_atual).read_text())
        except (OSError, ValueError):
            d = None
        self._mostrar_resultados(d)
        img, _ = ler_frame(self.saida_atual)
        if img is not None:
            self.img_depois = img
            self.seg_vista.pack()
            self.v_vista.set("Depois")
            self._mostrar_vista()

    def _mostrar_resultados(self, d):
        self.c_res.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 14))
        if d:
            frames = d.get("frames_com_texto", [])
            textos = {det.get("texto", "").strip() for f in frames for det in f.get("deteccoes", [])} - {""}
            dados = [(f"{len(frames)}", f"de {d.get('frames_processados')} frames com texto")]
            rem = d.get("remocao")
            if rem:
                dados += [(f"{rem['pct_pixels_reais_de_outros_frames']:.0f}%", "pixels reais de outros frames"),
                          (f"{rem['pct_pixels_fallback']:.0f}%", f"preenchidos ({self.v_preench.get().lower()})")]
            else:
                dados += [(f"{len(textos)}", "textos diferentes lidos"),
                          (f"{sum(len(f['deteccoes']) for f in frames)}", "caixas desenhadas")]
        else:
            dados = [("—", "")] * 3
        for (v, r), (a, b) in zip(self.estat, dados):
            v.configure(text=a)
            r.configure(text=b)

        def existe(p):
            return bool(p) and Path(p).exists()
        for nome, p in (("Abrir vídeo", self.saida_atual), ("Pasta", self.saida_atual), ("JSON", self.json_atual)):
            self.b_acoes[nome].configure(state="normal" if existe(p) else "disabled")
        if self.v_modo.get() == "Remover" and existe(self.mascara_atual):
            self.b_acoes["Máscara"].pack(side="left", padx=(0, 6), after=self.b_acoes["Abrir vídeo"])
        else:
            self.b_acoes["Máscara"].pack_forget()

    def _cancelar(self):
        proc = self.proc
        if not proc:
            return
        self._repor_botao()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        if self.saida_atual:
            Path(self.saida_atual).with_name(".tmp_" + Path(self.saida_atual).name).unlink(missing_ok=True)
        self.barra.set(0)
        self._estado("Cancelado — o vídeo de saída pode estar incompleto.", ERRO)

    def _fechar(self):
        self._cancelar()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
