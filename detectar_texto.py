#!/usr/bin/env python3
"""
Deteta texto num vídeo e desenha bounding boxes — ou remove o texto (pré-processamento).

Pipeline (PP-OCR via RapidOCR / ONNX Runtime, corre em CPU):
  1. Deteção  (modelo DBNet)  -> encontra as regiões do frame que contêm texto
  2. Reconhecimento (SVTR/CRNN) -> lê o texto e dá uma confiança
  3. Só se mantêm as caixas cujo texto foi reconhecido com confiança >= --min-conf
  4. (opcional, --remover) apaga o texto SEM pixels sintéticos de IA:
       a. máscara justa às letras (cor do texto vs. fundo à volta), não à caixa inteira
       b. preenche com pixels REAIS de frames vizinhos onde esse fundo está visível
          (alinhados pelo movimento da câmara e validados à volta de cada zona)
       c. só o que nunca aparece noutro frame vai para o modo escolhido (suave/inpaint/preencher/borrar)

Saídas:
  - vídeo anotado com as bounding boxes, ou vídeo com o texto removido
  - ficheiro JSON com, para cada frame analisado, o tempo, as caixas e o texto
  - (opcional, --salvar-mascara) vídeo da máscara: cinzento = pixels reais de outro frame,
    branco = pixels interpolados (fallback), preto = original intacto

Exemplos:
  python detectar_texto.py video.mp4
  python detectar_texto.py video.mp4 --a-cada 5 --min-conf 0.6
  python detectar_texto.py video.mp4 --remover suave --sem-perdas --salvar-mascara
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from rapidocr import RapidOCR

ORIGINAL, TEMPORAL, FALLBACK = 0, 128, 255


class EscritorVideo:
    """Grava frames BGR. Usa ffmpeg (alta qualidade ou sem perdas) se existir; senão OpenCV."""

    def __init__(self, caminho, fps, largura, altura, sem_perdas=False):
        self.proc = None
        self.cv = None
        if shutil.which("ffmpeg"):
            cmd = ["ffmpeg", "-loglevel", "error", "-y",
                   "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{largura}x{altura}",
                   "-r", str(fps), "-i", "-"]
            if sem_perdas:
                # RGB sem perdas: pixels idênticos aos processados (ficheiro grande)
                cmd += ["-c:v", "libx264rgb", "-qp", "0", "-preset", "veryfast"]
            else:
                cmd += ["-c:v", "libx264", "-crf", "18", "-preset", "medium",
                        "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]
            cmd.append(str(caminho))
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            print("Aviso: ffmpeg não encontrado, a usar OpenCV (mp4v, qualidade inferior).")
            self.cv = cv2.VideoWriter(str(caminho), cv2.VideoWriter_fourcc(*"mp4v"),
                                      fps, (largura, altura))

    def write(self, frame):
        if self.proc:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            self.cv.write(frame)

    def release(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
        else:
            self.cv.release()


def tem_audio(video):
    if not shutil.which("ffprobe"):
        return False
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                        "stream=index", "-of", "csv=p=0", str(video)], capture_output=True, text=True)
    return bool(r.stdout.strip())


def juntar_audio(video_sem_audio, original, saida):
    """Junta o áudio do original ao vídeo processado. Copia o áudio sem recodificar; se o codec
    não couber no contentor de saída, converte para AAC. Devolve uma descrição do que foi feito."""
    base = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(video_sem_audio), "-i", str(original),
            "-map", "0:v:0", "-map", "1:a?", "-c:v", "copy"]
    for codec_audio, descricao in ((["-c:a", "copy"], "copiado sem alterações"),
                                   (["-c:a", "aac", "-b:a", "320k"], "convertido para AAC 320k")):
        r = subprocess.run(base + codec_audio + [str(saida)], capture_output=True, text=True)
        if r.returncode == 0:
            return descricao
    raise RuntimeError(f"Falha ao juntar o áudio: {r.stderr.strip()}")


def analisar_frame(motor, frame, so_detecao, min_conf):
    """Devolve uma lista de dicts {caixa, poligono, texto, confianca} para um frame."""
    if so_detecao:
        res = motor(frame, use_rec=False, use_cls=False)
        textos = [None] * (0 if res.boxes is None else len(res.boxes))
    else:
        res = motor(frame)
        textos = res.txts or []

    if res.boxes is None:
        return []

    deteccoes = []
    for i, poli in enumerate(res.boxes):
        conf = float(res.scores[i]) if res.scores is not None else 1.0
        texto = textos[i] if i < len(textos) else None
        if not so_detecao and (conf < min_conf or not (texto or "").strip()):
            continue
        poli = np.asarray(poli, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(poli)
        deteccoes.append({
            "caixa": [int(x), int(y), int(w), int(h)],  # x, y, largura, altura
            "poligono": poli.tolist(),                  # 4 cantos (texto rodado)
            "texto": texto,
            "confianca": round(conf, 4),
        })
    return deteccoes


# ----------------------------------------------------------------------------- máscara

def _elipse(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def preencher_suave(img, mascara, iteracoes=40):
    """Interpolação harmónica: cada pixel em falta fica a média dos vizinhos (equação de Laplace),
    resolvida com uma pirâmide push-pull + iterações de Jacobi. Transição suave das cores da borda,
    sem facetas nem "riscos" como o Telea, e sem inventar textura (nada de IA)."""
    m = mascara > 0
    if not m.any():
        return img
    f = img.astype(np.float32)
    w = (~m).astype(np.float32)
    niveis = [(f * w[..., None], w)]
    while min(niveis[-1][1].shape) > 8:
        a, ww = niveis[-1]
        niveis.append((cv2.pyrDown(a), cv2.pyrDown(ww)))
    a, ww = niveis[-1]
    est = a / np.maximum(ww, 1e-6)[..., None]
    for a, ww in reversed(niveis[:-1]):
        cima = cv2.pyrUp(est, dstsize=(ww.shape[1], ww.shape[0]))
        c = np.clip(ww * 2, 0, 1)[..., None]
        est = c * (a / np.maximum(ww, 1e-6)[..., None]) + (1 - c) * cima
    est = np.ascontiguousarray(est, dtype=np.float32)
    m8 = m.astype(np.uint8)
    est = cv2.copyTo(est, m8, f.copy())
    k = np.array([[0, .25, 0], [.25, 0, .25], [0, .25, 0]], np.float32)
    for _ in range(iteracoes):
        est = cv2.copyTo(cv2.filter2D(est, -1, k), m8, est)
    return np.clip(est + 0.5, 0, 255).astype(np.uint8)


def _cresce_contorno(lab, letras, alt, tol=20.0):
    """Muitos textos têm contorno/sombra de cor uniforme (ex.: preto) colado às letras. Essa cor
    pode existir também no fundo (barba, roupa), por isso não sai do histograma: aqui cresce-se a
    partir das letras só por pixels com a cor dominante da faixa imediatamente à volta delas."""
    faixa = cv2.dilate(letras, _elipse(2)) & ~letras
    cores = lab[faixa > 0]
    if len(cores) < 50:
        return letras
    ref = np.median(cores, axis=0)
    perto = np.linalg.norm(lab - ref, axis=2) < tol
    if perto[faixa > 0].mean() < 0.5:      # não há um contorno uniforme
        return letras
    R = max(3, int(round(0.15 * alt)))
    permitido = ((perto & (cv2.dilate(letras, _elipse(R)) > 0)).astype(np.uint8) * 255) | letras
    atual = letras
    for _ in range(R):                     # crescimento ligado às letras, 1 px de cada vez
        seguinte = cv2.dilate(atual, np.ones((3, 3), np.uint8)) & permitido
        if np.array_equal(seguinte, atual):
            break
        atual = seguinte
    return atual


def criar_mascara(frame, deteccoes, margem, tipo="letras", bins=32, razao=4.0, limiar_residuo=14):
    """Máscara uint8 (255 = texto).

    tipo="caixa":  o polígono inteiro de cada deteção, alargado `margem` px.
    tipo="letras": só os pixels das letras. As cores do texto são as que aparecem muito mais
                   dentro das caixas do que no fundo à volta (histograma Lab). Depois junta-se o
                   contorno/sombra das letras e alarga-se um pouco para o anti-aliasing.
                   Se numa caixa quase nada bater com a cor do texto, usa-se a caixa inteira.
    """
    h, w = frame.shape[:2]
    mascara = np.zeros((h, w), np.uint8)
    if not deteccoes:
        return mascara
    caixas = []
    todas = np.zeros((h, w), np.uint8)
    for d in deteccoes:
        c = np.zeros((h, w), np.uint8)
        cv2.fillPoly(c, [np.array(d["poligono"], np.int32)], 255)
        caixas.append(c)
        todas |= c

    if tipo == "caixa":
        return cv2.dilate(todas, _elipse(margem)) if margem > 0 else todas

    # trabalha só no recorte à volta do texto (mais rápido)
    alt_med = int(np.median([d["caixa"][3] for d in deteccoes]))
    largura_anel = max(16, alt_med)
    ys, xs = np.nonzero(todas)
    p = largura_anel + 2 * margem + alt_med // 4 + 8
    y0, y1 = max(ys.min() - p, 0), min(ys.max() + p + 1, h)
    x0, x1 = max(xs.min() - p, 0), min(xs.max() + p + 1, w)
    rec = frame[y0:y1, x0:x1]
    todas_r = todas[y0:y1, x0:x1]

    lab_u8 = cv2.cvtColor(cv2.GaussianBlur(rec, (3, 3), 0), cv2.COLOR_BGR2Lab)
    q = (lab_u8 // (256 // bins)).astype(np.int32)
    idx = (q[..., 0] * bins + q[..., 1]) * bins + q[..., 2]
    anel = (cv2.dilate(todas_r, np.ones((2 * largura_anel + 1,) * 2, np.uint8))
            & ~cv2.dilate(todas_r, np.ones((9, 9), np.uint8)))
    h_dentro = np.bincount(idx[todas_r > 0], minlength=bins ** 3).astype(np.float32)
    h_dentro /= h_dentro.sum()
    h_fora = np.bincount(idx[anel > 0], minlength=bins ** 3).astype(np.float32)
    h_fora /= max(h_fora.sum(), 1.0)
    cor_texto = (h_dentro > razao * h_fora) & (h_dentro > 2e-4)
    candidatos = (cor_texto[idx] & (todas_r > 0)).astype(np.uint8) * 255
    candidatos = cv2.morphologyEx(candidatos, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    letras = np.zeros_like(todas_r)
    caixa_inteira = np.zeros_like(todas_r)
    for d, c in zip(deteccoes, caixas):
        c = c[y0:y1, x0:x1]
        m = candidatos & c
        if (m > 0).sum() < 0.05 * (c > 0).sum():
            caixa_inteira |= c
        else:
            letras |= m
    if letras.any():
        letras = _cresce_contorno(lab_u8.astype(np.float32), letras, alt_med)
        # sombra semitransparente / halo: perto das letras, o que difere muito do fundo estimado
        # (2 passagens: a 2.ª apanha a borda anti-aliased que a 1.ª deixou à volta da sombra)
        cinza_rec = cv2.cvtColor(rec, cv2.COLOR_BGR2GRAY)
        R = max(3, int(round(0.12 * alt_med)))
        zona = cv2.dilate(todas_r, _elipse(2 * R)) > 0
        for _ in range(2):
            est = preencher_suave(rec, cv2.dilate(letras, _elipse(2)), iteracoes=0)
            resid = cv2.absdiff(cinza_rec, cv2.cvtColor(est, cv2.COLOR_BGR2GRAY))
            perto = (cv2.dilate(letras, _elipse(R)) > 0) & zona
            sombra = ((resid > limiar_residuo) & perto).astype(np.uint8) * 255
            letras |= cv2.morphologyEx(sombra, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        r = margem + int(round(0.03 * alt_med))
        letras = cv2.dilate(letras, _elipse(r)) if r > 0 else letras
        letras = cv2.morphologyEx(letras, cv2.MORPH_CLOSE, _elipse(max(2, r)))
        # buracos pequenos cercados pela máscara (sombra entre letras, miolo de "o", "a"...)
        _, rot, est_cc, _ = cv2.connectedComponentsWithStats((letras == 0).astype(np.uint8), connectivity=4)
        hh, ww = letras.shape
        x, y, bw, bh, area = est_cc.T
        encher = (x > 0) & (y > 0) & (x + bw < ww) & (y + bh < hh) & (area < 0.5 * alt_med ** 2)
        encher[0] = False
        letras[encher[rot]] = 255
    if caixa_inteira.any():
        letras |= cv2.dilate(caixa_inteira, _elipse(margem)) if margem > 0 else caixa_inteira
    mascara[y0:y1, x0:x1] = letras
    return mascara


# ------------------------------------------------------------- preenchimento temporal

class PreenchedorTemporal:
    """Preenche a máscara do frame t com pixels reais de frames vizinhos.

    Para cada vizinho s: estima a homografia s->t com pontos ORB fora do texto (movimento da
    câmara), projeta s em t, e aceita os pixels que em s não estavam tapados por texto. A
    aceitação é feita por blocos: cada bloco só é copiado se, nos pixels visíveis em ambos os
    frames à volta dele, s alinhado for quase igual a t (diferença média < limiar). Isto rejeita
    zonas onde algo se mexeu (rosto, mãos) e o fundo de s já não corresponde ao de t.
    Câmara parada -> cópia direta, sem reamostragem: os pixels são exatamente os do outro frame.
    """

    def __init__(self, janela=30, limiar=6.0, bloco=32):
        self.janela = janela
        self.limiar = limiar
        self.bloco = bloco
        self.orb = cv2.ORB_create(1500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # vizinhos por ordem de distância temporal (mais perto = fundo mais parecido)
        passos = sorted({k for k in (1, 2, 3, 4, 6, 8, 11, 15, 20, 25, 30, 40, 50, 60) if k <= janela}
                        | {janela})
        self.deslocamentos = [s * k for k in passos for s in (1, -1)]

    def caracteristicas(self, frame, mascara):
        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pequeno = cv2.resize(cinza, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        fora = cv2.resize(cv2.dilate(mascara, np.ones((15, 15), np.uint8)), pequeno.shape[::-1],
                          interpolation=cv2.INTER_NEAREST) == 0
        kp, des = self.orb.detectAndCompute(pequeno, fora.astype(np.uint8) * 255)
        pts = np.float32([k.pt for k in kp]) * 2 if kp else np.zeros((0, 2), np.float32)
        return {"cinza": cinza, "pts": pts, "des": des}

    def homografia_bruta(self, ft, fs):
        """Homografia 3x3 que leva coordenadas de s para t, ou None se falhar."""
        if ft["des"] is None or fs["des"] is None or len(ft["pts"]) < 12 or len(fs["pts"]) < 12:
            return None
        m = self.bf.match(ft["des"], fs["des"])
        if len(m) < 12:
            return None
        dst = ft["pts"][[x.queryIdx for x in m]]
        src = fs["pts"][[x.trainIdx for x in m]]
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
        if H is None or inl.sum() < 12:
            return None
        return H

    def homografia(self, t, s, buf):
        """s->t composta das homografias entre frames consecutivos (1 estimativa por frame em vez
        de uma por par). 'I' se a câmara está parada; None se falhar."""
        H = np.eye(3)
        passos = range(s, t, -1) if s > t else range(t, s, -1)
        for j in passos:
            A = buf[j].get("A")            # j -> j-1
            if A is None:                  # corte de cena / frame sem pontos: não usar
                return None
            H = A @ H
        if s < t:
            H = np.linalg.inv(H)
        h, w = buf[t]["cinza"].shape
        cantos = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        if np.abs(cv2.perspectiveTransform(cantos, H) - cantos).max() < 0.75:
            return "I"
        return H

    def preencher(self, t, buf):
        """Devolve (frame, fonte). fonte: ORIGINAL / TEMPORAL / (por preencher = FALLBACK)."""
        e = buf[t]
        frame, mascara = e["frame"].copy(), e["mascara"]
        fonte = np.zeros(mascara.shape, np.uint8)
        buraco = mascara > 0
        if not buraco.any():
            return frame, fonte
        fonte[buraco] = FALLBACK
        H_img, W_img = mascara.shape
        b = self.bloco
        ys, xs = np.nonzero(buraco)
        # recorte de trabalho: zona do texto + contexto, alinhado à grelha de blocos
        x0, y0 = max(xs.min() - 2 * b, 0), max(ys.min() - 2 * b, 0)
        x1, y1 = min(xs.max() + 2 * b + 1, W_img), min(ys.max() + 2 * b + 1, H_img)
        bw, bh = x1 - x0, y1 - y0
        nbx, nby = -(-bw // b), -(-bh // b)
        pad = ((0, nby * b - bh), (0, nbx * b - bw))

        buraco_c = buraco[y0:y1, x0:x1].copy()
        conhecido_t = ~buraco_c                      # pixels originais de t (sem texto)
        cinza_t = e["cinza"][y0:y1, x0:x1].astype(np.int16)
        T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)

        def por_bloco(a):  # soma por bloco b x b
            return np.pad(a, pad).reshape(nby, b, nbx, b).sum(axis=(1, 3))

        for k in self.deslocamentos:
            if not buraco_c.any():
                break
            s = t + k
            if s not in buf:
                continue
            es = buf[s]
            H = self.homografia(t, s, buf)
            if H is None:
                continue
            if isinstance(H, str):
                img_s = es["frame"][y0:y1, x0:x1]
                valido_s = es["mascara"][y0:y1, x0:x1] == 0
                cinza_s = es["cinza"][y0:y1, x0:x1].astype(np.int16)
            else:
                M = T @ H
                img_s = cv2.warpPerspective(es["frame"], M, (bw, bh), flags=cv2.INTER_LINEAR)
                cinza_s = cv2.warpPerspective(es["cinza"], M, (bw, bh), flags=cv2.INTER_LINEAR).astype(np.int16)
                dentro = cv2.warpPerspective(np.full_like(es["mascara"], 255), M, (bw, bh),
                                             flags=cv2.INTER_NEAREST)
                tapado = cv2.warpPerspective(cv2.dilate(es["mascara"], np.ones((3, 3), np.uint8)), M,
                                             (bw, bh), flags=cv2.INTER_NEAREST)
                valido_s = (dentro > 0) & (tapado == 0)
            cand = buraco_c & valido_s
            if not cand.any():
                continue
            # validação: diferença média em pixels conhecidos em ambos, numa janela 3x3 de blocos
            ambos = conhecido_t & valido_s
            dif = np.abs(cinza_t - cinza_s).astype(np.float32) * ambos
            soma = cv2.boxFilter(por_bloco(dif), -1, (3, 3), normalize=False,
                                 borderType=cv2.BORDER_CONSTANT)
            n = cv2.boxFilter(por_bloco(ambos.astype(np.float32)), -1, (3, 3), normalize=False,
                              borderType=cv2.BORDER_CONSTANT)
            ok_bloco = (n >= 0.15 * b * b) & (soma < self.limiar * np.maximum(n, 1))
            ok = np.kron(ok_bloco, np.ones((b, b), bool))[:bh, :bw]
            aceitar = cand & ok
            if not aceitar.any():
                continue
            frame[y0:y1, x0:x1][aceitar] = img_s[aceitar]
            fonte[y0:y1, x0:x1][aceitar] = TEMPORAL
            buraco_c &= ~aceitar
        return frame, fonte


def remover_texto(frame, mascara, modo):
    if not mascara.any():
        return frame
    if modo == "suave":
        ys, xs = np.nonzero(mascara)
        p = 24
        y0, y1 = max(ys.min() - p, 0), min(ys.max() + p + 1, frame.shape[0])
        x0, x1 = max(xs.min() - p, 0), min(xs.max() + p + 1, frame.shape[1])
        frame[y0:y1, x0:x1] = preencher_suave(frame[y0:y1, x0:x1], mascara[y0:y1, x0:x1])
        return frame
    if modo == "inpaint":
        # Inpainting clássico (Telea) só num recorte à volta do que falta: propaga as cores
        # vizinhas (que já incluem os pixels reais vindos de outros frames). Sem IA generativa.
        ys, xs = np.nonzero(mascara)
        p = 16
        y0, y1 = max(ys.min() - p, 0), min(ys.max() + p + 1, frame.shape[0])
        x0, x1 = max(xs.min() - p, 0), min(xs.max() + p + 1, frame.shape[1])
        frame[y0:y1, x0:x1] = cv2.inpaint(frame[y0:y1, x0:x1], mascara[y0:y1, x0:x1], 3,
                                          cv2.INPAINT_TELEA)
        return frame
    sel = mascara > 0
    if modo == "preencher":
        frame[sel] = 0  # preto sólido
    elif modo == "borrar":
        borrado = cv2.GaussianBlur(frame, (0, 0), 15)
        frame[sel] = borrado[sel]
    return frame


def desenhar(frame, deteccoes, cor=(0, 255, 0)):
    for d in deteccoes:
        poli = np.array(d["poligono"], dtype=np.int32)
        cv2.polylines(frame, [poli], True, cor, 2)
        if d["texto"]:
            x, y, _, _ = d["caixa"]
            rotulo = f'{d["texto"]} ({d["confianca"]:.2f})'
            (tw, th), _ = cv2.getTextSize(rotulo, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y0 = max(y - 4, th + 4)
            cv2.rectangle(frame, (x, y0 - th - 4), (x + tw + 4, y0 + 2), cor, -1)
            cv2.putText(frame, rotulo, (x + 2, y0 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def main():
    p = argparse.ArgumentParser(description="Deteta (ou remove) texto em vídeo.")
    p.add_argument("video", help="caminho do vídeo de entrada")
    p.add_argument("-o", "--saida", help="vídeo de saída (padrão: <nome>_texto.mp4 ou <nome>_sem_texto.mp4)")
    p.add_argument("--json", help="ficheiro JSON de resultados (padrão: <saida>.json)")
    p.add_argument("--a-cada", type=int, default=1,
                   help="analisar 1 em cada N frames (os outros reutilizam as caixas). Padrão: 1")
    p.add_argument("--min-conf", type=float, default=0.5,
                   help="confiança mínima do OCR para aceitar uma caixa. Padrão: 0.5")
    p.add_argument("--so-detecao", action="store_true",
                   help="só deteção (sem ler o texto) — mais rápido, mais falsos positivos")
    p.add_argument("--remover", choices=["suave", "inpaint", "preencher", "borrar"],
                   help="remove o texto em vez de desenhar as caixas. Primeiro usa pixels reais de "
                        "outros frames; este modo só se aplica ao que sobrar. Recomendado: suave")
    p.add_argument("--mascara", choices=["letras", "caixa"], default="letras",
                   help="'letras' = só os pixels do texto (padrão); 'caixa' = a caixa inteira")
    p.add_argument("--margem", type=int, default=3,
                   help="pixels a alargar à volta das letras/caixas ao remover. Padrão: 3")
    p.add_argument("--janela", type=int, default=30,
                   help="frames para trás e para a frente onde procurar o fundo real. 0 desliga. Padrão: 30")
    p.add_argument("--limiar-temporal", type=float, default=6.0,
                   help="diferença média máxima (0-255) à volta de um bloco para aceitar os pixels de "
                        "outro frame. Menor = mais rigoroso. Padrão: 6")
    p.add_argument("--persistir", type=int, default=3,
                   help="se um frame não tiver deteção, reutiliza as caixas anteriores durante até N frames "
                        "(evita o texto 'piscar' quando o detetor falha num frame). 0 desliga. Padrão: 3")
    p.add_argument("--salvar-mascara", action="store_true",
                   help="grava também <saida>_mascara.mp4 sem perdas (cinzento = pixels reais de outro "
                        "frame, branco = interpolado/preenchido)")
    p.add_argument("--sem-perdas", action="store_true",
                   help="grava o vídeo de saída sem perdas (libx264rgb qp 0) — recomendado para deepfakes")
    p.add_argument("--mostrar", action="store_true", help="mostrar o vídeo numa janela enquanto processa")
    args = p.parse_args()

    entrada = Path(args.video)
    if not entrada.exists():
        sys.exit(f"Vídeo não encontrado: {entrada}")
    sufixo = "_sem_texto" if args.remover else "_texto"
    saida = Path(args.saida or entrada.with_name(entrada.stem + sufixo + ".mp4"))
    saida_json = Path(args.json or saida.with_suffix(".json"))
    saida_mascara = saida.with_name(saida.stem + "_mascara.mp4")

    if args.remover and args.a_cada > 1:
        print("Aviso: com --remover e --a-cada > 1, texto em movimento pode deixar restos visíveis.")

    cap = cv2.VideoCapture(str(entrada))
    if not cap.isOpened():
        sys.exit(f"Não foi possível abrir o vídeo: {entrada}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    com_audio = shutil.which("ffmpeg") and tem_audio(entrada)
    # Com áudio: grava primeiro o vídeo num ficheiro temporário e depois junta o áudio original.
    saida_video = saida.with_name(".tmp_" + saida.name) if com_audio else saida
    writer = EscritorVideo(saida_video, fps, w, h, sem_perdas=args.sem_perdas)
    writer_mascara = (EscritorVideo(saida_mascara, fps, w, h, sem_perdas=True)
                      if args.remover and args.salvar_mascara else None)

    motor = RapidOCR()
    janela = max(args.janela, 0) if args.remover else 0
    temporal = PreenchedorTemporal(janela, args.limiar_temporal) if janela else None
    resultados = []
    ultimas = []
    ultimo_frame_com_texto = None
    n = 0
    frames_com_texto = 0
    px = {"mascara": 0, "temporal": 0, "fallback": 0}
    buf = {}          # frame -> {frame, mascara, cinza, pts, des}; guarda ±janela frames originais
    parar = False

    def emitir(t):
        """Processa e grava o frame t (precisa dos frames t-janela .. t+janela no buffer)."""
        nonlocal parar
        e = buf[t]
        if args.remover:
            if temporal:
                frame, fonte = temporal.preencher(t, buf)
            else:
                frame, fonte = e["frame"].copy(), np.where(e["mascara"] > 0, FALLBACK, ORIGINAL).astype(np.uint8)
            resto = np.where(fonte == FALLBACK, 255, 0).astype(np.uint8)
            frame = remover_texto(frame, resto, args.remover)
            px["mascara"] += int((fonte > 0).sum())
            px["temporal"] += int((fonte == TEMPORAL).sum())
            px["fallback"] += int((fonte == FALLBACK).sum())
            if writer_mascara:
                writer_mascara.write(cv2.cvtColor(fonte, cv2.COLOR_GRAY2BGR))
        else:
            frame = desenhar(e["frame"].copy(), e["deteccoes"])
        writer.write(frame)
        if args.mostrar:
            cv2.imshow("Texto (q para sair)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                parar = True

    proximo = 0  # próximo frame a gravar
    while not parar:
        ok, frame = cap.read()
        if not ok:
            break
        if n % max(args.a_cada, 1) == 0:
            atuais = analisar_frame(motor, frame, args.so_detecao, args.min_conf)
            if atuais:
                frames_com_texto += 1
                resultados.append({"frame": n, "tempo_s": round(n / fps, 3), "deteccoes": atuais})
                ultimas, ultimo_frame_com_texto = atuais, n
            elif ultimo_frame_com_texto is None or n - ultimo_frame_com_texto > args.persistir:
                ultimas = []
            # senão: o detetor falhou neste frame mas havia texto há pouco -> mantém as caixas anteriores

        e = {"frame": frame, "deteccoes": ultimas}
        if args.remover:
            e["mascara"] = criar_mascara(frame, ultimas, args.margem, args.mascara)
            if temporal:
                e.update(temporal.caracteristicas(frame, e["mascara"]))
                if n - 1 in buf:
                    e["A"] = temporal.homografia_bruta(buf[n - 1], e)
        buf[n] = e
        n += 1

        while proximo <= n - 1 - janela and not parar:
            emitir(proximo)
            buf.pop(proximo - janela, None)
            proximo += 1
        if n % 25 == 0:
            print(f"\r{n}/{total} frames", end="", flush=True)

    while proximo < n and not parar:
        emitir(proximo)
        buf.pop(proximo - janela, None)
        proximo += 1

    cap.release()
    writer.release()
    if com_audio:
        try:
            info_audio = juntar_audio(saida_video, entrada, saida)
        finally:
            saida_video.unlink(missing_ok=True)
    else:
        info_audio = "vídeo de entrada sem áudio" if shutil.which("ffmpeg") else "sem ffmpeg, áudio não incluído"
    if writer_mascara:
        writer_mascara.release()
    if args.mostrar:
        cv2.destroyAllWindows()

    remocao = None
    if args.remover:
        tot = max(px["mascara"], 1)
        remocao = {
            "modo_fallback": args.remover, "mascara": args.mascara, "margem": args.margem,
            "janela": janela, "limiar_temporal": args.limiar_temporal,
            "pixels_mascara": px["mascara"],
            "pct_pixels_reais_de_outros_frames": round(100 * px["temporal"] / tot, 2),
            "pct_pixels_fallback": round(100 * px["fallback"] / tot, 2),
        }
    saida_json.write_text(json.dumps({
        "video": str(entrada), "fps": fps, "largura": w, "altura": h,
        "frames_processados": n, "analisar_a_cada": args.a_cada,
        "remocao": remocao,
        "frames_com_texto": resultados,
    }, ensure_ascii=False, indent=2))

    print(f"\nFeito. {frames_com_texto} frames analisados com texto.")
    if remocao:
        print(f"Removido:   {remocao['pct_pixels_reais_de_outros_frames']}% com pixels reais de outros frames, "
              f"{remocao['pct_pixels_fallback']}% com '{args.remover}' (fundo nunca visível)")
    print(f"Vídeo:      {saida}")
    print(f"Áudio:      {info_audio}")
    if writer_mascara:
        print(f"Máscara:    {saida_mascara}")
    print(f"Resultados: {saida_json}")


if __name__ == "__main__":
    main()
