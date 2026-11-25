# Identificador de texto em vídeo

Deteta onde há texto escrito num vídeo e desenha bounding boxes à volta dele.

## Instalação
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Interface gráfica
```bash
.venv/bin/python gui.py
```
Escolhe o vídeo e o modo: **Identificar texto** (bounding boxes) ou **Remover texto** (sem IA
generativa: primeiro pixels reais de outros frames, depois o modo escolhido só para o fundo nunca
visível). Mostra o progresso, permite cancelar e no fim abre o vídeo, a máscara ou o JSON.

## Uso (linha de comandos)
```bash
.venv/bin/python detectar_texto.py video.mp4                 # todos os frames
.venv/bin/python detectar_texto.py video.mp4 --a-cada 5      # 1 em cada 5 frames (mais rápido)
.venv/bin/python detectar_texto.py video.mp4 --min-conf 0.7  # mais exigente
.venv/bin/python detectar_texto.py video.mp4 --so-detecao    # só caixas, sem ler o texto
```
Gera `video_texto.mp4` (com as caixas) e `video_texto.json` (frame, tempo, caixas, texto e confiança).

## Pesquisa de modelos OCR

| Modelo | Deteção | Reconhecimento | Notas |
|---|---|---|---|
| **PP-OCR (PaddleOCR) via RapidOCR** | DBNet | SVTR/CRNN | **Escolhido.** ONNX, corre bem em CPU, multilíngue, pip simples |
| EasyOCR | CRAFT | CRNN | Bom, mas precisa de PyTorch (pesado, lento em CPU) |
| OpenCV EAST / DB | EAST/DB | — | Só deteção, sem confirmar se o texto é legível |
| Tesseract | fraco em cenas | LSTM | Pensado para documentos, mau em texto de cena/vídeo |
| docTR | DBNet/LinkNet | CRNN/ViTSTR | Boa qualidade, mais pesado |

## Como funciona
1. **Deteção** (modelo de deteção DBNet) encontra regiões com texto em cada frame.
2. **OCR** lê cada região e dá uma confiança.
3. Só se mantêm as caixas com confiança ≥ `--min-conf`, ou seja, texto que é mesmo detetável/legível.

## Remover o texto (pré-processamento para deteção de deepfakes)
```bash
.venv/bin/python detectar_texto.py video.mp4 --remover suave --sem-perdas --salvar-mascara
```
Gera `video_sem_texto.mp4`, `video_sem_texto_mascara.mp4` e `video_sem_texto.json`.

**Sem IA generativa.** A remoção é feita em 3 passos:
1. **Máscara justa às letras** (`--mascara letras`, padrão): as cores do texto são as que aparecem
   muito mais dentro das caixas do que à volta. Junta-se o contorno/sombra (cor uniforme colada às
   letras, e o que difere muito do fundo estimado) e os buracos pequenos entre letras. Altera muito
   menos pixels do que apagar a caixa inteira (`--mascara caixa`).
2. **Pixels reais de outros frames:** procura até `--janela` frames para trás e para a frente
   (padrão 30) onde esse fundo esteja visível (o texto mudou ou a câmara/pessoa mexeu-se). Os frames
   são alinhados pelo movimento da câmara (ORB + homografia) e cada bloco só é copiado se, à volta
   dele, o outro frame coincidir com o atual (`--limiar-temporal`, padrão 6). Com a câmara parada
   a cópia é direta, sem reamostragem.
3. **O resto** (fundo que nunca aparece, por exemplo texto parado o vídeo todo) vai para o modo de `--remover`:

| Modo | O que faz | Quando usar |
|---|---|---|
| `suave` | Interpolação harmónica (Laplace): transição suave das cores da borda | **Recomendado.** Sem facetas nem riscos |
| `inpaint` | Inpainting clássico Telea (OpenCV) | Comportamento antigo; deixa facetas em áreas grandes |
| `preencher` | Preto sólido | Quando o detetor vai ignorar as regiões da máscara |
| `borrar` | Desfoque gaussiano | Só esconder; pode deixar um "fantasma" do texto |

Máscara gravada (`--salvar-mascara`, sem perdas): **preto** = original, **cinzento** = pixels reais
vindos de outro frame, **branco** = interpolado pelo modo acima. O JSON indica a percentagem de cada.

Outras opções: `--margem` (px extra à volta das letras, padrão 3), `--persistir` (mantém as caixas
até N frames quando o detetor falha, padrão 3), `--sem-perdas` (saída libx264rgb qp 0).

Limites: onde o fundo nunca é visível noutro frame não há informação real. Sem IA, o melhor
possível é uma transição suave, que continua a notar-se sobre textura (barba, cortinas). Nesses
casos, para o detetor de deepfakes, pode ser preferível ignorar as zonas brancas da máscara.

Cuidados para deepfakes:
- **Não** usar inpainting generativo (LaMa, ProPainter…): introduz pixels sintéticos — exatamente o que o detetor procura.
- Recomprimir altera os traços de compressão: usar `--sem-perdas`.
- Aplicar **o mesmo** pré-processamento a vídeos reais e falsos, senão o modelo aprende "processado vs. não processado".
- Usar `--a-cada 1` (padrão) ao remover.
- O áudio original é copiado sem recodificar (só converte para AAC se o codec não couber no `.mp4`).
