"""
Módulo de Reporte PDF — App 2: Modelos Estadísticos
===================================================
Genera el reporte institucional de la aplicación de Modelos Estadísticos
(cafetería universitaria, 7:00-11:00 AM) a partir de los datos vivos de
Supabase y de las funciones que ya existen en:

    data_manager.py         load_data / clean_data / next_id
    stats_utils.py          descriptive_stats / executive_summary / detect_outliers_iqr
    viz_utils.py            plot_hist / plot_box / plot_box_by_group / plot_bars
    mod_distribuciones.py   plot_ajuste_con_qq / simular_poisson_llegadas / tabla_bondad_ajuste
    mod_hipotesis.py        comparar_dos_grupos / entrenar_regresion_logistica
    mod_bayes.py            teorema_bayes / plot_bayes_barras / entrenar_naive_bayes / comparar_modelos
    mod_ia_analista.py      analizar_seccion_ia (Gemini, opcional con fallback)

Estructura del documento
------------------------
    Resumen Ejecutivo (front matter, sin numerar)
     1. Definición del Problema
     2. Planteamiento de Objetivos
     3. Marco Conceptual
     4. Adquisición y Análisis de Datos
     5. Estadística Descriptiva (U1)
     6. Distribuciones de Probabilidad (U2)
     7. Pruebas de Hipótesis (U3)
     8. Estadística Bayesiana (U4)
     9. Conclusiones y Recomendaciones
    10. Referencias Bibliográficas

Fase de trabajo
---------------
FASE 1 (este archivo): pipeline completo — contexto, textos, tres pasadas
(borrador → paginado → final), resumen ejecutivo y los diez capítulos. La
clase `ReporteCafeteriaEst` reutiliza la lógica institucional de la App 1
(portada, header/footer, índices, tablas multipágina).

API pública
-----------
    generar_reporte_est_pdf(df=None, min_registros=20) -> (bytes | None, str | None)
    generar_reporte_est(df=None) -> bytes          (lanza RuntimeError si falla)
    ReporteCafeteriaEst                            (alias: ReporteEstadistico)

API interna (una llamada por etapa)
-----------------------------------
    _construir_contexto_est(df)   -> dict con KPIs, pruebas, modelos y PNG
    _preparar_tareas_ia_est(ctx)  -> dict de textos (Gemini o respaldo)
    _render_reporte_est(ctx, textos) -> ReporteCafeteriaEst ya paginado
    _resumen_ejecutivo_est(pdf, ctx, textos)
    _construir_contenido_est(pdf, ctx, textos)

Las funciones que aporta la Fase 2 (ya implementadas en sus módulos) se
invocan con `hasattr(...)`; si faltaran, el capítulo muestra una nota de
"pendiente" en lugar de fallar. Firmas disponibles:

    stats_utils.media_ponderada(df, value_col, weight_col) -> float
    stats_utils.matriz_correlacion(df, cols=None) -> pd.DataFrame
    viz_utils.plot_violin(df, value_col, group_col, title, ylabel) -> Figure
    viz_utils.plot_matriz_correlacion(df, cols=None, title=...) -> Figure
    viz_utils.plot_arbol_bayes(resultado, etiqueta_a, etiqueta_b) -> Figure
    mod_distribuciones.simular_bernoulli(p=0.5, n=1000) -> dict
    mod_distribuciones.simular_binomial(n_ensayos=10, p=0.5, n=1000) -> dict
    mod_distribuciones.simular_geometrica(p=0.3, n=1000) -> dict
    mod_distribuciones.simular_hipergeometrica(N=200, K=60, n=10) -> dict
    mod_distribuciones.simular_gamma(shape=2.0, scale=1.5, n=1000) -> dict
        (cada simular_* devuelve {"fig": Figure, "resumen": dict})
    mod_hipotesis.anova(df, valor_col, grupo_col) -> dict
    mod_hipotesis.chi_cuadrado(df, var1, var2) -> dict
"""
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # backend sin GUI: el PDF se arma en servidor, nunca en Tk
import matplotlib.pyplot as plt
from fpdf import FPDF

import data_manager as dm
import stats_utils as su
import viz_utils as vu
import mod_distribuciones as md
import mod_hipotesis as mh
import mod_bayes as mb

try:
    from mod_ia_analista import analizar_seccion_ia
    HAS_IA = True
except ImportError:  # pragma: no cover - la IA es opcional
    HAS_IA = False

    def analizar_seccion_ia(seccion, datos_metricas, fallback_texto,
                            imagen_png_bytes=None, instruccion_extra=None,
                            doble_capa=True, tipo="analisis"):
        """Stub con la misma firma: devuelve el texto de respaldo."""
        return {"tecnico": fallback_texto, "simple": None}


# ============================================================
# CONSTANTES DEL DOCUMENTO
# ============================================================
TITULO_CORTO = "Cafetería Universitaria — Modelos Estadísticos"
AUTORES = ("Rodríguez Avilés Selene", "Raygoza Flores Gabriel")
DOCENTE = "Dr. Valdez Acosta Fevrier Adolfo"
INSTITUCION = "Instituto Tecnológico de Tijuana - TecNM"
PROGRAMA = "Maestría en Inteligencia Artificial"

CAPITULOS = [
    ("1", "Definición del Problema"),
    ("2", "Planteamiento de Objetivos"),
    ("3", "Marco Conceptual"),
    ("4", "Adquisición y Análisis de Datos"),
    ("5", "Estadística Descriptiva (U1)"),
    ("6", "Distribuciones de Probabilidad (U2)"),
    ("7", "Pruebas de Hipótesis (U3)"),
    ("8", "Estadística Bayesiana (U4)"),
    ("9", "Conclusiones y Recomendaciones"),
    ("10", "Referencias Bibliográficas"),
]

# Paleta institucional
AZUL = (0, 51, 102)
GRIS = (90, 90, 90)
VERDE = (20, 110, 60)


# ============================================================
# HELPERS
# ============================================================
def _fig_a_png(fig, dpi=110):
    """Convierte una figura de Matplotlib a PNG (y libera la figura)."""
    if fig is None:
        return None
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception:
        try:
            plt.close(fig)
        except Exception:
            pass
        return None


def _get_asset_path(nombre: str) -> str | None:
    """Ruta absoluta a un archivo de assets/, o None si no existe."""
    p = Path(__file__).parent / "assets" / nombre
    return str(p) if p.exists() else None


def _fmt(valor, decimales=3, sufijo="") -> str:
    """Formatea números para tablas del PDF (sin notación científica)."""
    if valor is None:
        return "-"
    try:
        if pd.isna(valor):
            return "-"
    except (TypeError, ValueError):
        pass
    if isinstance(valor, (int, np.integer)):
        return f"{int(valor):,}{sufijo}"
    if isinstance(valor, (float, np.floating)):
        return f"{float(valor):,.{decimales}f}{sufijo}"
    return str(valor)


def _pct(valor, decimales=1) -> str:
    return _fmt(valor, decimales, " %")


# ============================================================
# CLASE PDF
# ============================================================
class ReporteCafeteriaEst(FPDF):
    """PDF A4 con portada institucional, índices y tablas multipágina."""

    # Defaults a nivel de clase: fpdf2 puede llamar header()/footer()
    # antes de que _init_registros() se ejecute.
    indice_caps: dict = {}
    indice_figs: dict = {}
    indice_tabs: dict = {}
    fig_counter: int = 0
    tab_counter: int = 0
    cap_actual: str = ""
    total_pages: int = 0
    page_skip_header: bool = False
    page_skip_footer: bool = False
    _paginas_sin_header: set = set()
    _paginas_sin_footer: set = set()
    modo: str = "final"

    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.FONT = "Helvetica"
        self._register_unicode_fonts()
        self._init_registros()
        self.modo = "final"

    # ---------- FUENTES ----------
    def _register_unicode_fonts(self):
        """Registra DejaVu Sans (Unicode) desde los TTF de matplotlib."""
        try:
            import matplotlib
            font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
            variantes = {"": "DejaVuSans.ttf", "B": "DejaVuSans-Bold.ttf",
                         "I": "DejaVuSans-Oblique.ttf",
                         "BI": "DejaVuSans-BoldOblique.ttf"}
            for estilo, fname in variantes.items():
                fpath = font_dir / fname
                if not fpath.exists():
                    return
                self.add_font("DejaVu", estilo, str(fpath))
            self.FONT = "DejaVu"
        except Exception:
            pass

    # ---------- REGISTROS / ÍNDICES ----------
    def _init_registros(self):
        self.indice_caps = {}
        self.indice_figs = {}
        self.indice_tabs = {}
        self.fig_counter = 0
        self.tab_counter = 0
        self.cap_actual = ""
        self.total_pages = 0
        self.page_skip_header = False
        self.page_skip_footer = False
        self._paginas_sin_header = set()
        self._paginas_sin_footer = set()

    def _saltar_header_footer(self, skip: bool):
        self.page_skip_header = skip
        self.page_skip_footer = skip
        if skip:
            pagina = self.page_no() or 1
            self._paginas_sin_header.add(pagina)
            self._paginas_sin_footer.add(pagina)

    def header(self):
        if self.page_skip_header or self.page_no() in self._paginas_sin_header:
            return
        if not self.cap_actual:
            return
        self.set_font(self.FONT, "B", 8)
        self.set_text_color(120, 120, 120)
        self.cell(95, 5, TITULO_CORTO, 0, 0, "L")
        self.cell(0, 5, self.cap_actual, 0, 1, "R")
        self.set_draw_color(200, 200, 200)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2)

    def footer(self):
        if self.page_skip_footer or self.page_no() in self._paginas_sin_footer:
            return
        self.set_y(-15)
        self.set_font(self.FONT, "I", 8)
        self.set_text_color(150, 150, 150)
        if self.total_pages > 0:
            texto = f"Página {self.page_no()} de {self.total_pages}"
        else:
            texto = f"Página {self.page_no()}"
        self.cell(0, 10, texto, 0, 0, "C")

    # ---------- BÁSICOS ----------
    def titulo_capitulo(self, num, titulo):
        """Abre página, registra el capítulo en el índice y dibuja el título."""
        alias = f"{num}. {titulo}"
        self.cap_actual = f"Cap. {num}: {titulo[:40]}"
        if self.page_no() > 0:
            self.add_page()
        self.indice_caps[alias] = self.page_no()
        self.set_font(self.FONT, "B", 14)
        self.set_text_color(*AZUL)
        self.multi_cell(0, 8, alias, 0, "L")
        self.set_draw_color(*AZUL)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)
        self.set_text_color(0, 0, 0)

    def subtitulo(self, titulo):
        self.set_x(self.l_margin)
        self.set_font(self.FONT, "B", 10.5)
        self.set_text_color(51, 51, 51)
        self.multi_cell(0, 6, titulo, 0, "L")
        self.ln(1.5)
        self.set_text_color(0, 0, 0)

    def parrafo(self, texto):
        self.set_x(self.l_margin)
        self.set_font(self.FONT, "", 9)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 5, str(texto))
        self.ln(2.5)

    def nota_doble(self, tecnico: str, simple: str | None):
        """Análisis IA en dos capas: técnico (cursiva) + palabras simples."""
        if tecnico:
            self.set_x(self.l_margin)
            self.set_font(self.FONT, "I", 8)
            self.set_text_color(80, 80, 80)
            self.multi_cell(0, 4, f">> Análisis técnico: {tecnico}")
            self.ln(1)
        if simple:
            self.set_x(self.l_margin)
            self.set_font(self.FONT, "", 9)
            self.set_text_color(*VERDE)
            self.multi_cell(0, 4.5, f"En palabras simples: {simple}")
        self.set_text_color(0, 0, 0)
        self.ln(2.5)

    def preguntas_guia(self, preguntas: list[str]):
        if not preguntas:
            return
        self.set_x(self.l_margin)
        self.set_font(self.FONT, "I", 8)
        self.set_text_color(100, 100, 100)
        self.multi_cell(0, 4, "Preguntas para profundizar:")
        for i, p in enumerate(preguntas, 1):
            self.set_x(self.l_margin)
            self.multi_cell(0, 4, f"  {i}. {p}")
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def nota_pendiente(self, texto):
        """Aviso visible para funciones que se agregan en la Fase 2."""
        self.set_x(self.l_margin)
        self.set_font(self.FONT, "I", 8)
        self.set_text_color(160, 90, 0)
        self.multi_cell(0, 4.5, f"[Pendiente Fase 2] {texto}")
        self.set_text_color(0, 0, 0)
        self.ln(2)

    # ---------- TABLAS ----------
    def _encabezado_tabla(self, cols, widths, font_size=7):
        self.set_x(self.l_margin)
        self.set_font(self.FONT, "B", font_size)
        self.set_fill_color(*AZUL)
        self.set_text_color(255, 255, 255)
        for col, w in zip(cols, widths):
            self.cell(w, 6, self._trunc(str(col), w - 2, font_size),
                      1, 0, "C", True)
        self.set_text_color(0, 0, 0)
        self.ln(6)
        self.set_font(self.FONT, "", font_size)

    def _trunc(self, texto, ancho, font_size):
        """Recorta un texto para que quepa en `ancho` mm."""
        if ancho <= 1:
            return ""
        if self.get_string_width(texto) <= ancho:
            return texto
        recorte = texto
        while recorte and self.get_string_width(recorte + "...") > ancho:
            recorte = recorte[:-1]
        return (recorte + "...") if recorte else ""

    def tabla_metrica(self, filas, ancho1=60, ancho2=60,
                      encabezados=("Métrica", "Valor")):
        """Tabla de dos columnas con celdas multilínea y salto de página."""
        ancho_util = self.w - self.l_margin - self.r_margin
        if ancho1 + ancho2 > ancho_util:
            ancho2 = ancho_util - ancho1

        def encabezado():
            self.set_x(self.l_margin)
            self.set_font(self.FONT, "B", 8)
            self.set_fill_color(*AZUL)
            self.set_text_color(255, 255, 255)
            self.cell(ancho1, 7, str(encabezados[0]), 1, 0, "C", True)
            self.cell(ancho2, 7, str(encabezados[1]), 1, 1, "C", True)
            self.set_font(self.FONT, "", 8)
            self.set_text_color(0, 0, 0)

        encabezado()
        for k, v in filas:
            if self.get_y() + 7 > self.h - self.b_margin - 10:
                self.add_page()
                encabezado()
            x0, y0 = self.l_margin, self.get_y()
            self.set_xy(x0, y0)
            self.multi_cell(ancho1, 5, str(k), 1, "L")
            y1 = self.get_y()
            self.set_xy(x0 + ancho1, y0)
            self.multi_cell(ancho2, 5, str(v), 1, "L")
            y2 = self.get_y()
            self.set_xy(x0, max(y1, y2))
        self.ln(3)

    def tabla_df(self, df, cols=None, widths=None, font_size=7,
                 max_rows=40, caption_extra=""):
        """Dibuja un DataFrame como tabla multipágina.

        - `cols`: subconjunto/orden de columnas a mostrar.
        - `widths`: anchos relativos (se escalan al ancho útil).
        - Las filas se recortan a `max_rows` para no inflar el reporte.
        """
        if df is None or len(df) == 0:
            self.parrafo("(Sin datos disponibles)")
            return
        cols = [c for c in (cols or list(df.columns)) if c in df.columns]
        if not cols:
            self.parrafo("(Sin columnas disponibles)")
            return

        ancho_util = self.w - self.l_margin - self.r_margin
        if widths is None:
            widths = [1] * len(cols)
        base = sum(widths) if sum(widths) > 0 else len(cols)
        widths = [ancho_util * w / base for w in widths]

        data = df.loc[:, cols].head(max_rows)

        self._encabezado_tabla(cols, widths, font_size)
        for _, row in data.iterrows():
            if self.get_y() + 5 > self.h - self.b_margin - 12:
                self.add_page()
                self._encabezado_tabla(cols, widths, font_size)
            self.set_x(self.l_margin)
            self.set_font(self.FONT, "", font_size)
            for col, w in zip(cols, widths):
                valor = row[col]
                numerico = isinstance(
                    valor, (int, float, np.integer, np.floating)
                )
                texto = _fmt(valor) if numerico else str(valor)
                self.cell(w, 5, self._trunc(texto, w - 1.5, font_size),
                          1, 0, "R" if numerico else "L")
            self.ln(5)

        if len(df) > max_rows:
            self.set_x(self.l_margin)
            self.set_font(self.FONT, "I", 7)
            self.set_text_color(*GRIS)
            self.multi_cell(0, 4, f"{caption_extra}Mostrando {max_rows} de "
                                  f"{len(df)} filas.")
            self.set_text_color(0, 0, 0)
        self.ln(3)

    # ---------- FIGURAS ----------
    def agregar_figura(self, fig_o_bytes, titulo="", ancho=160):
        if fig_o_bytes is None:
            return
        if isinstance(fig_o_bytes, (bytes, bytearray)):
            img_bytes = bytes(fig_o_bytes)
        else:
            img_bytes = _fig_a_png(fig_o_bytes)
        if not img_bytes:
            return
        self.set_x(self.l_margin)
        if titulo:
            self.set_font(self.FONT, "I", 8)
            self.set_text_color(*GRIS)
            self.multi_cell(0, 4.5, titulo, 0, "C")
        x = (self.w - ancho) / 2
        self.image(io.BytesIO(img_bytes), x=x, w=ancho)
        self.ln(2)

    def caption_figura(self, alias: str):
        """Registra el caption de una figura y lo dibuja DEBAJO de ella."""
        self.fig_counter += 1
        texto = f"Figura {self.fig_counter}. {alias}"
        self.indice_figs[texto] = self.page_no()
        self.set_x(self.l_margin)
        self.set_font(self.FONT, "I", 8)
        self.set_text_color(*GRIS)
        self.multi_cell(0, 4.5, texto, 0, "C")
        self.ln(2)
        self.set_text_color(0, 0, 0)

    def caption_tabla(self, alias: str):
        """Registra y dibuja el caption de una tabla (ARRIBA de la tabla)."""
        self.tab_counter += 1
        texto = f"Tabla {self.tab_counter}. {alias}"
        self.indice_tabs[texto] = self.page_no()
        self.set_x(self.l_margin)
        self.set_font(self.FONT, "B", 8)
        self.set_text_color(*GRIS)
        self.multi_cell(0, 4.5, texto, 0, "L")
        self.ln(1)
        self.set_text_color(0, 0, 0)

    # ---------- PORTADA INSTITUCIONAL ----------
    def portada_institucional(self):
        """Portada tipo tesis con los dos logos institucionales."""
        self._saltar_header_footer(True)
        self.add_page()

        logo_tec = _get_asset_path("logo_tecnm_virtual.png")
        logo_ia = _get_asset_path("logo_maestria_ia.png")
        if logo_tec:
            self.image(logo_tec, x=92, y=18, w=26)
        if logo_ia:
            self.image(logo_ia, x=91, y=232, w=28)

        self.set_y(66)
        self.set_font(self.FONT, "B", 14)
        self.set_text_color(*AZUL)
        self.cell(0, 8, "INSTITUTO TECNOLÓGICO DE TIJUANA", 0, 1, "C")
        self.set_font(self.FONT, "", 11)
        self.cell(0, 6, PROGRAMA, 0, 1, "C")
        self.ln(6)

        self.set_draw_color(*AZUL)
        self.line(60, self.get_y(), 150, self.get_y())
        self.ln(10)

        self.set_font(self.FONT, "B", 18)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 9, "Reporte de Modelos Estadísticos\n"
                              "Cafetería Universitaria", 0, "C")
        self.ln(3)
        self.set_font(self.FONT, "I", 12)
        self.cell(0, 7, "Proyecto Integrador — Unidades 1 a 4", 0, 1, "C")
        self.ln(6)

        self.set_draw_color(*AZUL)
        self.line(60, self.get_y(), 150, self.get_y())
        self.ln(14)

        self.set_font(self.FONT, "", 11)
        self.cell(0, 6, "Autores:", 0, 1, "C")
        self.set_font(self.FONT, "B", 11)
        for autor in AUTORES:
            self.cell(0, 6, autor, 0, 1, "C")
        self.ln(7)

        self.set_font(self.FONT, "", 10)
        self.cell(0, 6, f"Docente: {DOCENTE}", 0, 1, "C")
        self.cell(0, 6, INSTITUCION, 0, 1, "C")

        self.ln(8)
        self.set_font(self.FONT, "I", 10)
        self.cell(0, 6, "Tijuana, Baja California", 0, 1, "C")
        self.cell(0, 6, f"Noviembre {datetime.now().year}", 0, 1, "C")

        self._saltar_header_footer(False)

    # ---------- ÍNDICES ----------
    def _cabecera_indice(self, titulo: str):
        """Prepara una página de índice: sin encabezado, con pie."""
        self.page_skip_header = True
        self.page_skip_footer = False
        self.add_page()
        self.page_skip_header = False
        self.cap_actual = ""
        self.set_font(self.FONT, "B", 16)
        self.set_text_color(*AZUL)
        self.cell(0, 10, titulo, 0, 1, "L")
        self.ln(4)
        self.set_font(self.FONT, "", 10)
        self.set_text_color(0, 0, 0)

    def _fila_indice(self, titulo: str, pagina, con_puntos: bool = True):
        """Dibuja una entrada de índice con el número de página a la derecha."""
        if self.get_y() + 7 > self.h - self.b_margin - 8:
            self.add_page()
            self.cap_actual = ""
        self.set_x(self.l_margin)
        self.set_font(self.FONT, "", 10)
        self.set_text_color(0, 0, 0)
        ancho_total = self.w - self.l_margin - self.r_margin
        ancho_titulo = min(self.get_string_width(titulo) + 2, ancho_total - 20)
        titulo_mostrado = titulo
        if ancho_titulo >= ancho_total - 20:
            titulo_mostrado = self._trunc(titulo, ancho_total - 22, 10)
            ancho_titulo = self.get_string_width(titulo_mostrado) + 2
        self.cell(ancho_titulo, 6, titulo_mostrado, 0, 0, "L")
        if con_puntos:
            ancho_puntos = max(ancho_total - ancho_titulo - 15, 2)
            ancho_punto = self.get_string_width(".") or 1
            n_puntos = max(int(ancho_puntos / ancho_punto) - 1, 0)
            self.set_text_color(180, 180, 180)
            self.cell(ancho_puntos, 6, "." * n_puntos, 0, 0, "L")
        self.set_text_color(0, 0, 0)
        self.set_font(self.FONT, "B", 10)
        self.cell(15, 6, str(pagina), 0, 1, "R")
        self.set_font(self.FONT, "", 10)

    def indice_contenido(self):
        self._cabecera_indice("Índice de Contenido")
        for titulo, pagina in self.indice_caps.items():
            self._fila_indice(titulo, pagina, con_puntos=True)
        self._saltar_header_footer(False)

    def indice_figuras(self):
        self._cabecera_indice("Índice de Figuras")
        for titulo, pagina in self.indice_figs.items():
            self._fila_indice(titulo, pagina, con_puntos=False)
        self._saltar_header_footer(False)

    def indice_tablas(self):
        self._cabecera_indice("Índice de Tablas")
        for titulo, pagina in self.indice_tabs.items():
            self._fila_indice(titulo, pagina, con_puntos=False)
        self._saltar_header_footer(False)


# Alias de compatibilidad: el nombre anterior se conserva para scripts previos.
ReporteEstadistico = ReporteCafeteriaEst


# ============================================================
# FASE 1: CONTEXTO DE DATOS (se calcula UNA sola vez)
# ============================================================
def _png(fn, *args, **kwargs):
    """Ejecuta una función de gráficas y devuelve PNG (None si falla)."""
    try:
        return _fig_a_png(fn(*args, **kwargs))
    except Exception:
        return None


def _calcular_bayes(df: pd.DataFrame) -> dict:
    """Prior y verosimilitudes estimadas con los datos reales.

    A = el cliente es Profesor;  B = paga con Tarjeta.
    Si P(A) es 0 o 1 se acota a [0.01, 0.99] para que la fórmula siga
    siendo interpretable (se conserva el valor empírico aparte).
    """
    tipo = df["Tipo_Cliente"].astype(str)
    pago = df["Metodo_Pago"].fillna("(Nulo)").astype(str)
    mask_a = tipo == "Profesor"
    p_a = float(mask_a.mean()) if len(df) else 0.30
    p_b_dado_a = float((pago[mask_a] == "Tarjeta").mean()) if mask_a.any() else 0.0
    p_b_dado_no_a = float((pago[~mask_a] == "Tarjeta").mean()) if (~mask_a).any() else 0.0
    p_a_calc = min(max(p_a, 0.01), 0.99)
    return {
        "etiqueta_a": "Profesor",
        "etiqueta_b": "Paga con Tarjeta",
        "p_a_empirico": p_a,
        "p_b_dado_a_empirico": p_b_dado_a,
        "p_b_dado_no_a_empirico": p_b_dado_no_a,
        "resultado": mb.teorema_bayes(p_a_calc, p_b_dado_a, p_b_dado_no_a),
    }


def _contexto(df: pd.DataFrame | None = None) -> dict:
    """Reúne datos, estadísticos, modelos y figuras (PNG) del reporte."""
    if df is None:
        df, err = dm.load_data()
        if err:
            raise RuntimeError(f"No se pudo cargar de Supabase: {err}")
    if df is None or df.empty:
        raise RuntimeError(
            "No hay registros en la base de datos para generar el reporte."
        )

    ctx: dict = {"df": df}
    clean = dm.clean_data(df)
    ctx["clean"] = clean
    ctx["kpis"] = su.executive_summary(df, clean)
    ctx["desc"] = su.descriptive_stats(df)
    ctx["desc_clean"] = su.descriptive_stats(clean)
    ctx["freq_tipo"] = su.frequency_table(df, "Tipo_Cliente")
    ctx["freq_pago"] = su.frequency_table(df, "Metodo_Pago")
    ctx["out_summary"], ctx["out_detail"] = su.detect_outliers_iqr(df)
    ctx["nulos"] = df.isna().mean().mul(100).round(2).to_dict()
    ctx["n_duplicados"] = int(df.duplicated().sum())

    png: dict = {}
    # ---------------- Unidad 1: descriptiva ----------------
    png["hist_espera"] = _png(
        vu.plot_hist, df["Tiempo_Espera_Min"],
        "Histograma — Tiempo de Espera", "Tiempo_Espera_Min")
    png["hist_consumo"] = _png(
        vu.plot_hist, df["Consumo_Total_MXN"],
        "Histograma — Consumo Total", "Consumo_Total_MXN")
    png["box_servicio"] = _png(
        vu.plot_box, df["Tiempo_Servicio_Min"],
        "Boxplot — Tiempo de Servicio", "Tiempo_Servicio_Min")
    png["box_consumo_tipo"] = _png(
        vu.plot_box_by_group, df, "Consumo_Total_MXN", "Tipo_Cliente",
        "Consumo por tipo de cliente", "Consumo_Total_MXN")
    png["bars_tipo"] = _png(
        vu.plot_bars, df, "Tipo_Cliente", "Conteo — Tipo de Cliente")
    png["bars_pago"] = _png(
        vu.plot_bars, df, "Metodo_Pago", "Conteo — Método de Pago")

    # ---------------- Unidad 2: distribuciones ----------------
    png["ajuste_espera"] = _png(
        md.plot_ajuste_con_qq, df["Tiempo_Espera_Min"],
        dist="expon", title_prefix="Tiempo de Espera")
    png["ajuste_servicio"] = _png(
        md.plot_ajuste_con_qq, clean["Tiempo_Servicio_Min"],
        dist="norm", title_prefix="Tiempo de Servicio")

    try:
        fig_ph, fig_pq, res_pois = md.simular_poisson_llegadas(df)
    except Exception:
        fig_ph, fig_pq, res_pois = None, None, {}
    png["poisson_hist"] = _fig_a_png(fig_ph)
    png["poisson_qq"] = _fig_a_png(fig_pq)
    ctx["poisson"] = res_pois
    try:
        ctx["bondad"] = md.tabla_bondad_ajuste(df, clean)
    except Exception:
        ctx["bondad"] = pd.DataFrame()

    # ---------------- Unidad 3: hipótesis + clasificación ----------------
    res_hip: dict = {}
    grupos = [g for g in ("Estudiante", "Profesor")
              if g in set(df["Tipo_Cliente"].dropna().astype(str))]
    if len(grupos) == 2:
        try:
            res_hip = mh.comparar_dos_grupos(
                df, "Consumo_Total_MXN", "Tipo_Cliente", grupos[0], grupos[1])
        except Exception as exc:
            res_hip = {"error": f"No se pudo comparar los grupos: {exc}"}
        png["hip_box"] = _png(
            mh.plot_comparacion_grupos, df, "Consumo_Total_MXN", "Tipo_Cliente",
            grupos[0], grupos[1],
            f"Consumo_Total_MXN — {grupos[0]} vs {grupos[1]}")
    ctx["hip"] = res_hip
    ctx["hip_grupos"] = grupos

    try:
        res_logreg = mh.entrenar_regresion_logistica(df, target_col="Es_Hora_Pico")
    except Exception as exc:
        res_logreg = {"error": f"No se pudo entrenar la regresión logística: {exc}"}
    ctx["logreg"] = res_logreg
    if not res_logreg.get("error"):
        png["logreg_cm"] = _fig_a_png(res_logreg.get("fig_cm"))
        png["logreg_roc"] = _fig_a_png(res_logreg.get("fig_roc"))

    # ---------------- Unidad 4: bayesiana ----------------
    try:
        ctx["bayes"] = _calcular_bayes(df)
    except Exception:
        ctx["bayes"] = {}
    if ctx["bayes"]:
        png["bayes_barras"] = _png(
            mb.plot_bayes_barras, ctx["bayes"]["resultado"],
            ctx["bayes"]["etiqueta_a"], ctx["bayes"]["etiqueta_b"])

    try:
        res_nb = mb.entrenar_naive_bayes(df)
    except Exception as exc:
        res_nb = {"error": f"No se pudo entrenar Naive Bayes: {exc}"}
    ctx["nb"] = res_nb
    if not res_nb.get("error"):
        png["nb_cm"] = _fig_a_png(res_nb.get("fig_cm"))
        png["nb_proba"] = _fig_a_png(res_nb.get("fig_proba"))

    try:
        res_cmp = mb.comparar_modelos(df)
    except Exception as exc:
        res_cmp = {"error": f"No se pudo comparar los modelos: {exc}"}
    ctx["comparacion"] = res_cmp
    if not res_cmp.get("error"):
        png["modelos_bar"] = _fig_a_png(res_cmp.get("fig"))

    ctx.update(_contexto_opcional(df, clean, ctx.get("bayes") or {}))
    ctx["png"] = png
    for clave in ("corr_png", "violin_png", "arbol_bayes_png"):
        if ctx.get(clave):
            png[clave] = ctx[clave]
    return ctx


def _contexto_opcional(df: pd.DataFrame, clean: pd.DataFrame,
                       bayes: dict | None = None) -> dict:
    """Métricas de la Fase 2; se activan solas cuando existan las funciones."""
    extra: dict = {}
    extra["media_ponderada"] = None
    extra["corr"] = None
    extra["corr_png"] = None
    extra["violin_png"] = None
    extra["arbol_bayes_png"] = None
    extra["sims"] = {}
    extra["anova"] = None
    extra["chi2"] = None

    if hasattr(su, "media_ponderada"):
        try:
            extra["media_ponderada"] = su.media_ponderada(
                clean, "Consumo_Total_MXN", "Tiempo_Servicio_Min")
        except Exception:
            extra["media_ponderada"] = None

    if hasattr(su, "matriz_correlacion"):
        try:
            extra["corr"] = su.matriz_correlacion(clean)
        except Exception:
            extra["corr"] = None
    if hasattr(vu, "plot_matriz_correlacion"):
        extra["corr_png"] = _png(vu.plot_matriz_correlacion, clean)
    if hasattr(vu, "plot_violin"):
        extra["violin_png"] = _png(
            vu.plot_violin, df, "Consumo_Total_MXN", "Tipo_Cliente",
            "Consumo por tipo de cliente (violín)", "Consumo_Total_MXN")
    if bayes and hasattr(vu, "plot_arbol_bayes"):
        extra["arbol_bayes_png"] = _png(
            vu.plot_arbol_bayes, bayes.get("resultado") or {},
            bayes.get("etiqueta_a", "A"), bayes.get("etiqueta_b", "B"))

    if hasattr(mh, "anova"):
        try:
            extra["anova"] = mh.anova(df, "Consumo_Total_MXN", "Tipo_Cliente")
        except Exception:
            extra["anova"] = None
    if hasattr(mh, "chi_cuadrado"):
        try:
            extra["chi2"] = mh.chi_cuadrado(df, "Tipo_Cliente", "Es_Hora_Pico")
        except Exception:
            extra["chi2"] = None

    simulaciones = (
        ("bernoulli", "simular_bernoulli", {"p": 0.5, "n": 1000}),
        ("binomial", "simular_binomial", {"n_ensayos": 10, "p": 0.5, "n": 1000}),
        ("geometrica", "simular_geometrica", {"p": 0.3, "n": 1000}),
        ("hipergeometrica", "simular_hipergeometrica", {"N": 200, "K": 60, "n": 10}),
        ("gamma", "simular_gamma", {"shape": 2.0, "scale": 1.5, "n": 1000}),
    )
    for clave, nombre_fn, kwargs in simulaciones:
        fn = getattr(md, nombre_fn, None)
        if fn is None:
            continue
        try:
            r = fn(**kwargs)
        except Exception:
            continue
        if not isinstance(r, dict):
            continue
        extra["sims"][clave] = {
            "fig": _fig_a_png(r.get("fig")),
            "resumen": r.get("resumen") or {},
        }
    return extra


# ============================================================
# FASE 2: TEXTOS CON IA (paralelo, con respaldo determinista)
# ============================================================
def _ejecutar_ia_paralelo(tareas: list, fallbacks: dict) -> dict:
    """Lanza las llamadas a Gemini en paralelo y tolera fallos.

    Si Gemini no está configurado (sin API key, sin red, sin secrets) el
    módulo `mod_ia_analista` devuelve el texto de respaldo, así que el
    reporte SIEMPRE se genera con contenido.
    """
    resultados = {
        nombre: {"tecnico": fallbacks.get(nombre, ""), "simple": None}
        for nombre, _, _ in tareas
    }
    with ThreadPoolExecutor(max_workers=4) as executor:
        futuros = {
            executor.submit(analizar_seccion_ia, *args, **kwargs): nombre
            for nombre, args, kwargs in tareas
        }
        for fut in as_completed(futuros):
            nombre = futuros[fut]
            try:
                r = fut.result()
            except Exception:
                continue
            if isinstance(r, dict) and r.get("tecnico"):
                resultados[nombre] = r
    return resultados


def _preparar_textos(ctx: dict) -> dict:
    """Métricas por sección, textos de respaldo y tareas de IA."""
    k = ctx["kpis"]
    png = ctx["png"]
    hip = ctx["hip"] or {}
    logreg = ctx["logreg"] or {}
    bayes = ctx["bayes"] or {}
    nb = ctx["nb"] or {}
    comparacion = ctx["comparacion"] or {}
    poisson = ctx["poisson"] or {}
    desc = ctx["desc"]
    tabla_cmp = comparacion.get("tabla")
    mejor_modelo = (
        str(tabla_cmp["Modelo"].iloc[0])
        if tabla_cmp is not None and len(tabla_cmp) else "N/A"
    )

    datos_resumen = {
        "registros_totales": k["n_total"],
        "registros_limpios": k["n_clean"],
        "outliers": k["outliers"],
        "consumo_promedio_mxn": round(k["consumo_prom"], 2),
        "consumo_total_mxn": round(k["consumo_total"], 2),
        "espera_promedio_min": round(k["espera_prom"], 2),
        "espera_maxima_min": round(k["espera_max"], 2),
        "servicio_promedio_min": round(k["servicio_prom"], 2),
        "satisfaccion_promedio": round(k["sat_prom"], 2),
        "porcentaje_hora_pico": round(k["pct_pico"], 2),
    }

    fallbacks = {
        "resumen": (
            f"El estudio analiza {k['n_total']} registros de la cafetería en el "
            f"horario 7:00-11:00 AM. El consumo promedio es de "
            f"${k['consumo_prom']:.2f} MXN, la espera promedio de "
            f"{k['espera_prom']:.2f} minutos y la satisfacción media de "
            f"{k['sat_prom']:.2f}/5. El {k['pct_pico']:.1f}% de los clientes "
            f"llega en hora pico y se detectaron {k['outliers']} registros "
            f"atípicos mediante el criterio IQR."
        ),
        "desc": (
            f"Las cinco variables numéricas presentan dispersión moderada: la "
            f"espera promedio es {k['espera_prom']:.2f} min con un máximo de "
            f"{k['espera_max']:.2f} min, señal de que la cola tiene episodios "
            f"de saturación. La satisfacción media ({k['sat_prom']:.2f}/5) se "
            f"ubica por encima del punto neutro, lo que sugiere un servicio "
            f"aceptable pero con margen de mejora en los picos."
        ),
        "dist": (
            f"El ajuste exponencial del tiempo de espera arroja λ = "
            f"{_fmt(poisson.get('lambda'), 4)} llegadas por minuto, coherente "
            f"con un proceso de arribos aleatorio tipo Poisson. El tiempo de "
            f"servicio se aproxima a una normal con la media y desviación "
            f"observadas; el contraste KS permite decidir si el supuesto se "
            f"sostiene al nivel α = 0.05."
        ),
        "hip": (
            f"La comparación de grupos produce t = {_fmt(hip.get('ttest_stat'), 3)} "
            f"(p = {_fmt(hip.get('ttest_p'), 4)}) y U de Mann-Whitney "
            f"{_fmt(hip.get('mannwhitney_u'), 1)} "
            f"(p = {_fmt(hip.get('mannwhitney_p'), 4)}). El tamaño del efecto "
            f"de Cohen es {_fmt(hip.get('cohen_d'), 3)} "
            f"({hip.get('interpretacion_d', 'N/A')}). El modelo logístico "
            f"alcanza accuracy de {_fmt((logreg.get('metricas') or {}).get('accuracy'), 3)} "
            f"y AUC de {_fmt((logreg.get('metricas') or {}).get('auc'), 3)}."
        ),
        "bayes": (
            f"Actualizando la probabilidad de que un cliente sea Profesor con "
            f"la evidencia de que paga con Tarjeta, el prior "
            f"{_fmt((bayes.get('resultado') or {}).get('p_a'), 3)} se convierte "
            f"en posterior "
            f"{_fmt((bayes.get('resultado') or {}).get('p_a_dado_b'), 3)}. "
            f"El clasificador Naive Bayes obtiene accuracy "
            f"{_fmt((nb.get('metricas') or {}).get('accuracy'), 3)} y el "
            f"mejor modelo de la comparación es {mejor_modelo}."
        ),
        "concl": (
            f"Con {k['n_total']} observaciones el proyecto confirma que la "
            f"afluencia sigue un patrón predecible: la hora pico concentra el "
            f"{k['pct_pico']:.1f}% de las llegadas y explica los picos de "
            f"espera. Se recomienda reforzar personal entre 8:00 y 9:00, "
            f"habilitar prepago por aplicación y monitorear la satisfacción "
            f"como indicador de calidad del servicio."
        ),
    }

    return _preparar_tareas_ia(ctx, fallbacks, datos_resumen, png, desc)


def _preparar_tareas_ia(ctx, fallbacks, datos_resumen, png, desc) -> dict:
    """Arma las tareas de Gemini (sección, métricas, respaldo, imagen)."""
    k = ctx["kpis"]
    hip = ctx["hip"] or {}
    logreg = ctx["logreg"] or {}
    bayes = ctx["bayes"] or {}
    nb = ctx["nb"] or {}
    comparacion = ctx["comparacion"] or {}
    poisson = ctx["poisson"] or {}
    bondad = ctx["bondad"]

    datos_desc = {
        "tabla_descriptiva": desc.round(3).to_dict("records"),
        "outliers_detectados": k["outliers"],
        "porcentaje_hora_pico": round(k["pct_pico"], 2),
        "frecuencia_tipo_cliente": ctx["freq_tipo"].to_dict("records"),
        "frecuencia_metodo_pago": ctx["freq_pago"].to_dict("records"),
    }
    datos_dist = {
        "poisson": {clave: (round(val, 4) if isinstance(val, float) else val)
                    for clave, val in poisson.items()},
        "bondad_ajuste": bondad.round(4).to_dict("records") if not bondad.empty else [],
    }
    datos_hip = {
        "prueba_grupos": {
            clave: (round(val, 4) if isinstance(val, float) else val)
            for clave, val in hip.items() if clave != "error"
        },
        "regresion_logistica": logreg.get("metricas") or {},
    }
    datos_bayes = {
        "prior_empirico": round(bayes.get("p_a_empirico", 0.0), 4),
        "verosimilitud_a": round(bayes.get("p_b_dado_a_empirico", 0.0), 4),
        "verosimilitud_no_a": round(bayes.get("p_b_dado_no_a_empirico", 0.0), 4),
        "posterior": round((bayes.get("resultado") or {}).get("p_a_dado_b", 0.0), 4),
        "naive_bayes": nb.get("metricas") or {},
        "comparacion_modelos": comparacion.get("friedman") or {},
    }

    tareas = [
        ("resumen",
         ("Resumen Ejecutivo", datos_resumen, fallbacks["resumen"]),
         {"imagen_png_bytes": png.get("hist_consumo"),
          "instruccion_extra": (
              "Sintetiza los hallazgos del proyecto en 4-6 oraciones: perfil "
              "del cliente, consumo, espera, satisfacción y hora pico."),
          "doble_capa": True}),
        ("desc",
         ("Estadística Descriptiva (U1)", datos_desc, fallbacks["desc"]),
         {"imagen_png_bytes": png.get("hist_espera"),
          "instruccion_extra": (
              "Interpreta media, mediana, dispersión, asimetría aparente y "
              "outliers. Menciona qué variable requiere atención operativa."),
          "doble_capa": True}),
        ("dist",
         ("Distribuciones de Probabilidad (U2)", datos_dist, fallbacks["dist"]),
         {"imagen_png_bytes": png.get("ajuste_espera"),
          "instruccion_extra": (
              "Evalúa si Poisson explica las llegadas y si Exponencial/Normal "
              "son plausibles para espera y servicio según el p-value de KS."),
          "doble_capa": True}),
        ("hip",
         ("Pruebas de Hipótesis (U3)", datos_hip, fallbacks["hip"]),
         {"imagen_png_bytes": png.get("logreg_roc"),
          "instruccion_extra": (
              "Redacta la conclusión estadística (rechaza o no rechaza H0) "
              "para t, Mann-Whitney y el modelo logístico."),
          "doble_capa": True}),
        ("bayes",
         ("Estadística Bayesiana (U4)", datos_bayes, fallbacks["bayes"]),
         {"imagen_png_bytes": png.get("bayes_barras"),
          "instruccion_extra": (
              "Explica la actualización prior→posterior y la calidad de "
              "calibración de Naive Bayes (log-loss, Brier)."),
          "doble_capa": True}),
        ("concl",
         ("Conclusiones y Recomendaciones", datos_resumen, fallbacks["concl"]),
         {"instruccion_extra": (
             "Cierra con 3 conclusiones y 3 recomendaciones accionables para "
             "la cafetería, sin repetir cifras ya dadas en el resumen."),
          "doble_capa": True}),
    ]
    return _ejecutar_ia_paralelo(tareas, fallbacks)


def _texto(textos: dict, clave: str) -> tuple[str, str | None]:
    """Devuelve (tecnico, simple) de una sección, tolerando faltantes."""
    valor = textos.get(clave) or {}
    return valor.get("tecnico") or "", valor.get("simple")


# ============================================================
# FRONT MATTER: RESUMEN EJECUTIVO (sin numeración)
# ============================================================
def _cap_resumen_ejecutivo(pdf, ctx: dict, textos: dict):
    """Se dibuja después de los índices y antes del capítulo 1."""
    k = ctx["kpis"]
    pdf.cap_actual = "Resumen Ejecutivo"
    pdf.indice_caps["Resumen Ejecutivo"] = (pdf.page_no() or 0) + 1
    if pdf.page_no() > 0:
        pdf.add_page()
    pdf.subtitulo("Resumen Ejecutivo")
    pdf.nota_doble(*_texto(textos, "resumen"))

    pdf.subtitulo("Indicadores clave del proyecto")
    pdf.caption_tabla("Indicadores clave del proyecto")
    pdf.tabla_metrica([
        ("Registros en la base de datos", _fmt(k["n_total"])),
        ("Registros válidos tras limpieza", _fmt(k["n_clean"])),
        ("Registros atípicos (IQR)", _fmt(k["outliers"])),
        ("Consumo promedio", f"${k['consumo_prom']:.2f} MXN"),
        ("Consumo acumulado", f"${k['consumo_total']:,.2f} MXN"),
        ("Tiempo de espera promedio", f"{k['espera_prom']:.2f} min"),
        ("Tiempo de espera máximo", f"{k['espera_max']:.2f} min"),
        ("Tiempo de servicio promedio", f"{k['servicio_prom']:.2f} min"),
        ("Satisfacción promedio", f"{k['sat_prom']:.2f} / 5"),
        ("Clientes en hora pico", _pct(k["pct_pico"])),
    ])

    pdf.subtitulo("Estructura del documento")
    pdf.parrafo(
        "El reporte se organiza en diez capítulos. Los primeros cuatro "
        "documentan el problema, los objetivos, el marco conceptual y el "
        "proceso de adquisición de datos. Los capítulos 5 a 8 desarrollan las "
        "cuatro unidades del curso (estadística descriptiva, distribuciones de "
        "probabilidad, pruebas de hipótesis y estadística bayesiana). Los "
        "capítulos 9 y 10 cierran con conclusiones, recomendaciones y "
        "referencias bibliográficas."
    )
    pdf.preguntas_guia([
        "¿Los indicadores del reporte coinciden con lo que observa el personal "
        "de la cafetería?",
        "¿Qué decisión operativa se mantendría aunque empeoraran los "
        "indicadores?",
        "¿Qué dato adicional haría falta para explicar la satisfacción?",
    ])


# ============================================================
# CAPÍTULOS
# ============================================================
def _cap_1(pdf, ctx: dict, textos: dict):
    pdf.titulo_capitulo("1", "Definición del Problema")

    pdf.subtitulo("1.1 Contexto")
    pdf.parrafo(
        "La cafetería universitaria atiende a estudiantes, profesores y "
        "personal administrativo en una ventana operativa de cuatro horas "
        "(7:00 a 11:00 AM). En ese lapso la afluencia no es constante: se "
        "concentra en la hora pico de 8:00 a 9:00 AM, cuando coinciden la "
        "entrada de turnos escolares y los recesos del personal. La operación "
        "se apoya en un barista y una caja, por lo que cualquier variación "
        "brusca de la demanda se traduce de inmediato en filas y en "
        "percepción de mala calidad."
    )

    pdf.subtitulo("1.2 Situación problemática")
    pdf.parrafo(
        "La administración de la cafetería toma decisiones de personal, "
        "inventario y promociones con base en la observación empírica del "
        "encargado. No existe un análisis formal que cuantifique cuánto se "
        "espera, cuánto se consume ni cómo se relaciona el tiempo de servicio "
        "con la satisfacción del cliente. Esto tiene tres consecuencias "
        "concretas: (1) la asignación de personal no está alineada con la "
        "demanda real, (2) no se conocen los límites naturales de la cola para "
        "fijar metas de atención, y (3) no hay evidencia numérica para evaluar "
        "cambios operativos ya implementados."
    )

    pdf.subtitulo("1.3 Pregunta de investigación")
    pdf.parrafo(
        "¿Qué características de la demanda y del servicio de la cafetería "
        "universitaria permiten explicar el tiempo de espera, el consumo y la "
        "satisfacción de los clientes durante el horario matutino, y qué "
        "modelos estadísticos describen y predicen mejor ese comportamiento?"
    )

    pdf.subtitulo("1.4 Justificación")
    pdf.parrafo(
        "El proyecto se justifica en tres planos. En lo académico, integra las "
        "cuatro unidades de Modelos Estadísticos para el Aprendizaje "
        "Automático: estadística descriptiva, distribuciones de probabilidad, "
        "pruebas de hipótesis y estadística bayesiana. En lo metodológico, "
        "construye un flujo completo de captura, limpieza, análisis y "
        "comunicación de resultados sobre datos reales. En lo operativo, "
        "entrega a la cafetería indicadores y modelos que apoyan decisiones de "
        "turnos, inventario y atención al cliente."
    )

    pdf.subtitulo("1.5 Alcance y limitaciones")
    pdf.parrafo(
        "El alcance cubre el registro de clientes atendidos entre 7:00 y 11:00 "
        "AM, con nueve variables: identificador, hora de llegada, tiempo de "
        "espera, tiempo de servicio, tipo de cliente, método de pago, consumo, "
        "satisfacción y marca de hora pico. Quedan fuera del alcance el "
        "inventario por producto, los costos de operación y la medición de "
        "demanda de clientes que se retiran sin comprar. La muestra es de tipo "
        "conveniencia y su tamaño condiciona la potencia de las pruebas "
        "estadísticas."
    )


def _cap_2(pdf, ctx: dict, textos: dict):
    pdf.titulo_capitulo("2", "Planteamiento de Objetivos")

    pdf.subtitulo("2.1 Objetivo general")
    pdf.parrafo(
        "Analizar el comportamiento de la demanda y del servicio de la "
        "cafetería universitaria durante el horario matutino mediante técnicas "
        "de estadística descriptiva, inferencial y bayesiana, para generar "
        "evidencia cuantitativa que sustente decisiones operativas."
    )

    pdf.subtitulo("2.2 Objetivos específicos")
    objetivos = [
        "Describir las variables numéricas del servicio (espera, servicio, "
        "consumo, satisfacción y hora de llegada) con medidas de tendencia "
        "central, dispersión y posición.",
        "Identificar registros atípicos mediante el criterio del rango "
        "intercuartílico y evaluar su efecto en los indicadores.",
        "Ajustar distribuciones de probabilidad (Poisson, Exponencial y "
        "Normal) a las variables del proceso y validar el ajuste con la prueba "
        "de Kolmogorov-Smirnov.",
        "Contrastar hipótesis sobre diferencias de consumo entre tipos de "
        "cliente con pruebas paramétricas y no paramétricas.",
        "Construir un modelo de clasificación que prediga la llegada en hora "
        "pico y evaluar su desempeño con matriz de confusión y curva ROC.",
        "Actualizar probabilidades con el teorema de Bayes y comparar "
        "clasificadores con validación cruzada y prueba de Friedman.",
        "Comunicar los resultados en un reporte institucional reproducible que "
        "pueda regenerarse con los datos vigentes.",
    ]
    for i, obj in enumerate(objetivos, 1):
        pdf.parrafo(f"{i}. {obj}")

    pdf.subtitulo("2.3 Vinculación con las unidades del curso")
    pdf.caption_tabla("Objetivos específicos por unidad del curso")
    pdf.tabla_metrica([
        ("Unidad 1 — Estadística descriptiva", "Objetivos 1 y 2"),
        ("Unidad 2 — Distribuciones de probabilidad", "Objetivo 3"),
        ("Unidad 3 — Pruebas de hipótesis", "Objetivos 4 y 5"),
        ("Unidad 4 — Estadística bayesiana", "Objetivo 6"),
        ("Integración y comunicación", "Objetivo 7"),
    ], ancho1=95, ancho2=70, encabezados=("Unidad", "Objetivos cubiertos"))


def _cap_3(pdf, ctx: dict, textos: dict):
    pdf.titulo_capitulo("3", "Marco Conceptual")

    pdf.subtitulo("3.1 Estadística descriptiva")
    pdf.parrafo(
        "La estadística descriptiva resume un conjunto de observaciones sin "
        "suponer un modelo probabilístico. La media aritmética sintetiza el "
        "valor típico, pero es sensible a valores extremos; la mediana y la "
        "moda son medidas robustas de posición. La varianza y la desviación "
        "estándar cuantifican la dispersión, mientras que los percentiles "
        "describen la forma de la distribución. El rango intercuartílico "
        "(IQR = Q3 − Q1) define los límites Q1 − 1.5·IQR y Q3 + 1.5·IQR que se "
        "usan en este proyecto para marcar observaciones atípicas."
    )

    pdf.subtitulo("3.2 Distribuciones de probabilidad")
    pdf.parrafo(
        "Un modelo probabilístico asigna probabilidades a los valores de una "
        "variable aleatoria. La distribución de Poisson describe conteos de "
        "eventos en un intervalo fijo de tiempo, por lo que modela las "
        "llegadas por minuto con parámetro λ. La distribución Exponencial "
        "modela el tiempo entre eventos de un proceso Poisson y se usa para el "
        "tiempo de espera. La distribución Normal describe variables continuas "
        "simétricas y se propone para el tiempo de servicio. La prueba de "
        "Kolmogorov-Smirnov compara la distribución empírica acumulada con la "
        "teórica: si p > 0.05 no se rechaza el ajuste propuesto."
    )

    pdf.subtitulo("3.3 Inferencia y pruebas de hipótesis")
    pdf.parrafo(
        "Una prueba de hipótesis contrasta una hipótesis nula H0 contra una "
        "alternativa H1 con un nivel de significancia α. La prueba t de "
        "Student compara medias de dos grupos independientes y supone "
        "normalidad y homocedasticidad; cuando estos supuestos no se cumplen "
        "se recurre a Mann-Whitney U, que compara rangos y no requiere "
        "normalidad. El tamaño del efecto de Cohen mide la magnitud práctica "
        "de la diferencia, independientemente de la significancia estadística. "
        "Para más de dos grupos se emplea ANOVA y, para asociación entre "
        "variables categóricas, la prueba chi-cuadrado de independencia."
    )

    pdf.subtitulo("3.4 Estadística bayesiana y aprendizaje automático")
    pdf.parrafo(
        "El teorema de Bayes actualiza una probabilidad a priori P(A) con la "
        "evidencia de un evento B: P(A|B) = P(B|A)·P(A) / P(B). El resultado es "
        "la probabilidad posterior, que combina creencia previa y datos. El "
        "clasificador Naive Bayes aplica este principio suponiendo "
        "independencia condicional entre predictores. La regresión logística "
        "modela la probabilidad de una clase mediante una función logística y "
        "se evalúa con accuracy, precision, recall, F1 y AUC-ROC. La validación "
        "cruzada estratificada y la prueba de Friedman permiten comparar varios "
        "modelos de forma justa."
    )

    pdf.caption_tabla("Conceptos clave y su aplicación en el proyecto")
    pdf.tabla_metrica([
        ("Media, mediana y desviación estándar", "Perfil del servicio"),
        ("Rango intercuartílico (IQR)", "Detección de outliers"),
        ("Poisson(λ)", "Llegadas por minuto"),
        ("Exponencial(λ)", "Tiempo de espera"),
        ("Normal(μ, σ)", "Tiempo de servicio"),
        ("t de Student / Mann-Whitney U", "Comparación de grupos"),
        ("Regresión logística", "Predicción de hora pico"),
        ("Teorema de Bayes / Naive Bayes", "Actualización de probabilidades"),
    ], ancho1=95, ancho2=70,
        encabezados=("Concepto", "Aplicación en la cafetería"))


def _cap_4(pdf, ctx: dict, textos: dict):
    pdf.titulo_capitulo("4", "Adquisición y Análisis de Datos")

    k = ctx["kpis"]

    pdf.subtitulo("4.1 Fuente de los datos")
    pdf.parrafo(
        "Los datos provienen de la aplicación web desarrollada para este "
        "proyecto. El formulario de captura valida cada registro en el momento "
        "de guardarlo (hora entre 7.00 y 11.00, tiempos positivos, catálogos "
        "cerrados para tipo de cliente y método de pago, consumo no negativo y "
        "satisfacción entre 1 y 5) y lo almacena en una tabla de Supabase. La "
        "base de datos es la fuente única de verdad: el reporte y el panel de "
        "análisis leen siempre el estado vigente, por lo que cualquier registro "
        "nuevo se refleja de inmediato."
    )
    pdf.parrafo(
        f"La base contiene {_fmt(k['n_total'])} registros con nueve variables. "
        "El periodo observado corresponde al horario matutino de la cafetería "
        "(7:00 a 11:00 AM), que es cuando se concentra la demanda y se producen "
        "las filas más largas."
    )

    pdf.subtitulo("4.2 Diccionario de variables")
    pdf.caption_tabla("Diccionario de variables del conjunto de datos")
    diccionario = pd.DataFrame([
        ("ID_Cliente", "Entero", "Identificador único e incremental del cliente atendido."),
        ("Hora_Llegada", "Decimal", "Hora de llegada en formato decimal (8.25 = 8:15 AM)."),
        ("Tiempo_Espera_Min", "Decimal", "Minutos transcurridos antes de ser atendido."),
        ("Tiempo_Servicio_Min", "Decimal", "Minutos de atención del barista."),
        ("Tipo_Cliente", "Texto", "Estudiante, Profesor o Administrativo."),
        ("Metodo_Pago", "Texto", "Efectivo, Tarjeta o App; admite nulo."),
        ("Consumo_Total_MXN", "Decimal", "Monto gastado en pesos mexicanos."),
        ("Satisfaccion_1_al_5", "Entero", "Calificación del servicio de 1 a 5; admite nulo."),
        ("Es_Hora_Pico", "Texto", "Sí cuando la llegada ocurre entre 8:00 y 9:00 AM."),
    ], columns=["Variable", "Tipo", "Descripción"])
    pdf.tabla_df(diccionario, widths=[26, 13, 61], font_size=7.5)

    pdf.subtitulo("4.3 Calidad de los datos")
    pdf.parrafo(
        "Antes de cualquier análisis se revisa la integridad del conjunto: "
        "valores nulos, duplicados y registros atípicos. Los nulos se "
        "concentran en las variables que el formulario permite dejar vacías "
        "(método de pago y satisfacción), mientras que los atípicos "
        "corresponden, en su mayoría, a errores de captura intencionales "
        "(tiempos de servicio negativos o consumos fuera de rango)."
    )
    pdf.caption_tabla("Resumen de calidad de los datos")
    filas_calidad = [
        ("Registros totales", _fmt(k["n_total"])),
        ("Registros válidos tras limpieza", _fmt(k["n_clean"])),
        ("Registros descartados por limpieza", _fmt(k["n_total"] - k["n_clean"])),
        ("Registros duplicados exactos", _fmt(ctx["n_duplicados"])),
    ]
    for variable, pct in ctx["nulos"].items():
        filas_calidad.append((f"% nulos en {variable}", _pct(pct, 2)))
    pdf.tabla_metrica(filas_calidad)

    pdf.subtitulo("4.4 Tratamiento aplicado")
    pdf.parrafo(
        "La limpieza elimina los registros con tiempo de servicio menor o igual "
        "a cero y los consumos superiores a 250 pesos, que corresponden a "
        "errores de captura deliberados para la práctica. Sobre el conjunto "
        "original se aplica además el criterio del rango intercuartílico para "
        "documentar cuántos valores atípicos existen y en qué variables se "
        "concentran; el detalle se presenta en el capítulo 5."
    )

    pdf.agregar_figura(ctx["png"].get("bars_tipo"), ancho=130)
    pdf.caption_figura("Distribución de clientes por tipo (datos originales)")
    pdf.preguntas_guia([
        "¿El formulario de captura impide todos los errores posibles?",
        "¿Qué variable adicional convendría registrar para explicar la "
        "satisfacción?",
    ])


def _cap_5(pdf, ctx: dict, textos: dict):
    pdf.titulo_capitulo("5", "Estadística Descriptiva (Unidad 1)")
    k = ctx["kpis"]
    png = ctx["png"]

    pdf.subtitulo("5.1 Procedimiento")
    pdf.parrafo(
        "Para cada variable numérica se calculan n, media, mediana, moda, "
        "varianza, desviación estándar, mínimo, máximo y los percentiles 25, "
        "50 y 75. Los cálculos se repiten sobre el conjunto original y sobre el "
        "conjunto depurado para medir cuánto influyen los registros atípicos. "
        "Las variables categóricas se resumen con tablas de frecuencia absoluta "
        "y relativa."
    )

    pdf.subtitulo("5.2 Estadísticos descriptivos")
    pdf.caption_tabla("Estadísticos descriptivos del conjunto original")
    pdf.tabla_df(
        ctx["desc"],
        cols=["Variable", "n", "Media", "Mediana", "Moda", "Desv_Std",
              "Min", "P25", "P75", "Max"],
        widths=[24, 8, 12, 12, 10, 12, 10, 9, 9, 10],
        font_size=7,
    )
    pdf.parrafo(
        f"El tiempo de espera promedio es de {k['espera_prom']:.2f} minutos con "
        f"un máximo observado de {k['espera_max']:.2f} minutos. El consumo "
        f"promedio por cliente es de ${k['consumo_prom']:.2f} MXN y la "
        f"satisfacción media se ubica en {k['sat_prom']:.2f} sobre 5. La "
        "diferencia entre media y mediana en consumo y espera indica asimetría "
        "positiva: existen pocos clientes con valores muy altos que empujan el "
        "promedio hacia arriba."
    )

    comparacion = ctx["desc"][["Variable", "Media"]].merge(
        ctx["desc_clean"][["Variable", "Media"]], on="Variable", how="left",
        suffixes=(" original", " depurada"))
    comparacion["Diferencia"] = (
        comparacion["Media depurada"] - comparacion["Media original"]).round(3)
    pdf.caption_tabla("Efecto de la depuración sobre la media")
    pdf.tabla_df(comparacion, widths=[34, 22, 22, 22], font_size=7.5)
    pdf.parrafo(
        f"La depuración descarta {_fmt(k['n_total'] - k['n_clean'])} registros y "
        f"deja {_fmt(k['n_clean'])} observaciones válidas. Las diferencias en "
        "las medias son pequeñas, lo que confirma que los atípicos son pocos "
        "pero suficientes para inflar la dispersión."
    )

    ponderada = ctx.get("media_ponderada")
    if ponderada is not None:
        pdf.caption_tabla("Media ponderada del consumo por tiempo de servicio")
        pdf.tabla_metrica([
            ("Media simple del consumo (MXN)", _fmt(k["consumo_prom"], 3)),
            ("Media ponderada por tiempo de servicio (MXN)",
             _fmt(ponderada, 3)),
        ])
        pdf.parrafo(
            "La media ponderada usa el tiempo de servicio como peso: cada "
            "cliente aporta en proporción al tiempo que ocupó la caja. Si el "
            "resultado supera a la media simple, los clientes atendidos más "
            "lentamente consumen por encima del promedio y el indicador simple "
            "subestima la carga económica de la fila."
        )

    pdf.subtitulo("5.3 Distribuciones de frecuencia")
    pdf.caption_tabla("Frecuencia por tipo de cliente")
    pdf.tabla_df(ctx["freq_tipo"], widths=[40, 20, 20], font_size=7.5)
    pdf.caption_tabla("Frecuencia por método de pago")
    pdf.tabla_df(ctx["freq_pago"], widths=[40, 20, 20], font_size=7.5)
    pdf.agregar_figura(png.get("bars_pago"), ancho=125)
    pdf.caption_figura("Conteo de clientes por método de pago")

    pdf.subtitulo("5.4 Valores atípicos")
    pdf.caption_tabla("Detección de atípicos por el criterio IQR")
    pdf.tabla_df(ctx["out_summary"], widths=[30, 12, 12, 11, 15, 15, 12, 12],
                 font_size=7)
    pdf.parrafo(
        f"El criterio Q1 − 1.5·IQR y Q3 + 1.5·IQR marca {_fmt(k['outliers'])} "
        "registros con al menos una variable fuera de rango. El detalle con la "
        "columna «Outliers en» permite identificar si el problema es de captura "
        "(tiempos negativos, consumos de miles de pesos) o de operación real."
    )
    pdf.tabla_df(ctx["out_detail"],
                 cols=["ID_Cliente", "Hora_Llegada", "Tiempo_Espera_Min",
                       "Tiempo_Servicio_Min", "Consumo_Total_MXN", "Outliers en"],
                 widths=[14, 15, 17, 18, 18, 30], font_size=7, max_rows=12)

    pdf.subtitulo("5.5 Análisis de correlación")
    if ctx.get("corr") is not None:
        corr = ctx["corr"].round(3).reset_index()
        pdf.caption_tabla("Matriz de correlación de Pearson")
        pdf.tabla_df(corr, font_size=7)
        pdf.agregar_figura(png.get("corr_png"), ancho=140)
        pdf.caption_figura("Mapa de calor de la matriz de correlación")
    else:
        pdf.nota_pendiente(
            "La matriz de correlación se habilitará cuando se agregue "
            "stats_utils.matriz_correlacion.")

    pdf.subtitulo("5.6 Lectura estadística")
    pdf.nota_doble(*_texto(textos, "desc"))

    if png.get("violin_png"):
        pdf.agregar_figura(png.get("violin_png"), ancho=130)
        pdf.caption_figura("Distribución del consumo por tipo de cliente")
    pdf.agregar_figura(png.get("box_consumo_tipo"), ancho=130)
    pdf.caption_figura("Diagrama de caja del consumo por tipo de cliente")
    pdf.preguntas_guia([
        "¿Qué variable concentra la mayor dispersión y por qué?",
        "¿Qué haría con los atípicos antes de calcular promedios oficiales?",
        "¿La satisfacción depende más de la espera o del tiempo de servicio?",
    ])


def _cap_6(pdf, ctx: dict, textos: dict):
    pdf.titulo_capitulo("6", "Distribuciones de Probabilidad (Unidad 2)")
    png = ctx["png"]
    pois = ctx.get("poisson") or {}
    bondad = ctx.get("bondad")

    pdf.subtitulo("6.1 Modelos propuestos")
    pdf.parrafo(
        "Se proponen tres modelos. Para las llegadas se usa una distribución de "
        "Poisson con parámetro λ estimado como la media de llegadas por minuto "
        "en la ventana de 240 minutos. Para el tiempo de espera se usa una "
        "distribución Exponencial con λ = 1/media, que es la distribución "
        "natural del tiempo entre eventos de un proceso Poisson. Para el tiempo "
        "de servicio se usa una Normal con la media y la desviación estándar "
        "muestrales del conjunto depurado."
    )
    if bondad is not None and not bondad.empty:
        pdf.caption_tabla("Bondad de ajuste por Kolmogorov-Smirnov (α = 0.05)")
        pdf.tabla_df(bondad, widths=[26, 15, 30, 9, 11, 15], font_size=7.5)
        pdf.parrafo(
            "El estadístico KS mide la distancia máxima entre la función de "
            "distribución empírica y la teórica. Valores p mayores a 0.05 "
            "indican que el modelo propuesto no se rechaza; valores menores "
            "obligan a buscar otra familia de distribuciones o a segmentar los "
            "datos."
        )
    else:
        pdf.nota_pendiente(
            "La tabla de bondad de ajuste no está disponible para este "
            "conjunto de datos.")

    pdf.subtitulo("6.2 Ajuste de Poisson a las llegadas")
    if pois:
        pdf.caption_tabla("Parámetros del proceso de llegadas")
        pdf.tabla_metrica([
            ("λ estimada (llegadas por minuto)", _fmt(pois.get("lambda"), 4)),
            ("Media empírica", _fmt(pois.get("media_empirica"), 4)),
            ("Media simulada", _fmt(pois.get("media_simulada"), 4)),
            ("Minutos analizados", _fmt(pois.get("n_minutos"), 0)),
            ("Llegadas totales", _fmt(pois.get("total_llegadas"), 0)),
            ("Estadístico KS", _fmt(pois.get("ks_stat"), 4)),
            ("p-value KS", _fmt(pois.get("p_value"), 4)),
            ("Conclusión (α = 0.05)",
             "No se rechaza Poisson" if (pois.get("p_value") or 0) > 0.05
             else "Se rechaza Poisson"),
        ])
        pdf.parrafo(
            "Si la media simulada se aproxima a la empírica y el histograma "
            "comparativo se superpone, Poisson describe razonablemente el "
            "proceso de arribos. Esto habilita calcular probabilidades de "
            "atención con la fórmula de colas y dimensionar el personal en la "
            "hora pico."
        )
    else:
        pdf.nota_pendiente(
            "No fue posible estimar el proceso de llegadas con los datos "
            "actuales.")
    pdf.agregar_figura(png.get("poisson_hist"), ancho=160)
    pdf.caption_figura("Llegadas por minuto: empírico vs Poisson simulada")
    pdf.agregar_figura(png.get("poisson_qq"), ancho=120)
    pdf.caption_figura("Q-Q plot del ajuste Poisson a las llegadas")

    pdf.subtitulo("6.3 Tiempo de espera (Exponencial)")
    pdf.parrafo(
        "La distribución Exponencial supone que la probabilidad de que un "
        "cliente sea atendido en el siguiente instante no depende del tiempo "
        "que ya lleva esperando (falta de memoria). El panel izquierdo compara "
        "el histograma con la densidad teórica y el derecho comprueba el ajuste "
        "con un Q-Q plot: si los puntos siguen la recta, el modelo es "
        "razonable."
    )
    pdf.agregar_figura(png.get("ajuste_espera"), ancho=160)
    pdf.caption_figura("Ajuste Exponencial al tiempo de espera")

    pdf.subtitulo("6.4 Tiempo de servicio (Normal)")
    pdf.parrafo(
        "El tiempo de servicio depende de la combinación de varios factores "
        "(producto solicitado, método de pago, experiencia del barista), por lo "
        "que se propone un modelo Normal. El ajuste se evalúa con el mismo "
        "esquema de histograma más Q-Q plot."
    )
    pdf.agregar_figura(png.get("ajuste_servicio"), ancho=160)
    pdf.caption_figura("Ajuste Normal al tiempo de servicio")


ETIQUETAS_SIM = {
    "bernoulli": "Bernoulli (p = 0.5)",
    "binomial": "Binomial (n = 10, p = 0.5)",
    "geometrica": "Geométrica (p = 0.3)",
    "hipergeometrica": "Hipergeométrica (N = 200, K = 60, n = 10)",
    "gamma": "Gamma (shape = 2.0, scale = 1.5)",
}


def _filas_resumen(resumen: dict) -> list[tuple[str, str]]:
    """Convierte el resumen de una simulación en filas para tabla_metrica."""
    filas = []
    for clave, valor in (resumen or {}).items():
        if valor is None:
            continue
        if isinstance(valor, str) and not valor.strip():
            continue
        if isinstance(valor, (float, np.floating)) and not np.isfinite(valor):
            continue
        if isinstance(valor, (int, float, np.integer, np.floating)):
            texto = _fmt(valor, 4)
        else:
            texto = str(valor)
        filas.append((str(clave).replace("_", " ").capitalize(), texto))
    return filas


def _cap_6_simulaciones(pdf, ctx: dict):
    """Sección 6.5: simulaciones de apoyo de la Unidad 2."""
    pdf.subtitulo("6.5 Simulaciones de apoyo")
    pdf.parrafo(
        "Además de los modelos ajustados a los datos reales, el módulo de "
        "distribuciones permite simular variables aleatorias discretas y "
        "continuas para ilustrar su forma, su media teórica y su dispersión. "
        "Estas simulaciones sirven de apoyo didáctico: muestran cómo cambia la "
        "forma de la distribución al modificar sus parámetros y permiten "
        "comparar la media empírica con la esperanza teórica."
    )
    sims = ctx.get("sims") or {}
    if not sims:
        pdf.nota_pendiente(
            "Las simulaciones de Bernoulli, Binomial, Geométrica, "
            "Hipergeométrica y Gamma se habilitarán cuando se agreguen las "
            "funciones simular_* en mod_distribuciones.")
        return
    for clave, datos in sims.items():
        etiqueta = ETIQUETAS_SIM.get(clave, clave.capitalize())
        pdf.agregar_figura((datos or {}).get("fig"), ancho=120)
        pdf.caption_figura(f"Simulación de una distribución {etiqueta}")
        filas = _filas_resumen((datos or {}).get("resumen") or {})
        if filas:
            pdf.caption_tabla(f"Resumen de la simulación {etiqueta}")
            pdf.tabla_metrica(filas)


def _cap_6_lectura(pdf, ctx: dict, textos: dict):
    """Sección 6.6: interpretación de la Unidad 2."""
    pdf.subtitulo("6.6 Lectura estadística")
    pdf.nota_doble(*_texto(textos, "dist"))
    pdf.preguntas_guia([
        "¿El supuesto de falta de memoria es razonable para esta fila?",
        "¿Qué pasaría con la espera si λ se duplicara en la hora pico?",
        "¿Conviene modelar el servicio con otra distribución asimétrica?",
    ])


def _tabla_generica(pdf, obj, caption: str) -> bool:
    """Dibuja un resultado (DataFrame o dict) si existe; avisa si no."""
    if obj is None:
        return False
    if isinstance(obj, pd.DataFrame):
        if obj.empty:
            return False
        pdf.caption_tabla(caption)
        pdf.tabla_df(obj, font_size=7.5)
        return True
    if isinstance(obj, dict):
        filas = _filas_resumen(obj)
        if not filas:
            return False
        pdf.caption_tabla(caption)
        pdf.tabla_metrica(filas)
        return True
    pdf.caption_tabla(caption)
    pdf.parrafo(str(obj))
    return True


def _cap_7(pdf, ctx: dict, textos: dict):
    pdf.titulo_capitulo("7", "Pruebas de Hipótesis (Unidad 3)")
    png = ctx["png"]
    hip = ctx.get("hip") or {}
    grupos = ctx.get("hip_grupos") or []

    pdf.subtitulo("7.1 Planteamiento de la prueba")
    if hip and not hip.get("error"):
        pdf.parrafo(
            f"Se contrasta si el consumo promedio difiere entre los grupos "
            f"{hip.get('grupo_a')} y {hip.get('grupo_b')} de la variable "
            f"{hip.get('valor_col')}. La hipótesis nula establece que las medias "
            "poblacionales son iguales (H0: μ1 = μ2) y la alternativa que son "
            "distintas (H1: μ1 ≠ μ2), con un nivel de significancia α = 0.05."
        )
    else:
        pdf.parrafo(
            "Se contrasta si el consumo promedio difiere entre tipos de cliente "
            "con un nivel de significancia α = 0.05. La prueba no pudo "
            "calcularse con el conjunto actual porque faltan grupos o registros "
            "suficientes; se documenta el procedimiento y se muestra el "
            "resultado cuando exista evidencia suficiente."
        )
        if hip.get("error"):
            pdf.nota_pendiente(str(hip["error"]))

    pdf.subtitulo("7.2 Supuestos y pruebas")
    if hip and not hip.get("error"):
        pdf.caption_tabla("Verificación de supuestos y resultados de las pruebas")
        pdf.tabla_metrica([
            ("n del grupo A", _fmt(hip.get("n_a"), 0)),
            ("n del grupo B", _fmt(hip.get("n_b"), 0)),
            ("Media del grupo A", _fmt(hip.get("media_a"), 3)),
            ("Media del grupo B", _fmt(hip.get("media_b"), 3)),
            ("Desviación estándar A", _fmt(hip.get("std_a"), 3)),
            ("Desviación estándar B", _fmt(hip.get("std_b"), 3)),
            ("Shapiro-Wilk p (A)", _fmt(hip.get("shapiro_a_p"), 4)),
            ("Shapiro-Wilk p (B)", _fmt(hip.get("shapiro_b_p"), 4)),
            ("Levene p (homocedasticidad)", _fmt(hip.get("levene_p"), 4)),
            ("t de Student", _fmt(hip.get("ttest_stat"), 4)),
            ("p-value de t", _fmt(hip.get("ttest_p"), 4)),
            ("¿Varianzas iguales?",
             "Sí" if hip.get("ttest_equal_var") else "No (Welch)"),
            ("U de Mann-Whitney", _fmt(hip.get("mannwhitney_u"), 2)),
            ("p-value de Mann-Whitney", _fmt(hip.get("mannwhitney_p"), 4)),
            ("d de Cohen", _fmt(hip.get("cohen_d"), 3)),
            ("Magnitud del efecto", hip.get("interpretacion_d") or "N/A"),
        ])
        pdf.parrafo(
            "La prueba t se interpreta solo si Shapiro-Wilk no rechaza la "
            "normalidad (p > 0.05) y Levene no rechaza la homocedasticidad "
            "(p > 0.05); en caso contrario se reporta Mann-Whitney U, que "
            "compara rangos y no exige normalidad. El valor d de Cohen traduce "
            "la diferencia en desviaciones estándar: por encima de 0.8 se "
            "considera un efecto grande."
        )
        conclusion_t = (
            "se rechaza H0: existen diferencias significativas entre grupos"
            if (hip.get("ttest_p") or 1) < 0.05
            else "no se rechaza H0: no hay evidencia de diferencias entre grupos"
        )
        pdf.parrafo(
            f"Con α = 0.05 y un p-value de {_fmt(hip.get('ttest_p'), 4)} en la "
            f"prueba t, {conclusion_t}."
        )
        pdf.agregar_figura(png.get("hip_box"), ancho=140)
        pdf.caption_figura(
            f"Comparación del consumo entre {grupos[0] if grupos else 'grupo A'} "
            f"y {grupos[1] if len(grupos) > 1 else 'grupo B'}")

    pdf.subtitulo("7.3 Comparación de más de dos grupos (ANOVA)")
    if not _tabla_generica(pdf, ctx.get("anova"),
                           "ANOVA de un factor sobre el consumo por tipo de cliente"):
        pdf.nota_pendiente(
            "La prueba ANOVA y su gráfica de medias se habilitarán cuando se "
            "agregue mod_hipotesis.anova.")

    pdf.subtitulo("7.4 Asociación entre variables categóricas (chi-cuadrado)")
    if not _tabla_generica(pdf, ctx.get("chi2"),
                           "Prueba chi-cuadrado de independencia"):
        pdf.nota_pendiente(
            "La prueba chi-cuadrado se habilitará cuando se agregue "
            "mod_hipotesis.chi_cuadrado.")

    pdf.subtitulo("7.5 Regresión logística: predicción de hora pico")
    logreg = ctx.get("logreg") or {}
    if logreg and not logreg.get("error"):
        met = logreg.get("metricas") or {}
        pdf.parrafo(
            f"El modelo se entrenó con {_fmt(logreg.get('n_train'), 0)} "
            f"registros y se evaluó con {_fmt(logreg.get('n_test'), 0)}. La "
            "variable objetivo es Es_Hora_Pico y las predictoras son la hora de "
            "llegada, la espera, el servicio, el consumo y la satisfacción."
        )
        pdf.caption_tabla("Métricas del modelo de regresión logística")
        pdf.tabla_metrica([
            ("Accuracy", _fmt(met.get("accuracy"), 4)),
            ("Precision", _fmt(met.get("precision"), 4)),
            ("Recall (sensibilidad)", _fmt(met.get("recall"), 4)),
            ("F1-score", _fmt(met.get("f1"), 4)),
            ("AUC-ROC", _fmt(met.get("auc"), 4)),
        ])
        pdf.agregar_figura(png.get("logreg_cm"), ancho=105)
        pdf.caption_figura("Matriz de confusión del modelo logístico")
        pdf.agregar_figura(png.get("logreg_roc"), ancho=130)
        pdf.caption_figura("Curva ROC del modelo logístico")
        reporte = logreg.get("reporte")
        if isinstance(reporte, pd.DataFrame) and not reporte.empty:
            pdf.caption_tabla("Reporte de clasificación por clase")
            pdf.tabla_df(reporte.reset_index(), font_size=7.5)
    else:
        pdf.parrafo(
            "El modelo logístico no pudo entrenarse con el conjunto actual. La "
            "causa más frecuente es que una de las clases de Es_Hora_Pico tenga "
            "menos de 20 registros o que falten columnas predictoras válidas."
        )
        if logreg.get("error"):
            pdf.nota_pendiente(str(logreg["error"]))

    pdf.subtitulo("7.6 Lectura estadística")
    pdf.nota_doble(*_texto(textos, "hip"))
    pdf.preguntas_guia([
        "¿La diferencia entre grupos es estadísticamente significativa y "
        "relevante en pesos?",
        "¿Qué predictor pesa más en la probabilidad de llegar en hora pico?",
        "¿Qué decisión cambiaría si el p-value estuviera en 0.06?",
    ])


def _cap_8(pdf, ctx: dict, textos: dict):
    pdf.titulo_capitulo("8", "Estadística Bayesiana (Unidad 4)")
    png = ctx["png"]
    bayes = ctx.get("bayes") or {}
    nb = ctx.get("nb") or {}
    comparacion = ctx.get("comparacion") or {}

    pdf.subtitulo("8.1 Teorema de Bayes aplicado a la cafetería")
    resultado = bayes.get("resultado") or {}
    if resultado:
        etiqueta_a = bayes.get("etiqueta_a", "A")
        etiqueta_b = bayes.get("etiqueta_b", "B")
        pdf.parrafo(
            f"Se define A = «el cliente es {etiqueta_a}» y B = «{etiqueta_b}». "
            f"La probabilidad a priori P(A) = {_fmt(resultado.get('p_a'), 4)} se "
            "estima con la frecuencia relativa del tipo de cliente en la base. "
            f"Las verosimilitudes son P(B|A) = {_fmt(resultado.get('p_b_dado_a'), 4)} "
            f"y P(B|¬A) = {_fmt(resultado.get('p_b_dado_no_a'), 4)}. La "
            f"probabilidad total del evento B es "
            f"P(B) = {_fmt(resultado.get('p_b'), 4)}."
        )
        pdf.parrafo(
            "Aplicando P(A|B) = P(B|A)·P(A) / P(B) se obtiene la probabilidad "
            "posterior, que combina la creencia previa con la evidencia "
            "observada en la caja."
        )
        pdf.caption_tabla("Componentes del teorema de Bayes")
        pdf.tabla_metrica([
            (f"P({etiqueta_a}) — prior", _fmt(resultado.get("p_a"), 4)),
            (f"P(¬{etiqueta_a}) — complemento", _fmt(resultado.get("p_no_a"), 4)),
            (f"P({etiqueta_b}|{etiqueta_a})", _fmt(resultado.get("p_b_dado_a"), 4)),
            (f"P({etiqueta_b}|¬{etiqueta_a})",
             _fmt(resultado.get("p_b_dado_no_a"), 4)),
            (f"P({etiqueta_b}) — evidencia", _fmt(resultado.get("p_b"), 4)),
            (f"P({etiqueta_a}|{etiqueta_b}) — posterior",
             _fmt(resultado.get("p_a_dado_b"), 4)),
            ("Prior empírico sin acotar", _fmt(bayes.get("p_a_empirico"), 4)),
        ])
        pdf.parrafo(
            "Cuando el prior empírico es 0 o 1 la fórmula se indetermina, por "
            "lo que el cálculo lo acota al intervalo [0.01, 0.99]. El valor "
            "empírico se conserva en la tabla para que la lectura no pierda "
            "trazabilidad."
        )
        pdf.agregar_figura(png.get("bayes_barras"), ancho=125)
        pdf.caption_figura("Comparación entre probabilidad a priori y posterior")
        if png.get("arbol_bayes_png"):
            pdf.agregar_figura(png.get("arbol_bayes_png"), ancho=150)
            pdf.caption_figura("Árbol de decisión bayesiano")
    else:
        pdf.parrafo(
            "No fue posible estimar los componentes del teorema de Bayes con el "
            "conjunto actual porque falta alguno de los valores de tipo de "
            "cliente o método de pago."
        )

    pdf.subtitulo("8.2 Clasificador Naive Bayes")
    if nb and not nb.get("error"):
        met = nb.get("metricas") or {}
        pdf.parrafo(
            f"El clasificador se entrenó con {_fmt(nb.get('n_train'), 0)} "
            f"registros, {_fmt(nb.get('n_features'), 0)} predictores y se "
            f"evaluó con {_fmt(nb.get('n_test'), 0)} registros. Supone "
            "independencia condicional entre predictores, supuesto que rara vez "
            "se cumple de forma exacta pero que suele dar un desempeño aceptable "
            "en problemas de clasificación simples."
        )
        pdf.caption_tabla("Métricas y calibración de Naive Bayes")
        pdf.tabla_metrica([
            ("Accuracy", _fmt(met.get("accuracy"), 4)),
            ("Log-loss", _fmt(met.get("log_loss"), 4)),
            ("Brier score", _fmt(met.get("brier_score"), 4)),
            ("Predictores utilizados", _fmt(nb.get("n_features"), 0)),
        ])
        pdf.parrafo(
            "El log-loss y el Brier score miden la calidad de las "
            "probabilidades, no solo de la etiqueta: cuanto más cerca de cero, "
            "mejor calibrado está el modelo. Un log-loss bajo con accuracy "
            "similar a otro modelo indica que Naive Bayes ordena mejor las "
            "probabilidades, algo útil para priorizar turnos."
        )
        pdf.agregar_figura(png.get("nb_cm"), ancho=105)
        pdf.caption_figura("Matriz de confusión de Naive Bayes")
        pdf.agregar_figura(png.get("nb_proba"), ancho=130)
        pdf.caption_figura("Probabilidades posteriores promedio por clase real")
        reporte = nb.get("reporte")
        if isinstance(reporte, pd.DataFrame) and not reporte.empty:
            pdf.caption_tabla("Reporte de clasificación de Naive Bayes")
            pdf.tabla_df(reporte.reset_index(), font_size=7.5)
    else:
        pdf.parrafo(
            "Naive Bayes no pudo entrenarse con los datos actuales; se "
            "conservan las secciones descriptivas y el cálculo teórico del "
            "teorema de Bayes."
        )
        if nb.get("error"):
            pdf.nota_pendiente(str(nb["error"]))

    pdf.subtitulo("8.3 Comparación de modelos con validación cruzada")
    tabla_cmp = comparacion.get("tabla")
    if isinstance(tabla_cmp, pd.DataFrame) and not tabla_cmp.empty:
        pdf.parrafo(
            "Se comparan la regresión logística, Naive Bayes y un clasificador "
            "de referencia mediante validación cruzada estratificada de 5 "
            "pliegues. La tabla muestra la accuracy media y su desviación "
            "estándar entre pliegues; una desviación pequeña indica un "
            "desempeño estable."
        )
        pdf.caption_tabla("Accuracy media por modelo (validación cruzada 5-fold)")
        pdf.tabla_df(tabla_cmp, widths=[40, 25, 25], font_size=7.5)
        pdf.agregar_figura(png.get("modelos_bar"), ancho=145)
        pdf.caption_figura("Comparación gráfica de la accuracy por modelo")
        friedman = comparacion.get("friedman") or {}
        if friedman.get("p") is not None:
            pdf.caption_tabla("Prueba de Friedman para modelos pareados")
            pdf.tabla_metrica([
                ("Estadístico de Friedman", _fmt(friedman.get("stat"), 4)),
                ("p-value", _fmt(friedman.get("p"), 4)),
                ("Interpretación", friedman.get("interpretacion") or "N/A"),
            ])
            pdf.parrafo(
                "La prueba de Friedman contrasta si los modelos tienen el mismo "
                "desempeño esperado. Un p-value menor a 0.05 indica que al "
                "menos uno difiere de los demás; en ese caso se recomienda "
                "elegir el de mayor accuracy media y verificar su estabilidad "
                "con la desviación estándar."
            )
    else:
        pdf.parrafo(
            "La comparación de modelos no está disponible para el conjunto "
            "actual; se requiere al menos una clase por pliegue en la variable "
            "objetivo."
        )
        if comparacion.get("error"):
            pdf.nota_pendiente(str(comparacion["error"]))

    pdf.subtitulo("8.4 Lectura estadística")
    pdf.nota_doble(*_texto(textos, "bayes"))
    pdf.preguntas_guia([
        "¿La evidencia de pago con tarjeta cambia de forma importante la "
        "probabilidad de ser profesor?",
        "¿Qué modelo conviene para decidir el turno y cuál para estimar "
        "probabilidades?",
        "¿Cómo se actualizaría el prior si la población de clientes cambia?",
    ])


def _cap_9(pdf, ctx: dict, textos: dict):
    pdf.titulo_capitulo("9", "Conclusiones y Recomendaciones")
    k = ctx["kpis"]
    hip = ctx.get("hip") or {}
    logreg = ctx.get("logreg") or {}
    nb = ctx.get("nb") or {}
    comparacion = ctx.get("comparacion") or {}
    pois = ctx.get("poisson") or {}

    pdf.subtitulo("9.1 Síntesis de hallazgos por unidad")
    tabla_cmp = comparacion.get("tabla")
    mejor_modelo = (
        str(tabla_cmp["Modelo"].iloc[0])
        if isinstance(tabla_cmp, pd.DataFrame) and len(tabla_cmp) else "N/A"
    )
    pdf.caption_tabla("Hallazgos principales por unidad del curso")
    pdf.tabla_metrica([
        ("Unidad 1 — Descriptiva",
         f"Consumo medio ${k['consumo_prom']:.2f} MXN; espera media "
         f"{k['espera_prom']:.2f} min; {_fmt(k['outliers'])} atípicos."),
        ("Unidad 2 — Distribuciones",
         f"λ = {_fmt(pois.get('lambda'), 3)} llegadas/min; KS p = "
         f"{_fmt(pois.get('p_value'), 4)}."),
        ("Unidad 3 — Hipótesis",
         f"t p = {_fmt(hip.get('ttest_p'), 4)}; Mann-Whitney p = "
         f"{_fmt(hip.get('mannwhitney_p'), 4)}; AUC logístico = "
         f"{_fmt((logreg.get('metricas') or {}).get('auc'), 3)}."),
        ("Unidad 4 — Bayesiana",
         f"Posterior = {_fmt((ctx.get('bayes') or {}).get('resultado', {}).get('p_a_dado_b'), 3)}; "
         f"accuracy Naive Bayes = "
         f"{_fmt((nb.get('metricas') or {}).get('accuracy'), 3)}."),
        ("Integración", f"Mejor modelo de la comparación: {mejor_modelo}."),
    ], ancho1=42, ancho2=123, encabezados=("Unidad", "Hallazgo principal"))

    pdf.subtitulo("9.2 Conclusiones")
    pdf.nota_doble(*_texto(textos, "concl"))

    pdf.subtitulo("9.3 Recomendaciones")
    recomendaciones = [
        "Reforzar el personal entre 8:00 y 9:00 AM, franja en la que se "
        f"concentra el {k['pct_pico']:.1f}% de las llegadas y se producen las "
        "esperas máximas.",
        "Habilitar prepago por aplicación para reducir el tiempo de servicio "
        "en caja y liberar al barista durante la hora pico.",
        "Establecer una meta de tiempo de espera con base en el percentil 75 y "
        "no en el promedio, para proteger a los clientes más afectados.",
        "Revisar el formulario de captura para impedir tiempos de servicio "
        "negativos y consumos fuera de rango, que generan la mayoría de los "
        "atípicos.",
        "Dar seguimiento mensual a la satisfacción y a su relación con la "
        "espera, usando este reporte como línea base.",
    ]
    for i, rec in enumerate(recomendaciones, 1):
        pdf.parrafo(f"{i}. {rec}")

    pdf.subtitulo("9.4 Limitaciones y trabajo futuro")
    pdf.parrafo(
        "La muestra es de conveniencia y el tamaño condiciona la potencia de "
        "las pruebas. No se registran clientes que se retiran sin comprar ni el "
        "detalle de productos, por lo que no es posible estimar pérdida de "
        "ventas por abandono de fila. Como trabajo futuro se propone ampliar el "
        "periodo de observación, incluir inventario por producto y validar los "
        "clasificadores con datos de un semestre completo."
    )
    pdf.preguntas_guia([
        "¿Qué recomendación se implementaría primero y cómo se mediría su "
        "efecto?",
        "¿Qué indicador conviene vigilar semanalmente?",
    ])


def _cap_10(pdf, ctx: dict, textos: dict):
    pdf.titulo_capitulo("10", "Referencias Bibliográficas")
    pdf.parrafo(
        "Las referencias se presentan en formato APA (7a edición). Se incluyen "
        "las fuentes metodológicas del curso, la documentación de las "
        "bibliotecas utilizadas y la documentación de los servicios de datos."
    )
    referencias = [
        "Anderson, D. R., Sweeney, D. J., & Williams, T. A. (2019). "
        "Estadística para negocios y economía (13a ed.). Cengage Learning.",
        "Devore, J. L. (2016). Probabilidad y estadística para ingeniería y "
        "ciencias (9a ed.). Cengage Learning.",
        "FPDF2 contributors. (2024). fpdf2 documentation. "
        "https://pyfpdf.github.io/fpdf2/",
        "Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). "
        "Array programming with NumPy. Nature, 585, 357-362.",
        "McKinney, W. (2022). Python for data analysis (3a ed.). O'Reilly Media.",
        "Montgomery, D. C., & Runger, G. C. (2018). Applied statistics and "
        "probability for engineers (7a ed.). Wiley.",
        "Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). "
        "Scikit-learn: Machine learning in Python. Journal of Machine Learning "
        "Research, 12, 2825-2830.",
        "Ross, S. M. (2019). Introduction to probability models (12a ed.). "
        "Academic Press.",
        "Streamlit Inc. (2024). Streamlit documentation. "
        "https://docs.streamlit.io",
        "Supabase Inc. (2024). Supabase documentation. "
        "https://supabase.com/docs",
    ]
    for referencia in referencias:
        pdf.parrafo(referencia)

    pdf.subtitulo("Nota sobre la reproducibilidad")
    pdf.parrafo(
        "Todos los cálculos se ejecutan al momento de generar el reporte a "
        "partir de la base de datos vigente; no se almacenan resultados "
        "intermedios. Por ello, dos ejecuciones con los mismos datos producen "
        "el mismo documento salvo por la fecha de consulta y los textos "
        "generados por el modelo de lenguaje, que se apoyan siempre en métricas "
        "calculadas y disponen de un texto de respaldo determinista."
    )


# ============================================================
# ENSAMBLADO DEL DOCUMENTO
# ============================================================
def _construir_contenido_est(pdf, ctx: dict, textos: dict):
    """Dibuja los diez capítulos numerados (sin el resumen ejecutivo)."""
    _cap_1(pdf, ctx, textos)
    _cap_2(pdf, ctx, textos)
    _cap_3(pdf, ctx, textos)
    _cap_4(pdf, ctx, textos)
    _cap_5(pdf, ctx, textos)
    _cap_6(pdf, ctx, textos)
    _cap_6_simulaciones(pdf, ctx)
    _cap_6_lectura(pdf, ctx, textos)
    _cap_7(pdf, ctx, textos)
    _cap_8(pdf, ctx, textos)
    _cap_9(pdf, ctx, textos)
    _cap_10(pdf, ctx, textos)


def _resumen_ejecutivo_est(pdf, ctx: dict, textos: dict):
    """Dibuja el resumen ejecutivo (front matter, sin numerar)."""
    _cap_resumen_ejecutivo(pdf, ctx, textos)


def _cuerpo(pdf, ctx: dict, textos: dict):
    """Dibuja resumen ejecutivo y los diez capítulos."""
    _resumen_ejecutivo_est(pdf, ctx, textos)
    _construir_contenido_est(pdf, ctx, textos)


def _render_documento(pdf, ctx: dict, textos: dict,
                      indices: dict | None = None,
                      total_pages: int = 0):
    """Dibuja un documento completo; `indices` precarga los índices."""
    pdf._init_registros()
    pdf.total_pages = total_pages
    if indices:
        pdf.indice_caps = dict(indices.get("caps") or {})
        pdf.indice_figs = dict(indices.get("figs") or {})
        pdf.indice_tabs = dict(indices.get("tabs") or {})

    pdf.portada_institucional()
    pdf.indice_contenido()
    pdf.indice_figuras()
    pdf.indice_tablas()

    # A partir de aquí se reconstruyen los índices con las páginas reales.
    pdf.indice_caps = {}
    pdf.indice_figs = {}
    pdf.indice_tabs = {}
    pdf.fig_counter = 0
    pdf.tab_counter = 0
    _cuerpo(pdf, ctx, textos)


def _entradas(pdf) -> dict:
    return {
        "caps": dict(pdf.indice_caps),
        "figs": dict(pdf.indice_figs),
        "tabs": dict(pdf.indice_tabs),
    }


def _construir_contexto_est(df: pd.DataFrame | None = None) -> dict:
    """Reúne datos, estadísticos, modelos y figuras del reporte."""
    return _contexto(df)


def _preparar_tareas_ia_est(ctx: dict) -> dict:
    """Arma las tareas de IA por capítulo y devuelve los textos del reporte."""
    return _preparar_textos(ctx)


def _render_reporte_est(ctx: dict, textos: dict):
    """Dibuja el documento con las tres pasadas y devuelve el PDF final.

    Pasada 1 descubre las entradas de los índices, la pasada 2 las pagina con
    el índice ya lleno y la pasada 3 fija el total de páginas del pie.
    """
    borrador = ReporteCafeteriaEst()
    borrador.modo = "borrador"
    _render_documento(borrador, ctx, textos)

    paginado = ReporteCafeteriaEst()
    paginado.modo = "borrador"
    _render_documento(paginado, ctx, textos, indices=_entradas(borrador))

    final = ReporteCafeteriaEst()
    final.modo = "final"
    _render_documento(final, ctx, textos, indices=_entradas(paginado),
                      total_pages=paginado.page_no())
    return final


def generar_reporte_est_pdf(df: pd.DataFrame | None = None,
                            min_registros: int = 20
                            ) -> tuple[bytes | None, str | None]:
    """Genera el PDF completo y devuelve `(pdf_bytes, error)`.

    `error` es None cuando el documento se construyó; en caso contrario
    explica el motivo (pocos registros, Supabase sin conexión, excepción).
    """
    try:
        if df is None:
            df, error_carga = dm.load_data()
            if error_carga:
                return None, f"No se pudo cargar de Supabase: {error_carga}"
        disponibles = 0 if df is None else len(df)
        if disponibles < min_registros:
            return None, (
                f"Se requieren al menos {min_registros} registros para el "
                f"reporte estadístico (disponibles: {disponibles})."
            )

        ctx = _construir_contexto_est(df)
        textos = _preparar_tareas_ia_est(ctx)
        pdf = _render_reporte_est(ctx, textos)
        return bytes(pdf.output()), None
    except Exception as exc:  # pragma: no cover - salvaguarda para la UI
        import traceback

        traceback.print_exc()
        return None, f"Error al generar reporte: {exc}"


def generar_reporte_est(df: pd.DataFrame | None = None) -> bytes:
    """Genera el PDF del reporte estadístico completo en memoria.

    Se ejecutan tres pasadas: la primera descubre las entradas de los índices,
    la segunda las pagina con el índice ya lleno y la tercera escribe el pie de
    página con el total de páginas definitivo.
    """
    pdf_bytes, error = generar_reporte_est_pdf(df)
    if error:
        raise RuntimeError(error)
    return pdf_bytes
