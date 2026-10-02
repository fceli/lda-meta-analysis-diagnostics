# Librerías que faltan (correr en terminal, o en una celda anteponiendo !)
# !pip install pandas numpy spacy gensim scipy matplotlib
# !python -m spacy download en_core_web_sm
# !python3 -m pip install spacy

################################################################################

import pandas as pd
import numpy as np
import re, glob, csv
from pathlib import Path
import spacy
from scipy import stats

# Gensim: modelado de tópicos (LDA) y DTM
import gensim
import gensim.corpora as corpora
from gensim.models import TfidfModel, LdaMulticore
from gensim.models import CoherenceModel

# Modelo en inglés de spaCy (el corpus está en inglés)
nlp = spacy.load("en_core_web_sm")

################################################################################


# Rutas de los archivos
ruta_pubmed = './datos/PubMed 8Jul2026.csv'
ruta_wos = './datos/WoS full 8JUL26.txt'

# PubMed viene en CSV normal
df_pubmed = pd.read_csv(ruta_pubmed)

# WoS exporta en TXT separado por tabs
df_wos = pd.read_csv(ruta_wos, sep='\t')

################################################################################

# Paso 1.1: abstracts y afiliaciones de PubMed vía NCBI Entrez
# El CSV que exporta PubMed no trae el abstract ni la afiliación institucional,
# así que los bajo aparte con los PMID que ya tengo, vía Biopython/Entrez. La
# afiliación del primer autor con dato disponible se usa después para sacar el
# país (paso 1.2, Figura 3.3 del Cap. 3).
# Necesita internet. Si falta el paquete: %pip install biopython
from Bio import Entrez
import time

Entrez.email = "tu_correo@ejemplo.com"  # NCBI pide un email para identificar al usuario

def obtener_abstracts_y_afiliaciones(pmids, tamano_lote=50, pausa=0.4):
    abstracts = {}
    afiliaciones = {}
    pmids = [str(p) for p in pmids]
    for inicio in range(0, len(pmids), tamano_lote):
        lote = pmids[inicio:inicio + tamano_lote]
        handle = Entrez.efetch(db="pubmed", id=",".join(lote), rettype="abstract", retmode="xml")
        registros = Entrez.read(handle)
        handle.close()
        for articulo in registros.get('PubmedArticle', []):
            pmid = str(articulo['MedlineCitation']['PMID'])
            try:
                partes_abstract = articulo['MedlineCitation']['Article']['Abstract']['AbstractText']
                texto = " ".join(str(parte) for parte in partes_abstract)
            except KeyError:
                texto = None
            abstracts[pmid] = texto
            afiliacion = None
            try:
                autores = articulo['MedlineCitation']['Article']['AuthorList']
                for autor in autores:
                    infos = autor.get('AffiliationInfo', [])
                    if infos:
                        afiliacion = str(infos[0].get('Affiliation', ''))
                        break
            except (KeyError, TypeError):
                pass
            afiliaciones[pmid] = afiliacion
        time.sleep(pausa)  # para no pasarme del límite de peticiones de NCBI (no tengo api_key)
    return abstracts, afiliaciones

print("Recuperando abstracts y afiliaciones de PubMed vía NCBI Entrez (puede tardar unos segundos)...")
abstracts_pubmed, afiliaciones_pubmed = obtener_abstracts_y_afiliaciones(df_pubmed['PMID'].tolist())
df_pubmed['Abstract'] = df_pubmed['PMID'].astype(str).map(abstracts_pubmed)
df_pubmed['Afiliacion'] = df_pubmed['PMID'].astype(str).map(afiliaciones_pubmed)

print(f"Abstracts recuperados: {df_pubmed['Abstract'].notna().sum()} de {len(df_pubmed)} registros PubMed.")
print(f"Afiliaciones recuperadas: {df_pubmed['Afiliacion'].notna().sum()} de {len(df_pubmed)} registros PubMed.")

################################################################################

# Paso 1.2: sacar el país de las afiliaciones (PubMed + WoS)
# Ni PubMed ni WoS traen una columna de país lista para usar, así que se saca
# por heurística del texto de afiliación (PubMed: AffiliationInfo vía Entrez;
# WoS: campo C1). El país se normaliza a ISO 3166-1 alfa-3 con pycountry, para
# después poder cruzarlo con un shapefile mundial (mapa, Figura 3.3).
# Si falta el paquete: %pip install pycountry
import re
import pycountry

ALIAS_PAIS = {
    'usa': 'United States', 'u s a': 'United States', 'united states of america': 'United States',
    'uk': 'United Kingdom', 'u k': 'United Kingdom', 'england': 'United Kingdom',
    'scotland': 'United Kingdom', 'wales': 'United Kingdom', 'northern ireland': 'United Kingdom',
    'peoples r china': 'China', 'peoples republic of china': 'China', 'china mainland': 'China',
    'south korea': 'Korea, Republic of', 'korea': 'Korea, Republic of', 'republic of korea': 'Korea, Republic of',
    'russia': 'Russian Federation', 'iran': 'Iran, Islamic Republic of',
    'taiwan': 'Taiwan, Province of China', 'vietnam': 'Viet Nam', 'czech republic': 'Czechia',
    'hong kong': 'Hong Kong', 'macau': 'Macao', 'uae': 'United Arab Emirates',
}

def normalizar_pais(candidato):
    if not candidato:
        return None
    c = re.sub(r'[^a-zA-Z ]', '', candidato).strip().lower()
    c = re.sub(r'\s+', ' ', c)
    if not c:
        return None
    nombre = ALIAS_PAIS.get(c)
    if nombre is None:
        try:
            nombre = pycountry.countries.search_fuzzy(c)[0].name
        except LookupError:
            return None
    try:
        return pycountry.countries.search_fuzzy(nombre)[0].alpha_3
    except LookupError:
        return None

def extraer_pais_pubmed(afiliacion):
    if not isinstance(afiliacion, str) or not afiliacion.strip():
        return None
    texto = re.sub(r'\S+@\S+', '', afiliacion).strip().rstrip('.')  # saco los emails
    segmentos = [s.strip() for s in texto.split(',') if s.strip()]
    for candidato in reversed(segmentos[-3:]):
        iso = normalizar_pais(candidato)
        if iso:
            return iso
    return None

def extraer_pais_wos(c1):
    if not isinstance(c1, str) or not c1.strip():
        return None
    primer_bloque = re.sub(r'^\[[^\]]*\]\s*', '', c1.split(';')[0])
    segmentos = [s.strip() for s in primer_bloque.split(',') if s.strip()]
    if not segmentos:
        return None
    ultimo = re.sub(r'\b\d{4,}\b', '', segmentos[-1]).strip()
    iso = normalizar_pais(ultimo)
    if iso:
        return iso
    return normalizar_pais(segmentos[-2]) if len(segmentos) >= 2 else None

df_pubmed['Pais_ISO3'] = df_pubmed['Afiliacion'].apply(extraer_pais_pubmed)
df_wos['Pais_ISO3'] = df_wos['C1'].apply(extraer_pais_wos) if 'C1' in df_wos.columns else None

print(f"Países extraídos PubMed: {df_pubmed['Pais_ISO3'].notna().sum()} de {len(df_pubmed)}")
print(f"Países extraídos WoS: {df_wos['Pais_ISO3'].notna().sum()} de {len(df_wos)}")

################################################################################

df_pubmed.info()

################################################################################

df_wos.info()

################################################################################

# Elimino columnas que quedaron 100% vacías en WoS (de 79 columnas a solo las útiles)
df_wos.dropna(axis=1, how='all', inplace=True)

# Paso 2.2: mapeo y estandarización de variables
# 1. PubMed (nombres de columna completos)
df_pub_std = pd.DataFrame()
df_pub_std['Titulo'] = df_pubmed['Title']
df_pub_std['Resumen'] = df_pubmed['Abstract']  # viene de NCBI Entrez (celda anterior)
df_pub_std['DOI'] = df_pubmed['DOI']
df_pub_std['Anio'] = df_pubmed['Publication Year']
df_pub_std['Citas'] = np.nan
df_pub_std['Fuente'] = 'PubMed'
df_pub_std['Revista'] = df_pubmed['Journal/Book']  # PubMed usa el nombre abreviado (norma NLM)
df_pub_std['Pais_ISO3'] = df_pubmed['Pais_ISO3']

# 2. Web of Science (etiquetas de 2 letras)
df_wos_std = pd.DataFrame()
df_wos_std['Titulo'] = df_wos['TI']   # Title
df_wos_std['Resumen'] = df_wos['AB']  # Abstract
df_wos_std['DOI'] = df_wos['DI']      # DOI
df_wos_std['Anio'] = df_wos['PY']     # Publication Year
df_wos_std['Citas'] = df_wos['TC']    # Times Cited (o 'Z9' para All Databases)
df_wos_std['Fuente'] = 'WoS'
df_wos_std['Revista'] = df_wos['SO']  # WoS sí trae el nombre completo de la revista
df_wos_std['Pais_ISO3'] = df_wos['Pais_ISO3']

# Paso 2.3: concatenar y deduplicar
df_consolidado = pd.concat([df_pub_std, df_wos_std], ignore_index=True)

# Limpio el DOI para poder cruzarlo exacto
df_consolidado['DOI'] = df_consolidado['DOI'].astype(str).str.strip().str.lower()
df_consolidado.loc[df_consolidado['DOI'] == 'nan', 'DOI'] = None

# Ordeno para que los registros con resumen queden primero, así al eliminar
# duplicados con keep='first' me quedo con el que sí tiene abstract.
df_consolidado = df_consolidado.sort_values(by='Resumen', na_position='last')

# Filtro 1: duplicados por DOI exacto
df_unico = df_consolidado.drop_duplicates(subset=['DOI'], keep='first')

# Filtro 2: duplicados por título exacto (normalizado)
df_unico['Titulo_limpio'] = df_unico['Titulo'].astype(str).str.lower().str.strip()
df_unico = df_unico.drop_duplicates(subset=['Titulo_limpio'], keep='first').reset_index(drop=True)

# Filtro 3: duplicados por similitud de título (rapidfuzz). Esto agarra los que
# sobreviven al cruce exacto por pequeñas diferencias de formato, puntuación u
# orden de palabras entre las exportaciones de PubMed y WoS.
from rapidfuzz import fuzz

UMBRAL_SIMILITUD = 92  # % de similitud (token_sort_ratio) para contar como duplicado
titulos = df_unico['Titulo_limpio'].tolist()
indices_a_eliminar = set()

for i in range(len(titulos)):
    if i in indices_a_eliminar:
        continue
    for j in range(i + 1, len(titulos)):
        if j in indices_a_eliminar:
            continue
        similitud = fuzz.token_sort_ratio(titulos[i], titulos[j])
        if similitud >= UMBRAL_SIMILITUD:
            # me quedo con el que tiene Resumen; si ambos o ninguno lo tiene, el primero
            tiene_resumen_i = pd.notna(df_unico.loc[i, 'Resumen'])
            tiene_resumen_j = pd.notna(df_unico.loc[j, 'Resumen'])
            if tiene_resumen_j and not tiene_resumen_i:
                indices_a_eliminar.add(i)
                break  # i ya no sirve como ancla, sigo con el próximo i
            else:
                indices_a_eliminar.add(j)

df_corpus = (df_unico.drop(index=indices_a_eliminar)
                      .drop(columns=['Titulo_limpio'])
                      .reset_index(drop=True)
                      .copy())

print(f"Registros únicos tras cruce por DOI y título exacto: {len(df_unico)}")
print(f"Duplicados adicionales detectados por similitud de título (rapidfuzz, umbral={UMBRAL_SIMILITUD}%): {len(indices_a_eliminar)}")
print(f"Registros únicos finales listos para NLP: {len(df_corpus)}")
print(f"De los cuales, {df_corpus['Resumen'].notna().sum()} contienen Resumen completo.")
print(f"De los cuales, {df_corpus['Pais_ISO3'].notna().sum()} tienen país de afiliación identificado.")

# Paso 2.4: normalización z-score por columna
columnas_cuantitativas = ['Anio', 'Citas']

for col in columnas_cuantitativas:
    mascara_validos = df_corpus[col].notna()
    if mascara_validos.any():
        df_corpus.loc[mascara_validos, f'{col}_Zscore'] = stats.zscore(df_corpus.loc[mascara_validos, col])

print("\n=== MUESTRA DEL CORPUS ESTANDARIZADO ===")
display(df_corpus.head(3))

################################################################################

# Paso 2.5: gráficos descriptivos del corpus (Cap. 3, Figuras 3.1-3.3)
import matplotlib.pyplot as plt
from pathlib import Path

FIG_DIR = Path("./figuras")
FIG_DIR.mkdir(exist_ok=True)

# Paleta "pastel analítica" (pasada por el validador de daltonismo de la skill de
# dataviz: luminosidad, croma mínimo, separación CVD y contraste, todo ok).
PASTEL = {"azul": "#4E7FBF", "naranja": "#D97B4A", "verde": "#3CA57F",
          "ambar": "#D9A22E", "violeta": "#8A76CF"}
GRID = "#e1e0d9"

# Figura 3.1: distribución anual apilada por fuente
tabla_anual = df_corpus.groupby(['Anio', 'Fuente']).size().unstack(fill_value=0)
for col in ['PubMed', 'WoS']:
    if col not in tabla_anual.columns:
        tabla_anual[col] = 0
tabla_anual = tabla_anual[['PubMed', 'WoS']].sort_index()

fig, ax = plt.subplots(figsize=(10, 6))
x_pos = range(len(tabla_anual))
ax.bar(x_pos, tabla_anual['PubMed'], color=PASTEL["azul"], label='PubMed', width=0.65)
ax.bar(x_pos, tabla_anual['WoS'], bottom=tabla_anual['PubMed'], color=PASTEL["naranja"],
       label='Web of Science', width=0.65)
ax.set_xticks(list(x_pos))
ax.set_xticklabels(tabla_anual.index.astype(int).astype(str), rotation=90)
ax.set_xlabel('Año de publicación')
ax.set_ylabel('Número de documentos')
ax.grid(axis='y', color=GRID, linewidth=0.8)  # sin gridlines secundarias
ax.set_axisbelow(True)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.legend(frameon=False, loc='upper left')
# no pongo plt.title(): el título va en el pie de figura de Word (Figura 3.1)
plt.tight_layout()
plt.savefig(FIG_DIR / "figura_3_1_distribucion_anual.png", dpi=200)
plt.show()

# Figura 3.2: distribución por fuente
conteo_fuente = df_corpus['Fuente'].value_counts()
colores_pie = [PASTEL["azul"] if s == 'PubMed' else PASTEL["naranja"] for s in conteo_fuente.index]
fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(conteo_fuente.values, labels=conteo_fuente.index, autopct='%1.1f%%',
       colors=colores_pie, startangle=90,
       wedgeprops={'linewidth': 2, 'edgecolor': 'white'})
plt.tight_layout()
plt.savefig(FIG_DIR / "figura_3_2_distribucion_fuente.png", dpi=200)
plt.show()

print(f"Figuras 3.1 y 3.2 guardadas en: {FIG_DIR}")

################################################################################

# Paso 2.6: mapa mundial de artículos por país (Cap. 3, Figura 3.3)
# Necesita geopandas. Si falta: %pip install geopandas
# Descarga una sola vez el shapefile de países de Natural Earth, en 50m (no
# 110m, porque ahí países chicos como Singapur no existen y quedaban afuera
# tanto del mapa como del recuadro de conteo).
import os
import urllib.request
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap

RUTA_SHAPEFILE = os.path.expanduser("~/Library/Caches/geodatasets/ne_50m_admin_0_countries.zip")
if not os.path.exists(RUTA_SHAPEFILE):
    os.makedirs(os.path.dirname(RUTA_SHAPEFILE), exist_ok=True)
    print("Descargando shapefile de países (Natural Earth 50m)...")
    urllib.request.urlretrieve(
        "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip",
        RUTA_SHAPEFILE,
    )

CMAP_SEQ = LinearSegmentedColormap.from_list("pastel_seq", ["#DCE7F5", PASTEL["azul"], "#274A75"])

conteo_pais = df_corpus['Pais_ISO3'].dropna().value_counts().reset_index()
conteo_pais.columns = ['ISO_A3', 'n']

world = gpd.read_file(f"zip://{RUTA_SHAPEFILE}")
world_merged = world.merge(conteo_pais, on='ISO_A3', how='left')
nombre_por_iso = dict(zip(world['ISO_A3'], world['NAME']))

# Mapa a ancho completo, sin panel lateral: el recuadro de país|artículos va
# superpuesto sobre el mapa mismo, en la esquina inferior izquierda (Pacífico
# Sur, zona sin datos), en vez de robarle un 22% del ancho de la figura como antes.
fig = plt.figure(figsize=(14, 7))
ax_map = fig.add_axes([0.01, 0.02, 0.98, 0.96])

world_merged.boundary.plot(ax=ax_map, linewidth=0.3, color='#c3c2b7')
world_merged.plot(column='n', ax=ax_map, cmap=CMAP_SEQ, linewidth=0.3, edgecolor='#c3c2b7',
                   missing_kwds={'color': '#f2f1ee', 'edgecolor': '#c3c2b7', 'linewidth': 0.3},
                   legend=True, legend_kwds={'label': 'N° de artículos', 'shrink': 0.5})

# Recorto la latitud para sacar la Antártida y el ártico (no hay datos ahí) y
# reducir el margen en blanco arriba y abajo; con aspect='auto' el mapa llena
# el recuadro en vez de quedar 'letterboxeado' por el aspecto 1:1 por defecto.
ax_map.set_xlim(-165, 178)
ax_map.set_ylim(-56, 82)
ax_map.set_aspect('auto')
ax_map.set_axis_off()

# Recuadro con el top 15 de países por número de artículos, sobre el mapa
top_paises = conteo_pais.sort_values('n', ascending=False).head(15).copy()
top_paises['Pais'] = top_paises['ISO_A3'].map(nombre_por_iso).fillna(top_paises['ISO_A3'])
ANCHO_NOMBRE = int(top_paises['Pais'].str.len().max()) + 2
lineas = [f"{row['Pais']:<{ANCHO_NOMBRE}}{int(row['n']):>4}" for _, row in top_paises.iterrows()]
encabezado = f"{'País':<{ANCHO_NOMBRE}}{'Art.':>4}"
texto_caja = encabezado + "\n" + "\n".join(lineas)
ax_map.text(0.02, 0.03, texto_caja, transform=ax_map.transAxes, ha='left', va='bottom',
            fontsize=9.5, family='monospace', color="#0b0b0b", zorder=10,
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#fcfcfbdd', edgecolor='#c3c2b7'))

# no pongo plt.title(): el título va en el pie de figura de Word (Figura 3.3)
plt.savefig(FIG_DIR / "figura_3_mapa_paises.png", dpi=200, bbox_inches='tight')
plt.show()

print(f"Países con artículo identificado: {len(conteo_pais)}")
print(top_paises[['Pais', 'n']].to_string(index=False))

################################################################################

# Paso 3: procesamiento de lenguaje natural (NLP)
import re
import spacy
import matplotlib.pyplot as plt
from pathlib import Path
from gensim import corpora
from gensim.models import TfidfModel, LdaModel
from gensim.models.coherencemodel import CoherenceModel

FIG_DIR = Path("./figuras")
FIG_DIR.mkdir(exist_ok=True)

# Cargo el modelo de spaCy
nlp = spacy.load("en_core_web_sm")

# Junto título y resumen (si no hay resumen, uso solo el título)
df_corpus['Texto_Completo'] = df_corpus['Titulo'].fillna('') + " " + df_corpus['Resumen'].fillna('')

# Stop words académicas extra, para quedarme solo con lo metodológico
stop_words_academicas = {'background', 'methods', 'results', 'conclusions', 'objective', 
                         'aim', 'study', 'ci', '95%', 'p', 'value', 'method', 'result', 
                         'conclusion', 'author', 'article', 'review', 'systematic', 'use',
                         'analysis', 'data', 'model', 'paper', 'research', 'patient', 'clinical'}

def preprocesar_texto(texto):
    texto_limpio = re.sub(r'[^a-zA-Z\s]', '', str(texto).lower())
    doc = nlp(texto_limpio)
    tokens = []
    for token in doc:
        # solo sustantivos, adjetivos, verbos y adverbios
        if token.pos_ in ['NOUN', 'ADJ', 'VERB', 'ADV']:
            lema = token.lemma_
            if lema not in nlp.Defaults.stop_words and lema not in stop_words_academicas and len(lema) > 2:
                tokens.append(lema)
    return tokens

print("Ejecutando NLP (Limpieza, Lematización y remoción de Stop Words)...")
df_corpus['Tokens'] = df_corpus['Texto_Completo'].apply(preprocesar_texto)

# Paso 4: matriz DTM y optimización de LDA
# 1. Diccionario y filtro de extremos
diccionario = corpora.Dictionary(df_corpus['Tokens'])
diccionario.filter_extremes(no_below=2, no_above=0.85)  # ajuste para corpus mixto
print(f"Vocabulario final tras incluir resúmenes: {len(diccionario)} términos únicos.")

# 2. Matriz de conteos (bag-of-words). Ojo: el LDA se entrena sobre estos conteos
# crudos, no sobre TF-IDF. LDA asume un proceso generativo multinomial sobre
# conteos de palabras (Blei et al., 2003) y ponderar antes por TF-IDF rompe ese
# supuesto. Lo probé sobre este mismo corpus: entrenando con TF-IDF, el 94% de
# los documentos colapsaba en un solo tópico, aunque la coherencia (Cv) daba
# alta igual — esa métrica sola no detecta el colapso. Dejo tfidf/corpus_tfidf
# calculado por si sirve para otra cosa, pero el LDA usa corpus_bow.
corpus_bow = [diccionario.doc2bow(tokens) for tokens in df_corpus['Tokens']]
tfidf = TfidfModel(corpus_bow)
corpus_tfidf = tfidf[corpus_bow]

# 3. Barrido para buscar el k óptimo (coherencia Cv + perplejidad)
rango_k = range(2, 21)
valores_coherencia = []
valores_perplejidad = []
modelos_lda = []

print("\nEntrenando modelos LDA (sobre BOW) y calculando coherencia (Cv) y perplejidad...")
for k in rango_k:
    modelo = LdaModel(corpus=corpus_bow,
                       id2word=diccionario,
                       num_topics=k,
                       random_state=42,  # semilla fija para poder reproducir
                       passes=20)
    modelos_lda.append(modelo)

    # coherencia semántica (Cv): más alta = mejor interpretabilidad
    cv_model = CoherenceModel(model=modelo, texts=df_corpus['Tokens'], 
                              dictionary=diccionario, coherence='c_v', processes=1)
    cv = cv_model.get_coherence()
    valores_coherencia.append(cv)

    # perplejidad: más baja = mejor ajuste estadístico al corpus
    bound = modelo.log_perplexity(corpus_bow)
    perplejidad = 2 ** (-bound)
    valores_perplejidad.append(perplejidad)

    print(f"k={k} tópicos -> Coherencia Cv: {cv:.4f} | Perplejidad: {perplejidad:.2f}")

# 4. Elegir el mejor modelo
# El criterio principal es la coherencia Cv máxima (mejor interpretabilidad).
# La perplejidad se reporta como referencia adicional nomás: en corpus chicos
# suele crecer casi monótonamente con k y no da un mínimo claro por sí sola.
indice_mejor_k = valores_coherencia.index(max(valores_coherencia))
mejor_k = rango_k[indice_mejor_k]
mejor_modelo_lda = modelos_lda[indice_mejor_k]

if mejor_k == rango_k[-1]:
    print(f"\n⚠️ ADVERTENCIA: el k óptimo (k={mejor_k}) cayó en el borde superior del rango probado.")
    print("   La coherencia podría seguir subiendo más allá de este rango; considera ampliarlo de nuevo.")

print(f"\n✅ El modelo óptimo determinado matemáticamente es de k={mejor_k} tópicos.")
print(f"   Coherencia Cv: {valores_coherencia[indice_mejor_k]:.4f} | Perplejidad: {valores_perplejidad[indice_mejor_k]:.2f}")

# Resultados visuales (Cap. 3, Figura 3.4)
# Gráfico con doble eje: Cv a la izquierda, perplejidad a la derecha, y una
# línea vertical marcando el k óptimo.
fig, ax1 = plt.subplots(figsize=(10, 5.5))
ax1.plot(list(rango_k), valores_coherencia, marker='o', color=PASTEL["azul"], linewidth=2,
         markersize=5, label='Coherencia (Cv)')
ax1.set_xlabel('Número de tópicos (k)')
ax1.set_ylabel('Coherencia Semántica (Cv)', color=PASTEL["azul"])
ax1.tick_params(axis='y', labelcolor=PASTEL["azul"])
ax1.set_xticks(list(rango_k))
ax1.grid(axis='y', color=GRID, linewidth=0.8)
ax1.set_axisbelow(True)
ax1.spines['top'].set_visible(False)

ax2 = ax1.twinx()
ax2.plot(list(rango_k), valores_perplejidad, marker='s', color=PASTEL["naranja"], linewidth=2,
         markersize=5, linestyle='--', label='Perplejidad')
ax2.set_ylabel('Perplejidad', color=PASTEL["naranja"])
ax2.tick_params(axis='y', labelcolor=PASTEL["naranja"])
ax2.spines['top'].set_visible(False)

ax1.axvline(mejor_k, color="#C9524B", linewidth=1.5, linestyle=':')
ax1.annotate(f'k={mejor_k} (óptimo)', xy=(mejor_k, max(valores_coherencia)),
             xytext=(mejor_k + 0.4, max(valores_coherencia)), color="#C9524B", fontsize=10, va='center')

lineas1, etiquetas1 = ax1.get_legend_handles_labels()
lineas2, etiquetas2 = ax2.get_legend_handles_labels()
ax1.legend(lineas1 + lineas2, etiquetas1 + etiquetas2, frameon=False, loc='upper left', bbox_to_anchor=(0.02, 0.85))

# no pongo plt.title(): el título va en el pie de figura de Word (Figura 3.4)
plt.tight_layout()
plt.savefig(FIG_DIR / "figura_3_3_coherencia_perplejidad.png", dpi=200)
plt.show()

print(f"\n=== ESTRUCTURA LATENTE DESCUBIERTA (Modelo k={mejor_k}) ===")
# las 8 palabras con más probabilidad (beta) por tópico
for idx, topic in mejor_modelo_lda.print_topics(num_words=8):
    print(f"Tópico {idx + 1}: {topic}")

################################################################################

# Paso 5.1: búsqueda dirigida de términos avanzados (prueba de hipótesis)
# términos que el marco teórico pide pero que el LDA sugiere que faltan
terminos_avanzados = ['bivariate', 'hsroc', 'hierarchical', 'prevalence', 'sroc']
frecuencias_avanzadas = {termino: 0 for termino in terminos_avanzados}

# cuento cuántas veces aparecen realmente en los tokens limpios
for tokens in df_corpus['Tokens']:
    for token in tokens:
        if token in frecuencias_avanzadas:
            frecuencias_avanzadas[token] += 1

print("=== FRECUENCIA DE TÉRMINOS METODOLÓGICOS ROBUSTOS EN EL CORPUS ===")
for termino, freq in frecuencias_avanzadas.items():
    print(f"Término '{termino}': {freq} apariciones en todo el corpus.")

# Paso 5.2: matriz de distribución de tópicos (theta)
import pandas as pd

# saca el tópico dominante, su probabilidad y el vector theta completo
# (proporción de cada tópico) para cada documento
def obtener_topicos_y_theta(modelo, corpus):
    topicos_dominantes, probabilidades, thetas = [], [], []
    for doc in corpus:
        distribucion = modelo.get_document_topics(doc, minimum_probability=0)
        theta_doc = [p for _, p in sorted(distribucion, key=lambda x: x[0])]
        distribucion_ordenada = sorted(distribucion, key=lambda x: x[1], reverse=True)
        topico_dom, prob = distribucion_ordenada[0]
        topicos_dominantes.append(topico_dom + 1)  # 1-indexado
        probabilidades.append(prob)
        thetas.append(theta_doc)
    return topicos_dominantes, probabilidades, thetas

# aplico la función al mejor modelo (sobre corpus_bow, igual que en el entrenamiento)
df_corpus['Topico_Dominante'], df_corpus['Prob_Topico'], thetas = obtener_topicos_y_theta(mejor_modelo_lda, corpus_bow)
theta_cols = [f"theta_t{i+1}" for i in range(mejor_k)]
df_corpus[theta_cols] = pd.DataFrame(thetas, index=df_corpus.index)

print(f"\nDistribución de documentos por tópico dominante (k={mejor_k}):")
print(df_corpus['Topico_Dominante'].value_counts().sort_index())

# Tabla 3.3: tópicos identificados (términos, etiqueta, prevalencia)
# El etiquetado semántico de un tópico no se puede automatizar del todo (Lau
# et al., 2011); estas etiquetas son solo un punto de partida a partir de los
# términos principales, falta validarlas.
n_total = len(df_corpus)
conteo_dominante = df_corpus['Topico_Dominante'].value_counts()

print(f"\n=== TABLA 3.3: TÓPICOS DEL MODELO ÓPTIMO (k={mejor_k}) ===")
filas_topicos = []
for i in range(mejor_k):
    terminos_top = mejor_modelo_lda.show_topic(i, topn=8)
    palabras = ", ".join(w for w, p in terminos_top)
    n_doc = int(conteo_dominante.get(i + 1, 0))
    prevalencia = 100 * n_doc / n_total
    filas_topicos.append({"Topico": f"t_{i+1}", "Terminos_principales": palabras,
                           "N_documentos": n_doc, "Prevalencia_%": round(prevalencia, 1)})
    print(f"t_{i+1} | n={n_doc} ({prevalencia:.1f}%) | {palabras}")

df_topicos = pd.DataFrame(filas_topicos)


# Agrupo los tópicos en dominios temáticos más amplios
# Asignación editorial (mismo caso que la Tabla 3.3, no se automatiza del todo;
# Lau et al., 2011), siguiendo la misma lógica de agrupar tópicos afines en
# dominios que se usa en otros estudios bibliométricos con LDA. También sirve
# para reordenar el eje de tópicos de los heatmaps (paso 5.7) por dominio.
DOMINIOS = {
    1: "Imagenología y Radiómica Diagnóstica",
    2: "Aplicaciones Clínicas Especializadas por Dominio",
    3: "Imagenología y Radiómica Diagnóstica",
    4: "Rigor Metodológico y Síntesis de Evidencia",
    5: "Imagenología y Radiómica Diagnóstica",
    6: "Imagenología y Radiómica Diagnóstica",
    7: "Aplicaciones Clínicas Especializadas por Dominio",
    8: "Rigor Metodológico y Síntesis de Evidencia",
    9: "Aplicaciones Clínicas Especializadas por Dominio",
    10: "Rigor Metodológico y Síntesis de Evidencia",
    11: "Aplicaciones Clínicas Especializadas por Dominio",
    12: "Rigor Metodológico y Síntesis de Evidencia",
    13: "Rigor Metodológico y Síntesis de Evidencia",
    14: "Imagenología y Radiómica Diagnóstica",
}
df_topicos["Dominio"] = df_topicos["Topico"].str.replace("t_", "").astype(int).map(DOMINIOS)

print("\n=== TÓPICOS AGRUPADOS POR DOMINIO ===")
for dominio in df_topicos["Dominio"].unique():
    sub = df_topicos[df_topicos["Dominio"] == dominio]
    prevalencia_dominio = sub["Prevalencia_%"].sum()
    topicos_str = ", ".join(sub["Topico"].str.replace("t_", "t"))
    print(f"- {dominio} ({topicos_str}): {prevalencia_dominio:.1f}% del corpus")

################################################################################

# Paso 5.3: contraste estadístico — modelos avanzados vs. baja prevalencia (Obj. 4)
# Clasifico por presencia de términos en cada documento, no por tópico LDA
# dominante (en el paso 5.2 vi que los tópicos separan por dominio clínico, no
# por enfoque estadístico). El vocabulario queda en inglés sin traducir porque
# el corpus original está en inglés.
from scipy.stats import chi2_contingency, fisher_exact

terminos_modelo_avanzado = {'bivariate', 'hierarchical', 'hsroc', 'sroc'}
terminos_baja_prevalencia = {'prevalence', 'imbalance', 'imbalanced', 'rare'}

def contiene_terminos(tokens, terminos):
    return any(t in terminos for t in tokens)

df_corpus['Menciona_Modelo_Avanzado'] = df_corpus['Tokens'].apply(lambda t: contiene_terminos(t, terminos_modelo_avanzado))
df_corpus['Menciona_Baja_Prevalencia'] = df_corpus['Tokens'].apply(lambda t: contiene_terminos(t, terminos_baja_prevalencia))

tabla_contingencia = pd.crosstab(df_corpus['Menciona_Baja_Prevalencia'], df_corpus['Menciona_Modelo_Avanzado'])
tabla_contingencia.index = ['Sin baja prevalencia', 'Con baja prevalencia']
tabla_contingencia.columns = ['Sin modelo avanzado', 'Con modelo avanzado']

print("=== TABLA DE CONTINGENCIA: baja prevalencia × modelo avanzado (bivariante/HSROC/jerárquico) ===")
print(tabla_contingencia)

# Chi-cuadrado si las frecuencias esperadas alcanzan (todas >= 5); si no, Fisher
# exacta, que es más robusta con muestras chicas.
chi2, p_chi2, dof, esperado = chi2_contingency(tabla_contingencia)
usa_fisher = (esperado < 5).any()

if usa_fisher:
    odds_ratio, p_valor = fisher_exact(tabla_contingencia)
    print(f"\nAlguna frecuencia esperada < 5 -> se usa la prueba exacta de Fisher.")
    print(f"Odds ratio: {odds_ratio:.3f} | p-valor: {p_valor:.4f}")
else:
    print(f"\nChi-cuadrado: {chi2:.3f} | gl: {dof} | p-valor: {p_chi2:.4f}")
    p_valor = p_chi2

alfa = 0.01  # umbral definido en el capítulo 1
if p_valor < alfa:
    print(f"\n✅ Asociación estadísticamente significativa (p<{alfa}): la mención de baja prevalencia")
    print("   SÍ está asociada con el uso de modelos jerárquicos/bivariantes avanzados.")
else:
    print(f"\n⚠️ No hay asociación estadísticamente significativa (p>={alfa}): la mención de baja prevalencia")
    print("   NO está asociada de forma significativa con el uso de modelos avanzados —")
    print("   consistente con la hipótesis de que los estudios de baja prevalencia no adoptan")
    print("   rutinariamente los modelos estadísticos jerárquicos recomendados.")

n_con_prevalencia = tabla_contingencia.loc['Con baja prevalencia'].sum()
n_con_ambos = tabla_contingencia.loc['Con baja prevalencia', 'Con modelo avanzado']
if n_con_prevalencia > 0:
    print(f"\nDe {n_con_prevalencia} documentos que mencionan baja prevalencia/desbalance de clases,")
    print(f"{n_con_ambos} ({100*n_con_ambos/n_con_prevalencia:.1f}%) también mencionan un modelo jerárquico/bivariante avanzado.")
else:
    print("\nNingún documento del corpus menciona baja prevalencia/desbalance de clases explícitamente.")

################################################################################

# Paso 5.4: exportar la matriz de resultados a Excel
columnas_exportar = ['DOI', 'Titulo', 'Anio', 'Fuente', 'Revista', 'Pais_ISO3',
                      'Topico_Dominante', 'Prob_Topico',
                      'Menciona_Modelo_Avanzado', 'Menciona_Baja_Prevalencia']
df_resultados = df_corpus[columnas_exportar].copy()

ruta_exportacion = './Resultados_LDA_Tesis.xlsx'
df_resultados.to_excel(ruta_exportacion, index=False)

print(f"✅ Matriz de resultados exportada exitosamente a: {ruta_exportacion}")
print("Puedes abrir este Excel para ver cómo se clasificó cada artículo matemáticamente.")

################################################################################

# Paso 5.5: Tabla 3.2 - distribución por revista
# PubMed da el nombre de revista abreviado (norma NLM) y WoS el nombre
# completo; los dejo tal cual los entrega cada fuente.
tabla_revistas = df_corpus.groupby('Revista').size().reset_index(name='n').sort_values('n', ascending=False)
principales = tabla_revistas[tabla_revistas['n'] >= 2]
resto_n_revistas = len(tabla_revistas) - len(principales)
resto_n_articulos = tabla_revistas[tabla_revistas['n'] < 2]['n'].sum()

print(f"Revistas distintas en el corpus: {len(tabla_revistas)}")
print(f"\n=== TABLA 3.2: REVISTAS CON 2 O MÁS ARTÍCULOS ===")
print(principales.to_string(index=False))
print(f"\nOtras revistas ({resto_n_revistas} revistas, 1 artículo cada una): {resto_n_articulos} artículos")

################################################################################

# Paso 5.6: Figura 3.5 - tendencia de tópicos en el tiempo (pequeños múltiplos)
import numpy as np
from scipy import stats as spstats
from matplotlib.lines import Line2D

COLOR_UP, COLOR_DOWN, COLOR_STABLE = "#C9524B", "#3E6FA8", "#9A9993"

tendencia = df_corpus.groupby('Anio')[theta_cols].mean()
anios_arr = tendencia.index.values

clasificacion_tendencia = {}
for i, col in enumerate(theta_cols, start=1):
    if len(anios_arr) >= 3:
        slope, intercept, r, p, se = spstats.linregress(anios_arr, tendencia[col].values)
        if p < 0.05 and slope > 0:
            clasificacion_tendencia[i] = 'creciente'
        elif p < 0.05 and slope < 0:
            clasificacion_tendencia[i] = 'decreciente'
        else:
            clasificacion_tendencia[i] = 'estable'
    else:
        clasificacion_tendencia[i] = 'estable'

color_map = {'creciente': COLOR_UP, 'decreciente': COLOR_DOWN, 'estable': COLOR_STABLE}

# Un mini-gráfico por tópico (pequeños múltiplos), cada uno con su propia
# escala, en vez de una sola figura con todas las líneas superpuestas
ncols = 5
nrows = int(np.ceil(mejor_k / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.3 * nrows))
axes = np.array(axes).reshape(-1)

for i, col in enumerate(theta_cols):
    ax = axes[i]
    topico_num = i + 1
    ax.plot(anios_arr, tendencia[col].values, color=color_map[clasificacion_tendencia[topico_num]], linewidth=1.4)
    ax.set_title(f"t_{topico_num}", fontsize=10, fontweight='bold')
    ax.tick_params(axis='both', labelsize=6.5)
    ax.grid(color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    xt = anios_arr[::max(1, len(anios_arr)//5)]
    ax.set_xticks(xt)
    ax.set_xticklabels([str(int(x)) for x in xt], rotation=45, fontsize=6)

for j in range(mejor_k, len(axes)):
    axes[j].axis('off')

leyenda = [Line2D([0], [0], color=COLOR_UP, lw=2, label='Creciente (p<0.05)'),
           Line2D([0], [0], color=COLOR_DOWN, lw=2, label='Decreciente (p<0.05)'),
           Line2D([0], [0], color=COLOR_STABLE, lw=2, label='Estable')]
fig.legend(handles=leyenda, frameon=False, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02))
# no pongo plt.title()/suptitle(): el título va en el pie de figura de Word (Figura 3.5)
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(FIG_DIR / "figura_3_tendencia_topicos.png", dpi=200, bbox_inches='tight')
plt.show()

print("Clasificación de tendencia por tópico:", clasificacion_tendencia)

################################################################################

# Paso 5.7: Figuras 3.6, 3.7 y 3.8 - heatmaps tópicos x año, x revista, x país
# Si falta algo: %pip install seaborn scikit-learn
# El eje horizontal siempre es el tópico. Las columnas se reordenan (sin volver
# a clusterizar) agrupando por el dominio temático del paso 5.2, con el nombre
# del dominio y una línea separadora arriba del heatmap. El dendrograma agrupa
# las categorías del eje vertical (año/revista/país) por su perfil de tópicos y
# va contra el eje Y, no el X. El número de clústeres no lo fijo a mano: pruebo
# particiones de 2 a 6 y me quedo con la que da mejor silueta promedio (Kaufman
# & Rousseeuw, 1990); la línea roja punteada marca dónde quedó el corte. Cada
# bloque de clúster se numera junto al dendrograma y se separa con una línea
# blanca en el heatmap.
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score

CMAP_SEQ = LinearSegmentedColormap.from_list("pastel_seq", ["#DCE7F5", PASTEL["azul"], "#274A75"])

# Orden de los dominios (de más a menos relevante para la pregunta de
# investigación de la tesis) y el orden de columnas que sale de ahí
ORDEN_DOMINIOS = ["Rigor Metodológico y Síntesis de Evidencia",
                   "Imagenología y Radiómica Diagnóstica",
                   "Aplicaciones Clínicas Especializadas por Dominio"]
ORDEN_TOPICOS_DOMINIO = sorted(
    [f"t_{i+1}" for i in range(mejor_k)],
    key=lambda c: (ORDEN_DOMINIOS.index(DOMINIOS[int(c.split('_')[1])]), int(c.split('_')[1]))
)


def mejor_corte(matriz_categorias):
    """Determina k=2..6 que maximiza la silueta promedio sobre un clustering
    jerárquico (Ward) de las categorías (filas), y la altura de corte asociada."""
    Z = linkage(matriz_categorias, method='ward')
    mejor_k_clust, mejor_score = None, -1
    max_k = min(6, matriz_categorias.shape[0] - 1)
    for k_clust in range(2, max_k + 1):
        labels = fcluster(Z, t=k_clust, criterion='maxclust')
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(matriz_categorias, labels)
        if score > mejor_score:
            mejor_score, mejor_k_clust = score, k_clust
    alturas = np.sort(Z[:, 2])
    n = matriz_categorias.shape[0]
    idx_corte = n - mejor_k_clust
    altura = (alturas[idx_corte - 1] + alturas[idx_corte]) / 2 if 0 < idx_corte < len(alturas) else alturas[-1]
    return Z, mejor_k_clust, mejor_score, altura


def etiquetar_clusters(cg, Z, k_clust, n_filas):
    """Numera cada bloque de clúster (1..k, de arriba hacia abajo) junto al
    dendrograma y dibuja una línea blanca en el heatmap entre bloques contiguos."""
    orden = cg.dendrogram_row.reordered_ind
    labels_originales = fcluster(Z, t=k_clust, criterion='maxclust')
    labels_orden = [labels_originales[i] for i in orden]

    mapa_id, siguiente_id = {}, 1
    for lab in labels_orden:
        if lab not in mapa_id:
            mapa_id[lab] = siguiente_id
            siguiente_id += 1
    ids_orden = [mapa_id[lab] for lab in labels_orden]

    bloques, inicio = [], 0
    for i in range(1, n_filas + 1):
        if i == n_filas or ids_orden[i] != ids_orden[inicio]:
            bloques.append((inicio, i, ids_orden[inicio]))
            inicio = i

    for ini, fin, cid in bloques:
        centro = (ini + fin) / 2
        cg.ax_row_dendrogram.text(-0.14, 1 - centro / n_filas, str(cid),
                                   transform=cg.ax_row_dendrogram.transAxes,
                                   ha='center', va='center', fontsize=13,
                                   fontweight='bold', color='#333333', clip_on=False)
        if ini > 0:
            cg.ax_heatmap.axhline(ini, color='white', linewidth=4)


# Nombres cortos para la franja de arriba del heatmap (los nombres completos de
# DOMINIOS no caben en el ancho de columna, sobre todo en la Figura 3.7 con 15
# filas); el nombre completo va en la Tabla 3.3 y en el texto.
NOMBRE_CORTO_DOMINIO = {
    "Rigor Metodológico y Síntesis de Evidencia": "Rigor Metodológico",
    "Imagenología y Radiómica Diagnóstica": "Imagenología/Radiómica",
    "Aplicaciones Clínicas Especializadas por Dominio": "Aplic. Clínicas",
}


def etiquetar_dominios(cg, columnas_ordenadas):
    """Rotula, en la franja superior del heatmap, los bloques contiguos de
    columnas que pertenecen al mismo dominio temático, con una línea separadora."""
    doms = [NOMBRE_CORTO_DOMINIO[DOMINIOS[int(c.split('_')[1])]] for c in columnas_ordenadas]
    n_cols = len(columnas_ordenadas)
    bloques, inicio = [], 0
    for i in range(1, n_cols + 1):
        if i == n_cols or doms[i] != doms[inicio]:
            bloques.append((inicio, i, doms[inicio]))
            inicio = i

    ax_top = cg.ax_col_dendrogram
    ax_top.clear()
    ax_top.set_xlim(0, n_cols)
    ax_top.axis('off')
    for ini, fin, dom in bloques:
        centro = (ini + fin) / 2
        ax_top.text(centro, 0.5, dom, ha='center', va='center', fontsize=7.5,
                     fontweight='bold', color='#333333', wrap=True)
        if ini > 0:
            cg.ax_heatmap.axvline(ini, color='white', linewidth=3)
            ax_top.axvline(ini, color='#999999', linewidth=0.8)


def heatmap_topicos_horizontal(matriz_cat_x_topico, xlabel, filename, figsize):
    """matriz_cat_x_topico: filas=categoría (año/revista/país), columnas=tópicos.
    Las columnas llegan reordenadas por dominio; el dendrograma agrupa las FILAS
    y va contra el eje Y."""
    matriz_cat_x_topico = matriz_cat_x_topico[ORDEN_TOPICOS_DOMINIO]
    Z, k_clust, score, altura = mejor_corte(matriz_cat_x_topico.values)
    cg = sns.clustermap(matriz_cat_x_topico, cmap=CMAP_SEQ, row_cluster=True, col_cluster=False,
                         row_linkage=Z, figsize=figsize, cbar_kws={'label': 'Proporción media (θ)'},
                         linewidths=0.3, linecolor='white', dendrogram_ratio=(0.18, 0.13),
                         cbar_pos=(1.02, 0.3, 0.025, 0.45))  # leyenda a la derecha
    cg.ax_heatmap.set_xlabel(xlabel)
    cg.ax_heatmap.set_ylabel('')
    etiquetar_dominios(cg, ORDEN_TOPICOS_DOMINIO)
    if cg.ax_row_dendrogram is not None:
        cg.ax_row_dendrogram.axvline(altura, color="#C9524B", linewidth=1, linestyle='--')
        etiquetar_clusters(cg, Z, k_clust, matriz_cat_x_topico.shape[0])
    cg.savefig(FIG_DIR / filename, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"  -> corte en {k_clust} clústeres (silueta={score:.3f}, altura={altura:.3f})")
    return k_clust, score, altura

# Figura 3.6: tópicos x año
matriz_anio = df_corpus.groupby('Anio')[theta_cols].mean()
matriz_anio.columns = [f"t_{i+1}" for i in range(mejor_k)]
print("Figura 3.6 (tópicos x año):")
heatmap_topicos_horizontal(matriz_anio, "Tópico", "figura_3_heatmap_anio.png", (10, 5))

# Figura 3.7: tópicos x revista (top 15)
tabla_rev_hm = df_corpus.groupby('Revista').size().reset_index(name='n').sort_values('n', ascending=False)
top_revistas = tabla_rev_hm.head(15)['Revista'].tolist()
df_top_revistas = df_corpus[df_corpus['Revista'].isin(top_revistas)]
matriz_revista = df_top_revistas.groupby('Revista')[theta_cols].mean()
matriz_revista = matriz_revista.loc[[r for r in top_revistas if r in matriz_revista.index]]
matriz_revista.columns = [f"t_{i+1}" for i in range(mejor_k)]
print("Figura 3.7 (tópicos x revista, top 15):")
heatmap_topicos_horizontal(matriz_revista, "Tópico", "figura_3_heatmap_revista.png", (10, 7))

# Figura 3.8: tópicos x país (top 12)
top_paises_iso = conteo_pais.sort_values('n', ascending=False).head(12)['ISO_A3'].tolist()
df_top_paises = df_corpus[df_corpus['Pais_ISO3'].isin(top_paises_iso)]
matriz_pais = df_top_paises.groupby('Pais_ISO3')[theta_cols].mean()
matriz_pais.index = [nombre_por_iso.get(iso, iso) for iso in matriz_pais.index]
matriz_pais.columns = [f"t_{i+1}" for i in range(mejor_k)]
orden_paises = [nombre_por_iso.get(iso, iso) for iso in top_paises_iso if nombre_por_iso.get(iso, iso) in matriz_pais.index]
matriz_pais = matriz_pais.loc[orden_paises]
print("Figura 3.8 (tópicos x país, top 12):")
heatmap_topicos_horizontal(matriz_pais, "Tópico", "figura_3_heatmap_pais.png", (10, 6))

print("\nFiguras 3.6, 3.7 y 3.8 (heatmaps) guardadas en:", FIG_DIR)

################################################################################

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# Configuración de la figura en alta resolución
fig, ax = plt.subplots(figsize=(16, 6))
ax.axis('off')

# Datos de cada paso, con el ícono en LaTeX para el centro del círculo
pasos = [
    {"titulo": "Texto Crudo", "desc": "Título y resumen\ndel artículo", "icono": r"$\mathcal{T}$"},
    {"titulo": "Limpieza", "desc": "Minúsculas y\nfiltro alfabético", "icono": r"$\star$"},
    {"titulo": "Tokenización", "desc": "Segmentación\nen palabras", "icono": r"$\{w_i\}$"},
    {"titulo": "Lematización", "desc": "Forma canónica\ny filtro gramatical", "icono": r"$\rightarrow$"},
    {"titulo": "Remoción", "desc": "Filtro de\nstop words", "icono": r"$\emptyset$"},
    {"titulo": "Matriz DTM", "desc": "Resultado final:\nDocumento-Término", "icono": r"$M_{d,t}$"}
]

n = len(pasos)
x = np.linspace(0, 10, n)
# alterno posiciones en Y (onda marcada para dejar espacio a los textos)
y = np.array([1.2 if i % 2 == 0 else -1.2 for i in range(n)])

# curva suave (spline) para la línea que conecta los círculos
x_smooth = np.linspace(x.min(), x.max(), 300)
spl = make_interp_spline(x, y, k=3)
y_smooth = spl(x_smooth)

# línea conectora: sombra gris gruesa + línea de color fina encima
ax.plot(x_smooth, y_smooth, color='#e0e0e0', lw=10, zorder=1)
ax.plot(x_smooth, y_smooth, color='#1f4e79', lw=3, zorder=2, alpha=0.6)

for i in range(n):
    # azul oscuro para los pasos intermedios, naranja para el resultado final
    color_borde = '#c55a11' if i == n-1 else '#1f4e79'
    
    # círculo principal
    circle_out = plt.Circle((x[i], y[i]), 0.55, color=color_borde, zorder=3)
    circle_in = plt.Circle((x[i], y[i]), 0.48, color='white', zorder=4)
    ax.add_patch(circle_out)
    ax.add_patch(circle_in)
    
    # ícono adentro del círculo
    ax.text(x[i], y[i], pasos[i]['icono'], ha='center', va='center', 
            fontsize=26, color=color_borde, zorder=5)
    
    # medalla con el número, arriba a la derecha del círculo
    badge_x = x[i] + 0.42
    badge_y = y[i] + 0.42
    badge_bg = plt.Circle((badge_x, badge_y), 0.22, color=color_borde, zorder=6)
    ax.add_patch(badge_bg)
    
    # número dentro de la medalla
    ax.text(badge_x, badge_y, str(i+1), ha='center', va='center', 
            fontsize=12, fontweight='bold', color='white', zorder=7, family='sans-serif')
    
    # título y descripción, afuera del círculo
    offset = 0.95
    va_val = 'bottom' if y[i] > 0 else 'top'
    y_text = y[i] + offset if y[i] > 0 else y[i] - offset
    
    # título en negrita
    ax.text(x[i], y_text, pasos[i]['titulo'], ha='center', va=va_val,
            fontsize=13, fontweight='bold', color='#2c3e50', family='sans-serif')
    
    # descripción, debajo o encima del título según toque
    y_desc = y_text + 0.28 if y[i] > 0 else y_text - 0.28
    ax.text(x[i], y_desc, pasos[i]['desc'], ha='center', va=va_val,
            fontsize=11, color='#595959', family='sans-serif')

# márgenes para que no se corte nada
plt.xlim(-1.5, 11.5)
plt.ylim(-4, 4)
plt.tight_layout()

# guardo la imagen
plt.savefig('ruta_nlp_con_iconos.png', dpi=600, transparent=True, bbox_inches='tight')
print("Imagen 'ruta_nlp_con_iconos.png' generada con éxito.")

################################################################################

# Bloque experimental (todavía no va en el capítulo III): diagrama de flujo tipo
# PRISMA para el proceso de identificación/consolidación del corpus. Lo armé
# para revisarlo antes de decidir si lo sumo como figura en la sección 3.2. No
# reemplaza a la Tabla 3.1, es solo otra forma de mostrar lo mismo. Los conteos
# están fijos con los valores del corpus actual (n=94); si el corpus cambia hay
# que actualizarlos a mano.
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

PASTEL = {"azul": "#4E7FBF", "naranja": "#D97B4A", "verde": "#3CA57F",
          "ambar": "#D9A22E", "violeta": "#8A76CF"}

fig, ax = plt.subplots(figsize=(11, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

def caja(x, y, w, h, texto, color_borde, fill="#fcfcfb", fontsize=9.5):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                          linewidth=1.3, edgecolor=color_borde, facecolor=fill, zorder=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, texto, ha='center', va='center', fontsize=fontsize,
            color="#222222", zorder=3, linespacing=1.4)

def flecha(x1, y1, x2, y2, color="#666666"):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                 arrowprops=dict(arrowstyle='-|>', color=color, lw=1.3), zorder=1)

fases = [("Identificación", 10.4), ("Consolidación", 7.6), ("Recuperación de datos", 4.9), ("Corpus final", 1.9)]
for texto, y in fases:
    ax.text(-1.35, y, texto, ha='left', va='center', fontsize=10.5, fontweight='bold',
            color="#333333", rotation=90)

caja(0.6, 10.0, 3.6, 1.1, "PubMed\nRegistros identificados\n(n = 78)", PASTEL["azul"])
caja(5.8, 10.0, 3.6, 1.1, "Web of Science\nRegistros identificados\n(n = 50)", PASTEL["naranja"])
flecha(2.4, 10.0, 4.6, 8.7)
flecha(7.6, 10.0, 5.4, 8.7)

caja(2.7, 7.6, 4.6, 1.1, "Registros consolidados\n(n = 128)", "#999999")
flecha(5.0, 7.6, 5.0, 6.9)

caja(6.9, 6.4, 3.0, 1.5,
     "Duplicados eliminados\n(DOI exacto + título\nnormalizado + RapidFuzz\ntoken_sort ≥ 92%)\n(n = 34)",
     "#B0473F", fill="#FBEDEC", fontsize=8.6)
flecha(6.9, 7.0, 5.0, 6.9, color="#B0473F")

caja(2.7, 5.3, 4.6, 1.1, "Registros únicos\n(n = 94)", PASTEL["verde"])
flecha(5.0, 5.3, 5.0, 4.6)

caja(2.0, 3.6, 2.8, 1.0, "Resúmenes recuperados\nvía NCBI Entrez\n94/94 (100%)", PASTEL["violeta"], fontsize=8.6)
caja(5.2, 3.6, 2.8, 1.0, "País de afiliación\nidentificado\n85/94 (90%)", PASTEL["violeta"], fontsize=8.6)
flecha(3.4, 3.6, 5.0, 2.9)
flecha(6.6, 3.6, 5.0, 2.9)

caja(2.7, 1.4, 4.6, 1.3, "Corpus final incluido\n(n = 94)\nPubMed = 57  |  WoS = 37", PASTEL["ambar"], fontsize=9.8)

plt.tight_layout()
plt.savefig(FIG_DIR / "figura_3_0_flujo_EXPERIMENTAL.png", dpi=200, bbox_inches='tight', facecolor='white')
plt.show()
print("Diagrama de flujo (experimental) guardado. Revisar antes de decidir si se integra al Capítulo III.")
