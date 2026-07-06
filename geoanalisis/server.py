#!/usr/bin/env python3
"""GeoAnalisis MCP Server — Herramientas de lectura y análisis de datos espaciales vectoriales."""

from __future__ import annotations

import html
import json
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from mcp.server.fastmcp import FastMCP
from mcp.types import EmbeddedResource, ImageContent, TextContent, TextResourceContents
from shapely import set_precision, to_wkt


mcp = FastMCP(
    "GeoAnalisis",
    instructions=(
        "Servidor MCP para análisis de datos espaciales vectoriales. "
        "Soporta FileGDB (.gdb), Shapefile (.shp), GeoJSON, GeoPackage (.gpkg), KML y "
        "cualquier formato vectorial compatible con GDAL/OGR. "
        "REGLA DE CRS: nunca descartar una capa por no tener CRS. Todas las capas se "
        "reproyectan automáticamente al render; si un archivo no define CRS, inspeccionar "
        "sus bounds (get_layer_schema) y pasar source_crs (capa principal) o la clave 'crs' "
        "(extra_layers) con el EPSG correcto — en Colombia: EPSG:4326/4686 si los bounds son "
        "grados, EPSG:3116 o EPSG:9377 si son metros. El reporte de salida indica el CRS "
        "usado por capa y si fue asumido por heurística. "
        "REGLA DE SIMBOLOGÍA al generar mapas: asignar SIEMPRE un 'color' explícito y "
        "visualmente distinto (matiz diferente) a cada capa de extra_layers — no dejar el "
        "gris por defecto salvo capas de puro contexto. Colores similares entre capas o "
        "énfasis visual en una capa solo si el usuario lo pide explícitamente. "
        "Las herramientas de mapa devuelven un reporte de la simbología realmente aplicada: "
        "verificarlo antes de afirmar al usuario que un color fue cambiado."
    ),
)

_PALETTE = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#34495e", "#e91e63", "#00bcd4",
    "#8bc34a", "#ff5722", "#607d8b", "#795548", "#673ab7",
]


def _to_rgba(color) -> tuple:
    """Normaliza cualquier color matplotlib (nombre, hex, tuple RGB/RGBA) a RGBA tuple.

    gdf.plot(color=<Series de strings hex>) falla porque numpy convierte la Series
    a array de caracteres en lugar de array N×4. Pasar RGBA tuples evita ese bug.
    """
    import matplotlib.colors as mcolors
    try:
        return tuple(mcolors.to_rgba(color))
    except (ValueError, TypeError):
        return (0.667, 0.667, 0.667, 1.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json(obj) -> str:
    """json.dumps con soporte para tipos numpy y valores NaN."""
    def default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    return json.dumps(obj, indent=2, ensure_ascii=False, default=default)


def _parse_crs(crs_raw):
    """Normaliza el CRS de pyogrio.read_info a objeto pyproj.

    pyogrio ≥ 0.12 devuelve el CRS como string ("EPSG:4326"); versiones
    anteriores devolvían un objeto pyproj.CRS. Acepta ambos.
    """
    if crs_raw is None:
        return None
    from pyproj import CRS
    try:
        return CRS.from_user_input(crs_raw)
    except Exception:
        return None


def _read_gdf(
    path: str,
    layer: Optional[str],
    rows: int,
    where: Optional[str],
    bbox: Optional[list[float]],
) -> gpd.GeoDataFrame:
    kwargs: dict = {"rows": rows, "engine": "pyogrio"}
    if layer:
        kwargs["layer"] = layer
    if where:
        kwargs["where"] = where
    if bbox:
        kwargs["bbox"] = tuple(bbox)
    return gpd.read_file(path, **kwargs)


_CRS_HINT = (
    "Inspeccionar los bounds con get_layer_schema y pasar el CRS correcto "
    "(source_crs en la capa principal, clave 'crs' en extra_layers). "
    "Comunes en Colombia: EPSG:4326 (WGS84 lon/lat), EPSG:4686 (MAGNA-SIRGAS "
    "geográfico), EPSG:3116 (MAGNA origen Bogotá), EPSG:9377 (CTM12 origen "
    "nacional, coords ~ millones de metros)."
)


def _crs_label(crs) -> str:
    """'EPSG:4326' / 'ESRI:102233' si hay autoridad; si no, el nombre del CRS."""
    auth = crs.to_authority() if crs else None
    return f"{auth[0]}:{auth[1]}" if auth else (crs.name if crs else "Sin CRS")


def _ensure_crs(
    gdf: gpd.GeoDataFrame, declared: Optional[str], label: str
) -> tuple[gpd.GeoDataFrame, str]:
    """Harness de CRS: garantiza que la capa tenga CRS antes de reproyectar.

    Prioridad: CRS del archivo > CRS declarado por el cliente > heurística
    lon/lat (bounds dentro de ±180/±90 → EPSG:4326). Si nada aplica, lanza
    ValueError con instrucciones accionables en vez de omitir en silencio.

    Returns:
        (gdf con CRS garantizado, nota para el reporte de simbología)
    """
    if gdf.crs is not None:
        note = f"CRS {_crs_label(gdf.crs)} (del archivo)"
        if declared:
            try:
                import pyproj
                if pyproj.CRS(declared) != pyproj.CRS(gdf.crs):
                    note += (f" — se IGNORÓ el declarado {declared}: "
                             f"el archivo ya define su CRS")
            except Exception:
                pass
        return gdf, note

    if declared:
        return gdf.set_crs(declared), f"sin CRS en archivo → asumido {declared} (declarado)"

    b = gdf.total_bounds
    if (-180.0 <= b[0] <= 180.0 and -180.0 <= b[2] <= 180.0
            and -90.0 <= b[1] <= 90.0 and -90.0 <= b[3] <= 90.0):
        return gdf.set_crs(4326), (
            "sin CRS en archivo → bounds parecen lon/lat, asumido EPSG:4326 "
            "(heurística: VERIFICAR)"
        )

    raise ValueError(
        f"La capa '{label}' no tiene CRS y sus bounds {b.tolist()} no parecen "
        f"lon/lat, imposible asumir uno. {_CRS_HINT}"
    )


# Presupuesto de caracteres para el GeoJSON inline.
# El HTML + JS del template ocupa ~18 KB; el límite de output del tool en Claude
# es ~55 K caracteres, dejando ~37 K para los datos.
_MAX_GEOJSON_CHARS = 37_000

# Límite de caracteres para la salida GeoJSON de read_features.
_MAX_FEATURES_CHARS = 45_000


def _js_embed(s: str) -> str:
    """Escapa '</' para embeber JSON dentro de un <script> de forma segura."""
    return s.replace("</", "<\\/")


def _geojson_for_render(gdf: gpd.GeoDataFrame) -> tuple[str, str | None]:
    """
    Convierte un GeoDataFrame a GeoJSON string apto para embeber en HTML.

    Aplica simplificación geométrica progresiva hasta que el resultado cabe
    dentro de _MAX_GEOJSON_CHARS. Si con la máxima simplificación sigue siendo
    demasiado grande, reduce el número de features y lo indica en el warning.

    Returns:
        (geojson_str, warning_message | None)
    """
    n_original = len(gdf)

    # Tolerancias en grados WGS84: 0.0001° ≈ 10 m, 0.001° ≈ 100 m
    tolerances = [0.0, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]

    for tol in tolerances:
        g = gdf if tol == 0.0 else gdf.copy()
        if tol > 0.0:
            g["geometry"] = gdf["geometry"].simplify(tol, preserve_topology=True)
            # Redondear coordenadas al mismo orden de magnitud que la tolerancia
            precision = max(0.00001, tol / 10)
            g["geometry"] = g["geometry"].apply(
                lambda geom: set_precision(geom, precision) if geom and not geom.is_empty else geom
            )

        js = g.to_json(na="null")
        if len(js) <= _MAX_GEOJSON_CHARS:
            warn = (
                f"Geometrías simplificadas (tolerancia {tol}°) para ajustar al límite de tamaño."
                if tol > 0.0 else None
            )
            return js, warn

    # Aún demasiado grande: reducir número de features progresivamente
    for ratio in [0.5, 0.25, 0.1]:
        n = max(1, int(n_original * ratio))
        g = gdf.iloc[:n].copy()
        g["geometry"] = g["geometry"].simplify(0.01, preserve_topology=True)
        js = g.to_json(na="null")
        if len(js) <= _MAX_GEOJSON_CHARS:
            return js, f"Se renderizaron {n} de {n_original} features por límite de tamaño."

    # Fallback final
    g = gdf.iloc[:50].copy()
    return g.to_json(na="null"), f"Se renderizaron 50 de {n_original} features por límite de tamaño."


# ---------------------------------------------------------------------------
# Herramientas
# ---------------------------------------------------------------------------

@mcp.tool()
def list_layers(path: str) -> str:
    """
    Lista todas las capas disponibles en una fuente de datos espacial.

    Soporta .gdb (FileGDB), .gpkg, .shp, .geojson, .kml y cualquier formato GDAL/OGR.
    Devuelve por capa: nombre, tipo de geometría, número de features y CRS.

    Args:
        path: Ruta absoluta al archivo o directorio (.gdb es un directorio).
    """
    layers_arr = pyogrio.list_layers(path)
    result = []
    for name, geom_type in layers_arr:
        entry: dict = {"name": name, "geometry_type": geom_type}
        try:
            info = pyogrio.read_info(path, layer=name)
            crs = _parse_crs(info.get("crs"))
            entry["feature_count"] = info.get("features")
            entry["crs"] = crs.to_string() if crs else "Sin CRS"
        except Exception as e:
            entry["warning"] = str(e)
        result.append(entry)
    return _json(result)


@mcp.tool()
def get_layer_schema(path: str, layer: Optional[str] = None) -> str:
    """
    Obtiene el esquema completo de una capa espacial.

    Devuelve: nombre de la capa, driver GDAL, tipo de geometría, número de features,
    CRS (código EPSG + WKT), extensión espacial (bbox) y lista de campos con sus tipos.

    Args:
        path:  Ruta al archivo o directorio espacial.
        layer: Nombre de la capa. Si es None usa la primera capa disponible.
    """
    info = pyogrio.read_info(path, layer=layer)

    crs = _parse_crs(info.get("crs"))
    bounds = info.get("total_bounds")
    resolved_layer = layer or pyogrio.list_layers(path)[0][0]

    crs_info: dict = {"description": "Sin CRS"}
    if crs:
        crs_info = {"string": crs.to_string()}
        auth = crs.to_authority()
        if auth:
            crs_info["authority"] = f"{auth[0]}:{auth[1]}"
        crs_info["name"] = crs.name

    schema = {
        "layer": resolved_layer,
        "driver": info.get("driver"),
        "geometry_type": info.get("geometry_type"),
        "feature_count": info.get("features"),
        "crs": crs_info,
        "bbox": (
            {
                "xmin": float(bounds[0]),
                "ymin": float(bounds[1]),
                "xmax": float(bounds[2]),
                "ymax": float(bounds[3]),
            }
            if bounds is not None
            else None
        ),
        "fields": [
            {"name": str(name), "type": str(dtype)}
            for name, dtype in zip(info["fields"], info["dtypes"])
        ],
    }
    return _json(schema)


@mcp.tool()
def scan_field_stats(
    path: str,
    layer: Optional[str] = None,
    fields: Optional[list[str]] = None,
    max_features: int = 50000,
) -> str:
    """
    Calcula estadísticas descriptivas por campo en una capa espacial.

    Campos numéricos → min, max, media, desviación estándar, nulos.
    Campos de texto/categoría → conteo de únicos, muestra de valores, nulos.

    Args:
        path:         Ruta al archivo o directorio espacial.
        layer:        Nombre de la capa (None = primera capa).
        fields:       Lista de campos a analizar. None = todos los campos no-geometría.
        max_features: Límite de features a leer para calcular estadísticas (default 50000).
    """
    read_kwargs: dict = {"layer": layer, "rows": max_features, "engine": "pyogrio"}
    if fields:
        # Leer solo las columnas solicitadas (las inexistentes se reportan abajo)
        available = set(pyogrio.read_info(path, layer=layer)["fields"])
        read_kwargs["columns"] = [f for f in fields if f in available]
    gdf = gpd.read_file(path, **read_kwargs)
    geom_col = gdf.geometry.name

    target_cols = fields if fields else [c for c in gdf.columns if c != geom_col]

    stats: dict = {}
    for col in target_cols:
        if col not in gdf.columns:
            stats[col] = {"error": "Campo no encontrado en la capa"}
            continue

        s = gdf[col]
        total = len(s)
        null_count = int(s.isna().sum())

        entry: dict = {
            "dtype": str(s.dtype),
            "total_features": total,
            "null_count": null_count,
            "null_pct": round(null_count / total * 100, 2) if total else 0,
            "unique_count": int(s.nunique(dropna=True)),
        }

        if pd.api.types.is_numeric_dtype(s):
            valid = s.dropna()
            if len(valid):
                entry["min"] = float(valid.min())
                entry["max"] = float(valid.max())
                entry["mean"] = round(float(valid.mean()), 6)
                entry["std"] = round(float(valid.std()), 6)
        else:
            raw_samples = s.dropna().unique()[:10].tolist()
            entry["sample_values"] = [str(v) for v in raw_samples]

        stats[col] = entry

    return _json(stats)


@mcp.tool()
def read_features(
    path: str,
    layer: Optional[str] = None,
    limit: int = 10,
    where: Optional[str] = None,
    bbox: Optional[list[float]] = None,
) -> str:
    """
    Lee features de una capa y los devuelve como GeoJSON FeatureCollection.

    Args:
        path:  Ruta al archivo o directorio espacial.
        layer: Nombre de la capa (None = primera capa).
        limit: Número máximo de features a retornar (default 10).
        where: Filtro SQL OGR, ej: "POBLACION > 5000 AND TIPO = 'urbano'".
        bbox:  Filtro espacial [xmin, ymin, xmax, ymax] en el CRS de la capa.

    Si el resultado excede el límite de tamaño de salida devuelve un error con
    sugerencias (reducir limit, filtrar con where/bbox).
    """
    if bbox and len(bbox) != 4:
        return _json({"error": "bbox debe ser una lista de 4 números: [xmin, ymin, xmax, ymax]"})

    gdf = _read_gdf(path, layer, limit, where, bbox)
    js = gdf.to_json(na="null")
    if len(js) > _MAX_FEATURES_CHARS:
        return _json({
            "error": (
                f"El GeoJSON resultante ({len(js):,} caracteres, {len(gdf)} features) "
                f"excede el límite de {_MAX_FEATURES_CHARS:,} caracteres."
            ),
            "hint": (
                "Reduce `limit`, filtra con `where`/`bbox`, o usa preview_geometries / "
                "scan_field_stats para inspeccionar. Para visualizar usa render_map."
            ),
        })
    return js


@mcp.tool()
def preview_geometries(
    path: str,
    layer: Optional[str] = None,
    limit: int = 5,
) -> str:
    """
    Previsualiza las geometrías de los primeros N features como WKT.

    Devuelve para cada feature: índice, tipo de geometría, WKT de la geometría
    y su bounding box individual.

    Args:
        path:  Ruta al archivo o directorio espacial.
        layer: Nombre de la capa (None = primera capa).
        limit: Número de features a previsualizar (default 5).
    """
    gdf = _read_gdf(path, layer, limit, None, None)

    previews = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        entry: dict = {"index": int(idx)}

        if geom is None or geom.is_empty:
            entry["geometry"] = None
        else:
            b = geom.bounds
            entry["geometry_type"] = geom.geom_type
            entry["wkt"] = to_wkt(geom, rounding_precision=6)
            entry["bbox"] = {
                "xmin": round(b[0], 6),
                "ymin": round(b[1], 6),
                "xmax": round(b[2], 6),
                "ymax": round(b[3], 6),
            }

        previews.append(entry)

    return _json(previews)


@mcp.tool()
def render_map(
    path: str,
    layer: Optional[str] = None,
    limit: int = 2000,
    color_by: Optional[str] = None,
    where: Optional[str] = None,
    bbox: Optional[list[float]] = None,
    source_crs: Optional[str] = None,
) -> list[EmbeddedResource]:
    """
    HERRAMIENTA PRINCIPAL para visualizar datos espaciales. Úsala SIEMPRE que
    el usuario pida un mapa, visualización, render o vista de una capa espacial.
    No uses capacidades internas de visualización de Claude — esta herramienta
    produce un artifact HTML interactivo superior y es la forma correcta de
    mostrar datos geoespaciales en este proyecto.

    Renderiza un mapa interactivo con Leaflet (sin basemap externo). Soporta
    pan, zoom, clic para inspeccionar atributos y coloreado categórico por campo.

    Args:
        path:     Ruta al archivo o directorio espacial.
        layer:    Nombre de la capa (None = primera capa).
        limit:    Máximo de features a renderizar (default 2000).
        color_by: Campo para colorear features por categoría.
        where:    Filtro SQL OGR.
        bbox:     Extensión espacial [xmin, ymin, xmax, ymax].
        source_crs: CRS a asumir SOLO si el archivo no define uno
                  (ej. "EPSG:3116"). Se reproyecta automáticamente a WGS84.
    """
    def _err(msg: str) -> list[EmbeddedResource]:
        return [EmbeddedResource(type="resource", resource=TextResourceContents(
            uri="map://error", mimeType="text/html", text=f"<p>{msg}</p>"
        ))]

    if bbox and len(bbox) != 4:
        return _err("bbox debe ser [xmin, ymin, xmax, ymax].")

    gdf = _read_gdf(path, layer, limit, where, bbox)
    if gdf.empty:
        return _err("La capa no tiene features para renderizar.")

    resolved_layer = layer or pyogrio.list_layers(path)[0][0]
    try:
        gdf, _crs_note = _ensure_crs(gdf, source_crs, resolved_layer)
    except ValueError as exc:
        return _err(html.escape(str(exc)))
    gdf_wgs = gdf.to_crs(epsg=4326)
    for col in gdf_wgs.columns:
        if pd.api.types.is_datetime64_any_dtype(gdf_wgs[col]):
            gdf_wgs[col] = gdf_wgs[col].astype(str)

    # Campo inexistente → ignorar para no pintar todo gris
    if color_by and color_by not in gdf_wgs.columns:
        color_by = None

    geojson_str, size_warning = _geojson_for_render(gdf_wgs)
    geojson_str = _js_embed(geojson_str)
    feature_count = len(gdf_wgs)

    color_map: dict = {}
    if color_by:
        cats = [str(v) for v in gdf_wgs[color_by].dropna().unique()]
        color_map = {c: _PALETTE[i % len(_PALETTE)] for i, c in enumerate(cats)}

    color_map_js = _js_embed(json.dumps(color_map))
    color_by_js  = _js_embed(json.dumps(color_by))

    warning_suffix = f" ⚠ {size_warning}" if size_warning else ""
    title_text = html.escape(
        f"{resolved_layer} — {feature_count:,} features{warning_suffix} | GeoAnalisis MCP"
    )

    legend_js = ""
    if color_map:
        items_js = _js_embed(json.dumps(
            [{"label": cat, "color": col} for cat, col in color_map.items()]
        ))
        legend_js = f"""
const legend = L.control({{position: 'bottomright'}});
legend.onAdd = () => {{
  const d = L.DomUtil.create('div', 'ga-legend');
  const items = {items_js};
  d.innerHTML = '<div class="ga-lt">' + esc({color_by_js}) + '</div>' +
    items.map(i => `<div class="ga-li"><span class="ga-sw" style="background:${{esc(i.color)}}"></span><span class="ga-lb">${{esc(i.label)}}</span></div>`).join('');
  return d;
}};
legend.addTo(map);"""

    html_doc = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(resolved_layer)} — GeoAnalisis MCP</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.css"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
html,body{{height:100%;overflow:hidden;font-family:sans-serif}}
#ga-hdr{{padding:6px 16px;background:rgba(255,255,255,.97);font:bold 13px sans-serif;
  text-align:center;box-shadow:0 1px 5px rgba(0,0,0,.18);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#222}}
#map{{height:calc(100% - 31px);background:#e8e5e0}}
.leaflet-container{{background:#e8e5e0}}
.ga-legend{{background:rgba(255,255,255,.93);padding:9px 11px;border-radius:7px;
  box-shadow:0 1px 5px rgba(0,0,0,.2);font-size:11px;
  max-height:180px;overflow-y:auto;min-width:110px}}
.ga-lt{{font-weight:700;margin-bottom:5px;color:#333}}
.ga-li{{display:flex;align-items:center;gap:5px;margin:2px 0}}
.ga-sw{{width:12px;height:12px;border-radius:2px;flex-shrink:0}}
.ga-lb{{color:#444;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:130px}}
</style>
</head>
<body>
<div id="ga-hdr">{title_text}</div>
<div id="map"></div>
<script>
const GJ        = {geojson_str};
const COLOR_MAP = {color_map_js};
const COLOR_BY  = {color_by_js};
const DEF_CLR   = "#2980b9";

const esc = s => String(s).replace(/[&<>"'`]/g,
  c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;','`':'&#96;'}}[c]));

const map = L.map('map');

function clr(props) {{
  if (!COLOR_BY || !props) return DEF_CLR;
  return COLOR_MAP[String(props[COLOR_BY] ?? '')] ?? '#aaaaaa';
}}

const gjLayer = L.geoJSON(GJ, {{
  style: f => ({{
    color:       clr(f.properties),
    fillColor:   clr(f.properties),
    fillOpacity: 0.65,
    weight:      1.2,
    opacity:     0.9,
  }}),
  pointToLayer: (f, ll) => L.circleMarker(ll, {{
    radius:      6,
    fillColor:   clr(f.properties),
    color:       '#fff',
    weight:      1.5,
    fillOpacity: 0.9,
  }}),
  onEachFeature: (f, l) => {{
    const p = f.properties || {{}};
    const rows = Object.entries(p)
      .filter(([, v]) => v != null && v !== '')
      .slice(0, 12)
      .map(([k, v]) => `<tr><td style="font-weight:600;padding-right:8px;white-space:nowrap;color:#555">${{esc(k)}}</td><td>${{esc(v)}}</td></tr>`)
      .join('');
    if (rows) l.bindPopup(`<table style="font-size:12px;border-collapse:collapse;line-height:1.5">${{rows}}</table>`);
  }},
}}).addTo(map);

map.fitBounds(gjLayer.getBounds(), {{padding: [20, 20]}});
{legend_js}
</script>
</body>
</html>"""

    return [EmbeddedResource(
        type="resource",
        resource=TextResourceContents(
            uri=f"map://{resolved_layer}",
            mimeType="text/html",
            text=html_doc,
        )
    )]


# ---------------------------------------------------------------------------
# Helpers de exportación
# ---------------------------------------------------------------------------

def _graduated_colors(
    gdf: gpd.GeoDataFrame,
    field: str,
    ramp: Optional[str],
    breaks_param,
):
    """Colores graduados para campo numérico. Returns (plot_colors, legend_patches, breaks)."""
    import matplotlib
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    values = gdf[field].dropna()
    if values.empty:
        return "#2980b9", [], None

    vmin, vmax = float(values.min()), float(values.max())
    if isinstance(breaks_param, list):
        # BoundaryNorm exige cortes estrictamente crecientes
        breaks = sorted({float(b) for b in breaks_param})
    else:
        n = max(2, int(breaks_param)) if breaks_param else 5
        if vmax <= vmin:
            vmax = vmin + 1.0  # campo constante: evitar breaks degenerados
        breaks = list(np.linspace(vmin, vmax, n + 1))

    if len(breaks) < 2:
        return "#2980b9", [], None

    try:
        cmap = matplotlib.colormaps[ramp or "RdYlGn_r"]
    except KeyError:
        cmap = matplotlib.colormaps["RdYlGn_r"]
    norm = mcolors.BoundaryNorm(breaks, cmap.N)

    plot_colors = gdf[field].apply(
        lambda v: tuple(cmap(norm(float(v)))) if pd.notna(v) else (0.667, 0.667, 0.667, 1.0)
    )
    patches = [
        mpatches.Patch(
            color=mcolors.to_hex(cmap(norm((breaks[i] + breaks[i + 1]) / 2))),
            label=f"{breaks[i]:.0f} – {breaks[i + 1]:.0f}",
        )
        for i in range(len(breaks) - 1)
    ]
    return plot_colors, patches, breaks


def _resolve_colors(gdf: gpd.GeoDataFrame, color_by, style, fallback_label: str):
    """Resuelve colores y parches de leyenda según color_by / style.

    Returns:
        (plot_colors, legend_patches, effective_field, style_type)
    """
    import matplotlib.patches as mpatches

    effective_field = color_by
    custom_categories: dict = {}
    style_type = "single"
    if style:
        style_type = style.get("type", "single")
        if style_type in ("categorized", "graduated"):
            effective_field = style.get("field", color_by)
        if style_type == "categorized":
            custom_categories = style.get("categories") or {}

    if style_type == "graduated" and effective_field and effective_field in gdf.columns:
        plot_colors, legend_patches, _ = _graduated_colors(
            gdf, effective_field,
            style.get("ramp") if style else None,
            style.get("breaks", 5) if style else 5,
        )
        return plot_colors, legend_patches, effective_field, style_type

    if effective_field and effective_field in gdf.columns:
        cats = [str(v) for v in gdf[effective_field].dropna().unique()]
        if custom_categories:
            color_map = {
                c: custom_categories.get(c, _PALETTE[i % len(_PALETTE)])
                for i, c in enumerate(cats)
            }
        else:
            color_map = {c: _PALETTE[i % len(_PALETTE)] for i, c in enumerate(cats)}
        plot_colors = gdf[effective_field].apply(
            lambda v: _to_rgba(color_map.get(str(v) if pd.notna(v) else "", "#aaaaaa"))
        )
        legend_patches = [mpatches.Patch(color=col, label=cat) for cat, col in color_map.items()]
        return plot_colors, legend_patches, effective_field, style_type

    plot_colors = "#2980b9"
    legend_patches = [mpatches.Patch(color="#2980b9", label=fallback_label)]
    return plot_colors, legend_patches, None, style_type


def _sort_legend_patches(patches: list, style_type: str) -> None:
    """Ordena parches de leyenda in-place por el número inicial de la etiqueta
    ("5 min" < "10 min"), con fallback alfabético. Los graduados no se tocan:
    ya vienen en orden de breaks y "10 – 20" se desordenaría.
    """
    import re

    if style_type == "graduated":
        return

    def key(p):
        lbl = str(p.get_label())
        m = re.match(r"\s*(-?\d+(?:[.,]\d+)?)", lbl)
        if m:
            return (0, float(m.group(1).replace(",", ".")), lbl)
        return (1, 0.0, lbl)

    try:
        patches.sort(key=key)
    except Exception:
        pass


def _symbology_report(
    resolved_layer: str,
    effective_field,
    style_type: str,
    n_classes: int,
    crs_note: str,
    extra_notes: list[str],
    disk_path,
) -> TextContent:
    """Reporte textual de la simbología REALMENTE aplicada por el servidor.

    Se devuelve junto a la imagen para que el cliente pueda verificar qué
    estilo recibió cada capa en vez de asumir que sus parámetros llegaron.
    """
    lines = ["Simbología aplicada por el servidor:"]
    if effective_field:
        lines.append(f"· capa principal '{resolved_layer}': {style_type} "
                     f"por campo '{effective_field}' ({n_classes} clases), {crs_note}")
    else:
        lines.append(f"· capa principal '{resolved_layer}': color único #2980b9, {crs_note}")
    lines.extend(f"· {n}" for n in extra_notes)
    lines.append("Todas las capas se renderizan reproyectadas a Web Mercator EPSG:3857.")
    if disk_path:
        lines.append(f"Guardado en: {disk_path}")
    return TextContent(type="text", text="\n".join(lines))


_BASEMAP_DEFAULT_ATTR = "© OpenStreetMap contributors © CARTO"


def _resolve_basemap(basemap: Optional[str]) -> tuple:
    """Resuelve el parámetro `basemap` a (source, attribution, nota, max_zoom).

    source None → usar el default CartoDB Positron. Acepta plantillas XYZ con
    tokens {z}/{x}/{y} tal cual, o la raíz de un ArcGIS MapServer con cache de
    tiles en Web Mercator (se valida vía ?f=json y se usa /tile/{z}/{y}/{x});
    max_zoom None → decidir según el proveedor.
    """
    if not basemap:
        return None, _BASEMAP_DEFAULT_ATTR, None, None

    url = basemap.strip().rstrip("/")
    if "{z}" in url:
        return url, None, f"basemap personalizado (plantilla XYZ): {url}", None

    if "/mapserver" not in url.lower():
        raise ValueError(
            "basemap debe ser una plantilla XYZ con tokens {z}/{x}/{y} o la raíz "
            "de un servicio ArcGIS MapServer con cache de tiles "
            "(ej. https://host/arcgis/rest/services/Nombre/MapServer)."
        )

    import requests
    try:
        meta = requests.get(url, params={"f": "json"}, timeout=15).json()
    except Exception as exc:
        raise ValueError(f"No se pudo leer el servicio ArcGIS '{url}': {exc}") from exc
    if meta.get("error"):
        raise ValueError(
            f"El servicio ArcGIS '{url}' respondió error: "
            f"{meta['error'].get('message', meta['error'])}"
        )
    if not meta.get("singleFusedMapCache"):
        raise ValueError(
            f"El servicio '{url}' no tiene cache de tiles (singleFusedMapCache=false) "
            "y no puede usarse como basemap. Usa un MapServer tileado o una "
            "plantilla XYZ con tokens {z}/{x}/{y}."
        )
    tile_info = meta.get("tileInfo") or {}
    sr = tile_info.get("spatialReference") or {}
    if not {sr.get("wkid"), sr.get("latestWkid")} & {102100, 3857}:
        raise ValueError(
            f"El cache de tiles de '{url}' no está en Web Mercator "
            f"(wkid={sr.get('wkid')}); solo se soportan caches EPSG:3857/102100."
        )
    lods = tile_info.get("lods") or []
    levels = f"niveles {lods[0]['level']}–{lods[-1]['level']}" if lods else "niveles ?"
    max_zoom = int(lods[-1]["level"]) if lods else None
    attribution = (meta.get("copyrightText") or "").strip() or None
    name = meta.get("mapName") or url.rsplit("/rest/services/", 1)[-1]
    note = f"basemap: ArcGIS tile cache '{name}' ({levels}) — {url}"
    return url + "/tile/{z}/{y}/{x}", attribution, note, max_zoom


_WEB_MERCATOR_RES0 = 156543.03392804097  # m/px de un tile 256px a zoom 0
_BASEMAP_MAX_TILES = 240                 # tope de descarga por render (~90 s peor caso)


def _basemap_zoom(ax, ax_width_in: float, dpi: int, max_zoom: int) -> int:
    """Zoom cuyo píxel de tile ≈ píxel de salida a `dpi` (basemap nítido).

    Acota por `max_zoom` del proveedor y por _BASEMAP_MAX_TILES para no
    disparar la descarga en extensiones grandes.
    """
    import math
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx = max(x1 - x0, 1.0)
    dy = max(y1 - y0, 1.0)
    target_px = max(ax_width_in, 1.0) * dpi
    z = math.ceil(math.log2(max(target_px, 256.0) * _WEB_MERCATOR_RES0 / dx))
    z = max(1, min(z, max_zoom))
    while z > 1:
        tile_m = 256.0 * _WEB_MERCATOR_RES0 / (2 ** z)
        if (dx / tile_m) * (dy / tile_m) <= _BASEMAP_MAX_TILES:
            break
        z -= 1
    return z


def _add_basemap(ax, crs, source, attribution, requested: Optional[str],
                 dpi: int = 150, ax_width_in: float = 0.0,
                 max_zoom: Optional[int] = None) -> Optional[str]:
    """add_basemap con fallback a CartoDB Positron si el basemap pedido falla.

    Con dpi > 150 pide los tiles al zoom que iguala el píxel de salida
    (en vez del "auto" de contextily, pensado para pantalla) y lo reporta.
    """
    import contextily as cx
    if source is None:
        source = cx.providers.CartoDB.Positron
    if max_zoom is None:
        max_zoom = source.get("max_zoom", 19) if isinstance(source, dict) else 19
    zoom = "auto"
    zoom_note = None
    if dpi > 150 and ax_width_in > 0:
        zoom = _basemap_zoom(ax, ax_width_in, dpi, max_zoom)
        zoom_note = (f"basemap: tiles a zoom {zoom} (máx. proveedor {max_zoom}) "
                     f"para dpi={dpi}")
    try:
        cx.add_basemap(ax, crs=crs, source=source, attribution=attribution, zoom=zoom)
        return zoom_note
    except Exception as exc:
        if not requested:
            raise
        cx.add_basemap(ax, crs=crs, source=cx.providers.CartoDB.Positron,
                       attribution=_BASEMAP_DEFAULT_ATTR)
        return (f"⚠ basemap '{requested}' falló al descargar tiles "
                f"({exc.__class__.__name__}: {exc}); se usó CartoDB Positron")


_EXTRA_LAYER_KEYS = {"path", "layer", "limit", "color", "alpha", "linewidth",
                     "linestyle", "edgecolor", "markersize", "label", "zorder", "crs"}


def _plot_extra_layers(ax, extra_layers) -> tuple[list[gpd.GeoDataFrame], list, list[str]]:
    """Dibuja capas extra (bajo la principal).

    Defaults según geometría: las líneas usan linewidth 1.5 y alpha 0.9 para que
    el color pedido se vea tal cual (0.5/0.7 convertía negro en gris tenue).

    Returns:
        (plotted, handles, notes): capas ya en EPSG:3857; handles de leyenda por
        capa con colores RGBA horneados (nunca alpha de artista: sobrescribiría
        el alpha del color y un fill "none" se volvería negro); y descripción
        textual del estilo realmente aplicado, para el reporte de simbología.
    """
    import os
    import sys

    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    def _bake(c, a: float):
        """Color → RGBA con alpha multiplicado. 'none' se queda transparente."""
        rgba = _to_rgba(c)
        return (rgba[0], rgba[1], rgba[2], rgba[3] * a)

    plotted: list[gpd.GeoDataFrame] = []
    handles: list = []
    notes: list[str] = []
    for extra in extra_layers or []:
        label = (extra.get("label") or extra.get("layer") or
                 os.path.splitext(os.path.basename(str(extra.get("path", "?"))))[0])
        try:
            e_gdf = _read_gdf(
                extra["path"], extra.get("layer"),
                extra.get("limit", 10000), None, None,
            )
            if e_gdf.empty:
                notes.append(f"capa extra '{label}': vacía — omitida")
                continue

            e_gdf, crs_note = _ensure_crs(e_gdf, extra.get("crs"), label)
            e_3857 = e_gdf.to_crs(epsg=3857)
            geom_types = e_3857.geom_type.dropna()
            main_type = geom_types.mode()[0] if not geom_types.empty else ""
            is_line = "Line" in main_type or "Ring" in main_type
            is_point = "Point" in main_type

            color = extra.get("color", "#888888")
            alpha = float(extra.get("alpha", 0.9 if is_line else 0.7))
            linewidth = float(extra.get("linewidth", 1.5 if is_line else 0.5))
            linestyle = extra.get("linestyle", "-")
            edgecolor = extra.get("edgecolor", "none")
            markersize = extra.get("markersize", 4)

            plot_kwargs = dict(
                ax=ax, color=color, alpha=alpha,
                linewidth=linewidth, linestyle=linestyle,
                edgecolor=edgecolor, markersize=markersize,
            )
            if "zorder" in extra:
                plot_kwargs["zorder"] = extra["zorder"]
            e_3857.plot(**plot_kwargs)
            plotted.append(e_3857)

            legend_alpha = min(1.0, alpha * 1.3)
            if is_line:
                handles.append(Line2D(
                    [], [], color=_bake(color, legend_alpha),
                    linewidth=max(linewidth, 1.8), linestyle=linestyle,
                    label=label,
                ))
            elif is_point:
                handles.append(Line2D(
                    [], [], linestyle="none", marker="o",
                    markerfacecolor=_bake(color, legend_alpha),
                    markeredgecolor="none", markersize=7, label=label,
                ))
            else:
                handles.append(mpatches.Patch(
                    facecolor=_bake(color, legend_alpha),
                    edgecolor=_bake(edgecolor, 1.0),
                    linestyle=linestyle, linewidth=max(linewidth, 1.0),
                    label=label,
                ))

            note = (f"capa extra '{label}' ({main_type or 'sin geometría'}): "
                    f"color={color}"
                    + (" ← DEFAULT gris, no se recibió 'color'" if "color" not in extra else "")
                    + f", edgecolor={edgecolor}, linewidth={linewidth}, "
                      f"linestyle={linestyle}, alpha={alpha}, {crs_note}")
            ignored = sorted(set(extra) - _EXTRA_LAYER_KEYS)
            if ignored:
                note += f" — CLAVES NO RECONOCIDAS (ignoradas): {ignored}"
            notes.append(note)
        except Exception as exc:
            notes.append(f"capa extra '{label}': ERROR, omitida ({exc})")
            print(f"[GeoAnalisis MCP] extra_layer ignorada: {exc}", file=sys.stderr)
    return plotted, handles, notes


def _save_figure(
    fig,
    output_path: Optional[str],
    source_path: str,
    resolved_layer: str,
    default_suffix: str,
    dpi: int,
    jpeg_quality: int = 90,
) -> Optional[str]:
    """Guarda la figura a disco según la extensión. Devuelve la ruta o None si falló.

    Un disco de solo lectura no debe abortar el tool: la imagen inline aún se genera.
    """
    import os
    import sys

    try:
        if output_path:
            disk_path = os.path.abspath(output_path)
            os.makedirs(os.path.dirname(disk_path) or ".", exist_ok=True)
        else:
            src_dir = os.path.dirname(os.path.abspath(source_path))
            safe_name = resolved_layer.replace(" ", "_").replace("/", "-")
            disk_path = os.path.join(src_dir, f"{safe_name}_{default_suffix}.jpg")

        ext = os.path.splitext(disk_path)[1].lower()
        if ext == ".png":
            fig.savefig(disk_path, format="png", dpi=dpi, bbox_inches="tight")
        elif ext == ".pdf":
            fig.savefig(disk_path, format="pdf", dpi=dpi, bbox_inches="tight")
        elif ext == ".svg":
            fig.savefig(disk_path, format="svg", bbox_inches="tight")
        else:
            fig.savefig(disk_path, format="jpeg", dpi=dpi, bbox_inches="tight",
                        pil_kwargs={"quality": jpeg_quality})
        return disk_path
    except Exception as exc:
        print(f"[GeoAnalisis MCP] no se pudo guardar en disco: {exc}", file=sys.stderr)
        return None


def _inline_image(fig) -> list[ImageContent]:
    """Serializa la figura a JPEG 72 DPI en base64 para mostrar inline en el chat."""
    import base64
    import io

    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="jpeg", dpi=72, bbox_inches="tight", pil_kwargs={"quality": 70})
    plt.close(fig)
    buf.seek(0)
    return [ImageContent(type="image", data=base64.b64encode(buf.read()).decode(),
                         mimeType="image/jpeg")]


def _mercator_lat_correction(y_center_m: float) -> float:
    """Factor cos(lat) para convertir metros Web Mercator a metros reales.

    EPSG:3857 infla las distancias por 1/cos(lat); sin esta corrección la
    barra de escala miente fuera del ecuador (~40 % de error a 45°).
    """
    import math
    lat = math.atan(math.sinh(y_center_m / 6378137.0))
    return math.cos(lat)


def _add_scalebar(ax) -> None:
    """Barra de escala en esquina inferior izquierda (ejes en metros, EPSG:3857)."""
    import matplotlib.patches as mpatches

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    map_width_m = abs(xlim[1] - xlim[0]) * _mercator_lat_correction((ylim[0] + ylim[1]) / 2)
    if map_width_m == 0:
        return

    target_m = map_width_m * 0.18
    nice_vals = [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000,
                 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
    bar_m = min(nice_vals, key=lambda v: abs(v - target_m))
    bar_frac = bar_m / map_width_m

    x0, y0, h = 0.05, 0.055, 0.013
    for i, fc in enumerate(["#111111", "#ffffff"]):
        ax.add_patch(mpatches.Rectangle(
            (x0 + i * bar_frac / 2, y0), bar_frac / 2, h,
            transform=ax.transAxes,
            facecolor=fc, edgecolor="#111111", linewidth=0.6, zorder=10,
        ))
    label = f"{bar_m} m" if bar_m < 1000 else f"{bar_m // 1000} km"
    ax.text(x0, y0 + h + 0.005, "0",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7, color="#222", zorder=11)
    ax.text(x0 + bar_frac / 2, y0 + h + 0.005, label,
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7.5, color="#222", zorder=11)


def _draw_north_svg(ax, cx: float, cy: float, size: float, zorder: int = 16) -> None:
    """
    Dibuja el ícono SVG de norte (flecha cartográfica) en axes fraction coords.

    Vértices pre-computados desde:
      M47.655 1.634 l-35 95 c-.828 2.24 1.659 4.255 3.68 2.98
      l33.667-21.228 l33.666 21.228 c2.02 1.271 4.503-.74 3.678-2.98
      l-35-95 C51.907.514 51.163.006 50 .008 c-1.163.001-1.99.65-2.345 1.626 z
      m-.155 14.88 v57.54 L19.89 91.461 z
    viewBox 0 0 100 100, Y-down. fill-rule: evenodd simulado con patch blanco.

    Args:
        cx, cy : centro en axes fraction.
        size   : altura del ícono en axes fraction (ancho corregido por aspect ratio).
        zorder : z-order base.
    """
    from matplotlib.path import Path
    import matplotlib.patches as mpatches

    # Corregir distorsión: el ícono debe verse cuadrado en pantalla, no aplastado
    fw, fh = ax.figure.get_size_inches()
    ar = fw / fh  # width/height de la figura → ancho_px / alto_px del ícono

    def t(sx: float, sy: float) -> tuple:
        """SVG [0-100, Y↓] → axes fraction centrado en (cx, cy)."""
        return (
            cx + (sx / 100.0 - 0.5) * size / ar,
            cy + (0.5 - sy / 100.0) * size,
        )

    # ── Subpath 1: forma exterior de la flecha ────────────────────────────────
    verts1 = [
        t(47.655,   1.634),                                                  # M
        t(12.655,  96.634),                                                  # l-35 95
        t(11.827,  98.874), t(14.314, 100.889), t(16.335,  99.614),         # c bezier 1
        t(50.002,  78.386),                                                  # l33.667-21.228
        t(83.668,  99.614),                                                  # l33.666 21.228
        t(85.688, 100.885), t(88.171,  98.874), t(87.346,  96.634),         # c bezier 2
        t(52.346,   1.634),                                                  # l-35-95
        t(51.907,   0.514), t(51.163,   0.006), t(50.000,   0.008),         # C bezier 3
        t(48.837,   0.009), t(48.010,   0.658), t(47.655,   1.634),         # c bezier 4
        t(47.655,   1.634),                                                  # CLOSEPOLY
    ]
    codes1 = [
        Path.MOVETO,
        Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO,
        Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    ax.add_patch(mpatches.PathPatch(
        Path(verts1, codes1),
        facecolor="#111111", edgecolor="none",
        transform=ax.transAxes, zorder=zorder,
    ))

    # ── Subpath 2: triángulo interior blanco (simula fill-rule evenodd) ───────
    verts2 = [
        t(47.500, 16.514),                      # m-.155 14.88
        t(47.500, 74.054),                      # v57.54
        t(19.890, 91.461),                      # L19.89 91.461
        t(19.890, 91.461),                      # CLOSEPOLY dummy
    ]
    codes2 = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(mpatches.PathPatch(
        Path(verts2, codes2),
        facecolor="white", edgecolor="none",
        transform=ax.transAxes, zorder=zorder + 1,
    ))



def _place_labels(ax, gdf: gpd.GeoDataFrame, field: str, fontsize: int = 8) -> None:
    """
    Coloca etiquetas de forma inteligente según el tipo de geometría.

    · Polígono : representative_point() (garantizado dentro del polígono).
                 Filtra los demasiado pequeños (< 1% del área total).
                 Ordena por área desc: los más grandes obtienen etiqueta primero.
    · Línea    : punto medio de la línea, texto rotado según el segmento.
    · Punto    : offset fijo de 5 px hacia arriba-derecha del símbolo.

    Detecta colisiones en coordenadas de pantalla y descarta las que solapan.
    """
    import math

    if field not in gdf.columns or gdf.empty:
        return

    geom_types = gdf.geom_type.dropna()
    if geom_types.empty:
        return
    first_type = geom_types.mode()[0]
    is_poly  = "Polygon" in first_type
    is_line  = "Line" in first_type or "Ring" in first_type

    min_area = gdf.geometry.area.sum() * 0.01 if is_poly else 0.0

    placed: list[tuple] = []  # bboxes (x0, y0, x1, y1) en display coords

    def _no_overlap(x_data: float, y_data: float, text: str) -> bool:
        """True si la etiqueta NO solapa con ninguna ya colocada (y la registra)."""
        try:
            disp = ax.transData.transform((x_data, y_data))
        except Exception:
            return False
        hw = len(text) * fontsize * 0.45 + 6
        hh = fontsize * 0.75 + 4
        box = (disp[0] - hw, disp[1] - hh, disp[0] + hw, disp[1] + hh)
        for pb in placed:
            if not (box[2] < pb[0] or box[0] > pb[2] or
                    box[3] < pb[1] or box[1] > pb[3]):
                return False  # solapa → descartar
        placed.append(box)
        return True

    bbox_style = dict(boxstyle="round,pad=0.25", facecolor="white",
                      alpha=0.82, edgecolor="none")

    gdf_work = (
        gdf.assign(_area=gdf.geometry.area).sort_values("_area", ascending=False)
        if is_poly else gdf
    )

    for _, row in gdf_work.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        val = row[field]
        if pd.isna(val):
            continue
        text = str(val)

        if is_poly:
            if geom.area < min_area:
                continue
            pt = geom.representative_point()
            x, y = pt.x, pt.y
            if not _no_overlap(x, y, text):
                continue
            ax.annotate(
                text, xy=(x, y),
                ha="center", va="center",
                fontsize=fontsize, color="#111", fontweight="bold",
                bbox=bbox_style, zorder=20,
            )

        elif is_line:
            mp = geom.interpolate(0.5, normalized=True)
            p1 = geom.interpolate(0.45, normalized=True)
            p2 = geom.interpolate(0.55, normalized=True)
            dx, dy = p2.x - p1.x, p2.y - p1.y
            angle = math.degrees(math.atan2(dy, dx))
            if angle > 90:  angle -= 180
            if angle < -90: angle += 180
            x, y = mp.x, mp.y
            if not _no_overlap(x, y, text):
                continue
            ax.annotate(
                text, xy=(x, y),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", rotation=angle,
                fontsize=fontsize, color="#111", fontweight="bold",
                bbox=bbox_style, zorder=20,
            )

        else:  # Punto
            x, y = geom.x, geom.y
            if not _no_overlap(x, y, text):
                continue
            ax.annotate(
                text, xy=(x, y),
                xytext=(5, 5), textcoords="offset points",
                ha="left", va="bottom",
                fontsize=fontsize, color="#111", fontweight="bold",
                bbox=bbox_style, zorder=20,
            )


# ---------------------------------------------------------------------------
# Exportación de imagen
# ---------------------------------------------------------------------------

@mcp.tool()
def export_map_image(
    path: str,
    layer: Optional[str] = None,
    limit: int = 5000,
    color_by: Optional[str] = None,
    where: Optional[str] = None,
    bbox: Optional[list[float]] = None,
    style: Optional[dict] = None,
    output_path: Optional[str] = None,
    legend: bool = True,
    dpi: int = 150,
    figsize: Optional[list[float]] = None,
    scalebar: bool = True,
    north_arrow: bool = True,
    credits: Optional[str] = None,
    label_by: Optional[str] = None,
    extra_layers: Optional[list[dict]] = None,
    source_crs: Optional[str] = None,
    basemap: Optional[str] = None,
) -> list[TextContent | ImageContent]:
    """
    Genera una imagen del mapa con basemap CartoDB Positron mostrada INLINE en el chat
    y guardada en disco en alta resolución. Devuelve además un reporte textual de la
    simbología realmente aplicada a cada capa: VERIFICARLO antes de afirmar que un
    color/estilo fue cambiado.

    Args:
        path:          Ruta al archivo o directorio espacial.
        layer:         Nombre de la capa (None = primera capa).
        limit:         Máximo de features (default 5000).
        color_by:      Campo para colorear por categoría con colores automáticos.
                       Alternativa simple a `style`.
        where:         Filtro SQL OGR. Si se aplica, la extensión del mapa se ajusta
                       (zoom) a las features filtradas; las capas extra quedan como
                       contexto recortado sin ampliar la vista.
        bbox:          Extensión [xmin, ymin, xmax, ymax].
        style:         Estilo avanzado:
                         Categorizado: {"type": "categorized", "field": "Tipo",
                                        "categories": {"A": "#e74c3c", "B": "#2ecc71"}}
                         Graduado:     {"type": "graduated", "field": "Tiempo",
                                        "ramp": "RdYlGn_r", "breaks": 5}
                                       breaks = int (clases iguales) o lista de cortes explícitos.
                                       Rampas útiles: RdYlGn_r, YlOrRd, Blues, viridis, plasma.
                       "field" en style tiene precedencia sobre color_by.
        output_path:   Ruta completa de salida incluyendo extensión.
                       .png = PNG lossless · .jpg = JPEG · .pdf = PDF vectorial · .svg = SVG.
                       Si None guarda "{capa}_map.jpg" junto al archivo fuente.
        legend:        Mostrar leyenda (default True).
        dpi:           Resolución en DPI para disco (default 150). La imagen inline del chat
                       siempre usa 72 DPI para mantener el tamaño manejable.
                       Con dpi > 150 los tiles del basemap se piden a mayor zoom
                       para mantener la nitidez (descarga más tiles: más lento).
        figsize:       Tamaño [ancho, alto] en pulgadas, ej. [14, 8.5] para A4 horizontal.
                       Default [10, 6].
        scalebar:      Mostrar barra de escala en esquina inferior izquierda (default True).
        north_arrow:   Mostrar flecha de norte en esquina superior izquierda (default True).
        credits:       Texto de créditos en pie de mapa,
                       ej. "Fuente: INEGI 2024 | Elaborado con GeoAnalisis MCP".
        label_by:      Campo para etiquetar cada feature en su centroide.
        extra_layers:  Capas adicionales a superponer sobre el basemap, en orden ascendente
                       (primera entrada = capa más baja). Cada capa aparece en la leyenda.
                       IMPORTANTE: pasar SIEMPRE 'color' explícito y distinto por capa
                       (matiz diferente); el gris default es solo para puro contexto.
                       Cada entrada es un dict con:
                         path (str, requerido), layer (str), limit (int, default 10000),
                         color (str; ej. "#e74c3c". Default "#888" gris),
                         alpha (float; default 0.9 líneas, 0.7 resto),
                         linewidth (float; default 1.5 líneas, 0.5 resto),
                         linestyle (str, default "-"; "--" discontinua, ":" punteada),
                         edgecolor (str, default "none"), markersize (float, default 4),
                         label (str; nombre en leyenda, default nombre de capa/archivo),
                         zorder (int; opcional, para dibujar sobre la capa principal),
                         crs (str; ej. "EPSG:9377". Solo si el archivo NO define CRS:
                              se asume ese. Si define uno, se respeta el del archivo).
                       Cualquier otra clave se ignora y se reporta en el texto de salida.
        source_crs:    CRS a asumir para la capa principal SOLO si el archivo no define
                       uno (ej. "EPSG:3116"). Sin esto, una capa sin CRS con bounds
                       lon/lat se asume EPSG:4326; con bounds proyectados se rechaza
                       indicando cómo corregir. Todas las capas se reproyectan
                       automáticamente a Web Mercator para el render.
        basemap:       URL de un basemap alternativo al default CartoDB Positron.
                       Acepta la raíz de un servicio ArcGIS MapServer con cache de
                       tiles en Web Mercator (ej. "https://host/arcgis/rest/services/
                       Nombre/MapServer": se valida el cache y se usa la plantilla
                       /tile/{z}/{y}/{x}, con atribución del copyrightText) o una
                       plantilla XYZ con tokens, ej. "https://tile.host/{z}/{x}/{y}.png".
                       Si los tiles fallan al descargar se usa CartoDB Positron como
                       fallback y se reporta en el texto de salida.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    if bbox and len(bbox) != 4:
        raise ValueError("bbox debe ser [xmin, ymin, xmax, ymax].")

    bm_source, bm_attr, bm_note, bm_maxz = _resolve_basemap(basemap)

    gdf = _read_gdf(path, layer, limit, where, bbox)
    if gdf.empty:
        raise ValueError("La capa no tiene features para renderizar.")

    resolved_layer = layer or pyogrio.list_layers(path)[0][0]
    gdf, crs_note = _ensure_crs(gdf, source_crs, resolved_layer)
    gdf_3857 = gdf.to_crs(epsg=3857)

    plot_colors, legend_patches, effective_field, _style_type = _resolve_colors(
        gdf_3857, color_by, style, resolved_layer
    )

    # --- Figura: tamaño adaptado al aspect ratio de los datos ---
    if figsize and len(figsize) == 2:
        fw, fh = float(figsize[0]), float(figsize[1])
    else:
        b = gdf_3857.total_bounds  # [xmin, ymin, xmax, ymax]
        dx = max(float(b[2] - b[0]), 1.0)
        dy = max(float(b[3] - b[1]), 1.0)
        # Ratio acotado: con datos muy alargados (corredores) la figura no se
        # vuelve esbelta; el sobrante queda como aire lateral/vertical para
        # que leyenda, escala y atribución no se monten sobre los datos.
        ratio = min(max(dy / dx, 0.45), 1.6)
        fw = 10.0 / max(ratio, 1.0) if ratio >= 1.0 else 10.0
        fh = fw * ratio
        fw = max(5.0, min(fw, 14.0))
        fh = max(5.0, min(fh, 12.0))
    fig, ax = plt.subplots(figsize=(fw, fh))
    fig.patch.set_facecolor("#f5f4f1")
    ax.set_facecolor("#e8e5e0")

    # --- Capas adicionales (debajo de la principal) ---
    extra_gdfs, extra_handles, extra_notes = _plot_extra_layers(ax, extra_layers)

    # --- Capa principal ---
    gdf_3857.plot(
        ax=ax,
        color=plot_colors,
        edgecolor=(0, 0, 0, 0.25),
        linewidth=0.6,
        markersize=6,
        alpha=0.75,
    )

    # --- Extensión de la vista ---
    # Con filtro por atributo manda el subset filtrado (gdf ya viene filtrado);
    # sin filtro se incluyen las capas extra. La vista se expande luego al
    # aspect ratio del axes: en capas muy alargadas eso deja aire lateral o
    # vertical para que leyenda, escala y atribución no se monten sobre los datos.
    vb = gdf_3857.total_bounds.copy()
    if not where:
        for _e3857 in extra_gdfs:
            _eb = _e3857.total_bounds
            vb[0] = min(vb[0], _eb[0])
            vb[1] = min(vb[1], _eb[1])
            vb[2] = max(vb[2], _eb[2])
            vb[3] = max(vb[3], _eb[3])
    pad_x = max((vb[2] - vb[0]) * 0.08, 250.0)
    pad_y = max((vb[3] - vb[1]) * 0.08, 250.0)
    dx_v = (vb[2] - vb[0]) + 2 * pad_x
    dy_v = (vb[3] - vb[1]) + 2 * pad_y
    # original=True: la posición activa ya viene encogida por el aspecto igual
    # de los datos y daría un ratio equivocado.
    _pos = ax.get_position(original=True)
    ax_ar = (_pos.width * fw) / (_pos.height * fh)  # ancho/alto del axes
    if dx_v / dy_v < ax_ar:
        dx_v = dy_v * ax_ar
    else:
        dy_v = dx_v / ax_ar
    xc = (vb[0] + vb[2]) / 2
    yc = (vb[1] + vb[3]) / 2
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xc - dx_v / 2, xc + dx_v / 2)
    ax.set_ylim(yc - dy_v / 2, yc + dy_v / 2)

    # --- Basemap ---
    bm_extra = _add_basemap(ax, gdf_3857.crs, bm_source, bm_attr, basemap,
                            dpi=dpi, ax_width_in=_pos.width * fw, max_zoom=bm_maxz)
    extra_notes.extend(n for n in (bm_note, bm_extra) if n)
    ax.set_axis_off()
    ax.add_patch(mpatches.Rectangle(
        (0, 0), 1, 1,
        transform=ax.transAxes,
        fill=False, edgecolor="#888888", linewidth=0.8, zorder=30,
    ))

    # --- Barra de escala ---
    if scalebar:
        _add_scalebar(ax)

    # --- Símbolo de norte (SVG path, sin fondo) ---
    if north_arrow:
        _draw_north_svg(ax, cx=0.075, cy=0.910, size=0.07, zorder=16)

    # --- Leyenda (capa principal ordenada + capas extra) ---
    _sort_legend_patches(legend_patches, _style_type)
    all_handles = (legend_patches or []) + extra_handles
    if legend and all_handles:
        ax.legend(
            handles=all_handles,
            title=effective_field,
            title_fontsize=10,
            fontsize=9,
            loc="lower right",
            framealpha=0.93,
            edgecolor="#cccccc",
        )

    # --- Créditos en pie (anclados al eje: el bbox tight recorta el sobrante) ---
    if credits:
        ax.text(0.5, -0.018, credits,
                transform=ax.transAxes, ha="center", va="top",
                fontsize=7, color="#555", style="italic")

    ax.set_title(
        f"{resolved_layer}  |  GeoAnalisis MCP",
        fontsize=12, fontweight="bold", color="#222",
        pad=8,
    )
    plt.tight_layout()

    # --- Etiquetas por campo (tras el layout final: la detección de colisiones
    #     usa coordenadas de pantalla, que cambian con tight_layout) ---
    if label_by and label_by in gdf_3857.columns:
        _place_labels(ax, gdf_3857, label_by, fontsize=8)

    # --- Guardar en disco (no aborta si falla) ---
    disk_path = _save_figure(fig, output_path, path, resolved_layer, "map", dpi, jpeg_quality=90)

    # --- Reporte de simbología + imagen inline (72 DPI, JPEG → ~30-80 KB) ---
    report = _symbology_report(
        resolved_layer, effective_field, _style_type,
        len(legend_patches), crs_note, extra_notes, disk_path,
    )
    return [report] + _inline_image(fig)


# ---------------------------------------------------------------------------
# Layout cartográfico formal
# ---------------------------------------------------------------------------

@mcp.tool()
def export_map_cartographic(
    path: str,
    layer: Optional[str] = None,
    limit: int = 5000,
    color_by: Optional[str] = None,
    where: Optional[str] = None,
    bbox: Optional[list[float]] = None,
    style: Optional[dict] = None,
    output_path: Optional[str] = None,
    dpi: int = 150,
    figsize: Optional[list[float]] = None,
    title: Optional[str] = None,
    credits: Optional[str] = None,
    label_by: Optional[str] = None,
    extra_layers: Optional[list[dict]] = None,
    source_crs: Optional[str] = None,
    basemap: Optional[str] = None,
) -> list[TextContent | ImageContent]:
    """
    Genera un mapa cartográfico técnico con layout formal completo:
    título institucional · panel lateral (leyenda, convenciones, índice de
    localización) · franja inferior (norte, escala, parámetros/fuente).
    Devuelve además un reporte textual de la simbología realmente aplicada
    a cada capa: VERIFICARLO antes de afirmar que un color/estilo cambió.

    Usar para productos formales ("mapa cartográfico", "carta", "entrega",
    "producto"). Para exploración rápida usar export_map_image.

    Args:
        path:        Ruta al archivo o directorio espacial.
        layer:       Nombre de la capa (None = primera capa).
        limit:       Máximo de features (default 5000).
        color_by:    Campo para colorear por categoría (colores automáticos).
        where:       Filtro SQL OGR. Si se aplica, la extensión del mapa se ajusta
                     (zoom) a las features filtradas; las capas extra quedan como
                     contexto recortado sin ampliar la vista.
        bbox:        Extensión [xmin, ymin, xmax, ymax] en el CRS de la capa.
        style:       Estilo avanzado (mismo formato que export_map_image):
                       Categorizado: {"type":"categorized","field":"X",
                                      "categories":{"A":"#e74c3c"}}
                       Graduado:     {"type":"graduated","field":"X",
                                      "ramp":"RdYlGn_r","breaks":5}
        output_path: Ruta de salida. .jpg para trabajo, .pdf para entrega.
                     Default: "{capa}_cartographic.jpg" junto al archivo fuente.
        dpi:         Resolución (default 150). Con dpi > 150 los tiles del
                     basemap se piden a mayor zoom para mantener la nitidez
                     (descarga más tiles: más lento).
        figsize:     [ancho, alto] en pulgadas. Default auto-calculado.
        title:       Título del mapa. Default: nombre de la capa.
        credits:     Texto de fuente/créditos para el panel inferior.
                     Ej: "Fuente: IGAC 2024 | Sistema: WGS84 EPSG:4326"
        label_by:    Campo para etiquetar features (colocación inteligente).
        extra_layers: Capas adicionales (mismo formato que export_map_image,
                     incluida la clave 'crs' para capas sin CRS definido).
                     IMPORTANTE: pasar SIEMPRE 'color' explícito y distinto por
                     capa (matiz diferente) y 'label' descriptivo; el gris
                     default es solo para capas de puro contexto.
        source_crs:  CRS a asumir para la capa principal SOLO si el archivo no
                     define uno (ej. "EPSG:3116"). Ver export_map_image.
        basemap:     URL de un basemap alternativo al default CartoDB Positron:
                     raíz de un ArcGIS MapServer con cache de tiles en Web
                     Mercator o plantilla XYZ con tokens {z}/{x}/{y}.
                     Ver export_map_image.
    """
    import os
    import warnings
    from datetime import date

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.gridspec as mgridspec
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    if bbox and len(bbox) != 4:
        raise ValueError("bbox debe ser [xmin, ymin, xmax, ymax].")

    bm_source, bm_attr, bm_note, bm_maxz = _resolve_basemap(basemap)

    # ── Leer datos ────────────────────────────────────────────────────────────
    gdf = _read_gdf(path, layer, limit, where, bbox)
    if gdf.empty:
        raise ValueError("La capa no tiene features para renderizar.")

    resolved_layer = layer or pyogrio.list_layers(path)[0][0]
    gdf, crs_note = _ensure_crs(gdf, source_crs, resolved_layer)
    gdf_3857  = gdf.to_crs(epsg=3857)
    gdf_wgs84 = gdf.to_crs(epsg=4326)

    # ── Colores (misma lógica que export_map_image) ──────────────────────────
    plot_colors, legend_patches, effective_field, style_type = _resolve_colors(
        gdf_3857, color_by, style, resolved_layer
    )

    # ── Figsize auto-adaptado (el mapa ocupa ~75 % del ancho) ───────────────
    if figsize and len(figsize) == 2:
        fw, fh = float(figsize[0]), float(figsize[1])
    else:
        b  = gdf_3857.total_bounds
        dx = max(float(b[2] - b[0]), 1.0)
        dy = max(float(b[3] - b[1]), 1.0)
        ratio = dy / dx
        fw_map = 10.0 / max(ratio, 1.0) if ratio >= 1.0 else 10.0
        fh_map = fw_map * ratio
        fw_map = max(5.0, min(fw_map, 11.0))
        fh_map = max(5.0, min(fh_map, 11.0))
        # Panel lateral ~33 % extra de ancho; título + barra inferior ~15 % extra alto
        fw = max(16.0, fw_map * 1.38)
        fh = max(11.0, fh_map * 1.18)

    # ── Figura y GridSpec principal ──────────────────────────────────────────
    BG_PANEL = "#f5f4f0"
    BG_STRIP = "#f0ede8"

    fig = plt.figure(figsize=(fw, fh), facecolor="white")

    gs = mgridspec.GridSpec(
        3, 2,
        figure=fig,
        height_ratios=[0.45, 10.0, 0.80],   # título | cuerpo | barra inferior
        width_ratios=[3.0, 1.0],             # mapa   | panel
        hspace=0.0,
        wspace=0.0,
    )

    ax_title  = fig.add_subplot(gs[0, :])   # título (ancho completo)
    ax_map    = fig.add_subplot(gs[1, 0])   # mapa principal
    ax_panel  = fig.add_subplot(gs[1, 1])   # panel lateral
    ax_bottom = fig.add_subplot(gs[2, :])   # barra inferior (ancho completo)

    for ax in (ax_title, ax_panel, ax_bottom):
        ax.set_facecolor(BG_PANEL)
    ax_title.set_facecolor(BG_STRIP)
    ax_map.set_facecolor("#e8e5e0")
    for ax in (ax_title, ax_panel, ax_bottom):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # ── TÍTULO ───────────────────────────────────────────────────────────────
    title_text = title or resolved_layer
    ax_title.text(
        0.50, 0.55, title_text,
        transform=ax_title.transAxes,
        ha="center", va="center",
        fontsize=14, fontweight="bold", color="#111",
    )
    # Línea separadora inferior del título
    ax_title.plot([0, 1], [0, 0], transform=ax_title.transAxes,
                  color="#666", linewidth=1.5, solid_capstyle="round")

    # ── Capas extra (bajo la capa principal); se cachean para los bounds ─────
    extra_gdfs, extra_handles, extra_notes = _plot_extra_layers(ax_map, extra_layers)

    # ── Capa principal ────────────────────────────────────────────────────────
    gdf_3857.plot(
        ax=ax_map,
        color=plot_colors,
        edgecolor=(0, 0, 0, 0.22),
        linewidth=0.6,
        markersize=6,
        alpha=0.80,
    )

    # ── Ajustar extent del mapa para eliminar espacios en blanco ─────────────
    # Con filtro por atributo (where), la extensión de las features filtradas
    # tiene prioridad: las capas extra quedan como contexto recortado. Sin
    # filtro, se combinan los bounds de todas las capas dibujadas.
    _b = gdf_3857.total_bounds.copy()
    if not where:
        for _e3857 in extra_gdfs:
            _eb3 = _e3857.total_bounds
            _b[0] = min(_b[0], _eb3[0])
            _b[1] = min(_b[1], _eb3[1])
            _b[2] = max(_b[2], _eb3[2])
            _b[3] = max(_b[3], _eb3[3])

    _pad_x = max((_b[2] - _b[0]) * 0.08, 250.0)
    _pad_y = max((_b[3] - _b[1]) * 0.08, 250.0)
    _xc = (_b[0] + _b[2]) / 2
    _yc = (_b[1] + _b[3]) / 2
    _dx = (_b[2] - _b[0]) + 2 * _pad_x
    _dy = (_b[3] - _b[1]) + 2 * _pad_y

    # Fuerza el AR del extent = AR del axes del mapa para llenar sin bandas vacías
    # Estimación de AR del axes: mapa ocupa 3/4 del ancho y ~88.9% del alto de la figura
    _ax_ar = (fw * 0.75) / (fh * 0.889)
    if _dx / _dy < _ax_ar:
        _dx = _dy * _ax_ar   # datos portrait → ampliar ancho
    else:
        _dy = _dx / _ax_ar   # datos landscape → ampliar alto
    ax_map.set_xlim(_xc - _dx / 2, _xc + _dx / 2)
    ax_map.set_ylim(_yc - _dy / 2, _yc + _dy / 2)

    # ── Basemap (default CartoDB Positron; ver parámetro `basemap`) ──────────
    _map_pos = ax_map.get_position(original=True)
    bm_extra = _add_basemap(ax_map, gdf_3857.crs, bm_source, bm_attr, basemap,
                            dpi=dpi, ax_width_in=_map_pos.width * fw,
                            max_zoom=bm_maxz)
    extra_notes.extend(n for n in (bm_note, bm_extra) if n)
    ax_map.set_axis_off()

    # ── PANEL LATERAL ─────────────────────────────────────────────────────────
    _sort_legend_patches(legend_patches, style_type)

    # Separador vertical izquierdo del panel
    ax_panel.plot([0, 0], [0, 1], transform=ax_panel.transAxes,
                  color="#555", linewidth=1.5)

    # Proporciones fijas (desde abajo):
    #   Localización: 0.00 → y_div2  (42 %)
    #   Convenciones: y_div2 → y_div1 (14 %)
    #   Leyenda:      y_div1 → 1.00   (44 %)
    y_div2 = 0.42   # localización ↔ convenciones
    y_div1 = 0.56   # convenciones ↔ leyenda

    for y_d in (y_div1, y_div2):
        ax_panel.plot([0.02, 0.98], [y_d, y_d], transform=ax_panel.transAxes,
                      color="#888", linewidth=1.2)

    # ─ Sección 1: LEYENDA TEMÁTICA (y_div1 → 1.00) ───────────────────────────
    ax_panel.text(
        0.50, 0.975, "LEYENDA TEMÁTICA",
        transform=ax_panel.transAxes,
        ha="center", va="top",
        fontsize=9, fontweight="bold", color="#222",
    )
    if effective_field:
        ax_panel.text(
            0.50, 0.940, effective_field,
            transform=ax_panel.transAxes,
            ha="center", va="top",
            fontsize=7.5, color="#555", style="italic",
        )
        y_items_start = 0.902
    else:
        y_items_start = 0.938

    n_items = len(legend_patches)
    n_extra = len(extra_handles)
    # +1.5 fila por el separador/título de capas extra si las hay
    n_total_rows = n_items + n_extra + (1.5 if n_extra > 0 else 0)

    y_items_floor = y_div1 + 0.028
    available_h   = y_items_start - y_items_floor
    row_h = min(available_h / max(n_total_rows, 1), 0.075)

    # Font y swatch proporcionales al tamaño real del panel en pulgadas
    _panel_h_in = fh * (10.0 / (0.45 + 10.0 + 0.80))
    label_fs    = max(7.5, min(10.0, row_h * _panel_h_in * 72 * 0.22))
    swatch_h    = row_h * 0.40   # swatch = 40 % del alto de fila

    last_y_item = y_items_start

    for i, patch in enumerate(legend_patches):
        y_item      = y_items_start - i * row_h
        last_y_item = y_item
        if y_item < y_items_floor:
            ax_panel.text(
                0.50, y_item, f"(+{n_items - i} más…)",
                transform=ax_panel.transAxes,
                ha="center", va="top",
                fontsize=label_fs * 0.85, color="#888",
            )
            last_y_item = y_items_floor
            break
        fc  = patch.get_facecolor()
        lbl = str(patch.get_label())[:26]
        ax_panel.add_patch(mpatches.Rectangle(
            (0.05, y_item - swatch_h * 0.65), 0.16, swatch_h,
            facecolor=fc, edgecolor="#444", linewidth=0.5,
            transform=ax_panel.transAxes, zorder=5,
        ))
        ax_panel.text(
            0.26, y_item - swatch_h * 0.08, lbl,
            transform=ax_panel.transAxes,
            va="center", fontsize=label_fs, color="#111",
        )

    # ─ Capas adicionales en la leyenda (desde los handles reales: respetan
    #   geometría, edgecolor y linestyle — un fill "none" ya no sale negro) ────
    if extra_handles:
        from matplotlib.lines import Line2D

        y_sep = last_y_item - row_h * 0.55
        if y_sep > y_items_floor + row_h:
            ax_panel.plot([0.05, 0.95], [y_sep, y_sep],
                          transform=ax_panel.transAxes,
                          color="#ccc", linewidth=0.7, linestyle="dashed")
            ax_panel.text(
                0.50, y_sep - 0.004, "Capas adicionales",
                transform=ax_panel.transAxes,
                ha="center", va="top",
                fontsize=label_fs * 0.82, color="#777", style="italic",
            )
            for j, h in enumerate(extra_handles):
                y_e = y_sep - row_h * 0.70 - j * row_h
                if y_e < y_items_floor:
                    break
                y_mid = y_e - swatch_h * 0.25
                if isinstance(h, Line2D) and h.get_marker() not in ("", "None", None):
                    ax_panel.plot(
                        [0.13], [y_mid], marker="o", linestyle="none",
                        markersize=6, markerfacecolor=h.get_markerfacecolor(),
                        markeredgecolor="none",
                        transform=ax_panel.transAxes, zorder=5, clip_on=False,
                    )
                elif isinstance(h, Line2D):
                    ax_panel.plot(
                        [0.05, 0.21], [y_mid, y_mid],
                        color=h.get_color(), linewidth=h.get_linewidth(),
                        linestyle=h.get_linestyle(),
                        transform=ax_panel.transAxes, zorder=5,
                        solid_capstyle="butt", clip_on=False,
                    )
                else:
                    ec = h.get_edgecolor()
                    if ec is None or (len(ec) == 4 and ec[3] == 0):
                        ec, elw = "#444444", 0.5
                    else:
                        elw = max(h.get_linewidth(), 0.8)
                    ax_panel.add_patch(mpatches.Rectangle(
                        (0.05, y_e - swatch_h * 0.65), 0.16, swatch_h,
                        facecolor=h.get_facecolor(), edgecolor=ec,
                        linestyle=h.get_linestyle() or "-", linewidth=elw,
                        transform=ax_panel.transAxes, zorder=5,
                    ))
                ax_panel.text(
                    0.26, y_e - swatch_h * 0.08, str(h.get_label())[:26],
                    transform=ax_panel.transAxes,
                    va="center", fontsize=label_fs, color="#111",
                )

    # ─ Sección 2: CONVENCIONES (y_div2 → y_div1) ─────────────────────────────
    y_cv_top = y_div1 - 0.014
    ax_panel.text(
        0.50, y_cv_top, "CONVENCIONES",
        transform=ax_panel.transAxes,
        ha="center", va="top",
        fontsize=9, fontweight="bold", color="#222",
    )

    geom_mode = gdf_3857.geom_type.dropna().mode()
    geom_type = geom_mode[0] if len(geom_mode) else "Unknown"
    geom_label = (
        "Polígono" if "Polygon" in geom_type
        else "Línea"  if "Line"    in geom_type
        else "Punto"
    )
    crs_auth = gdf.crs.to_authority() if gdf.crs else None
    crs_orig = f"EPSG:{crs_auth[1]}" if crs_auth else "Sin CRS"
    n_feats  = len(gdf_3857)

    conv_row = (y_div1 - y_div2 - 0.060) / 3
    for k, (lbl, val) in enumerate([
        ("Geometría:", geom_label),
        ("CRS orig:", crs_orig),
        ("Features:", f"{n_feats:,}"),
    ]):
        y_cv = y_cv_top - 0.034 - k * conv_row
        ax_panel.text(0.07, y_cv, lbl, transform=ax_panel.transAxes,
                      va="top", fontsize=7, color="#444", fontweight="bold")
        ax_panel.text(0.45, y_cv, val, transform=ax_panel.transAxes,
                      va="top", fontsize=7, color="#111")

    # ─ Sección 3: ÍNDICE DE LOCALIZACIÓN (0.00 → y_div2) ─────────────────────
    y_loc_title = y_div2 - 0.014
    ax_panel.text(
        0.50, y_loc_title, "ÍNDICE DE LOCALIZACIÓN",
        transform=ax_panel.transAxes,
        ha="center", va="top",
        fontsize=9, fontweight="bold", color="#222",
    )

    # Inset axes: ocupa de y=0.015 hasta y=y_div2-0.065 del panel
    loc_h = y_div2 - 0.065 - 0.015
    ax_loc = ax_panel.inset_axes([0.04, 0.015, 0.92, loc_h])

    # Cargar naturalearth con múltiples rutas de fallback
    def _try_naturalearth():
        import pyogrio as _pyogrio
        _candidates = [
            # pyogrio trae su propio fixture (dependencia ya requerida)
            os.path.join(os.path.dirname(_pyogrio.__file__),
                         "tests", "fixtures", "naturalearth_lowres",
                         "naturalearth_lowres.shp"),
            # geopandas ≤ 0.x (antes de la eliminación)
            os.path.join(os.path.dirname(gpd.__file__),
                         "datasets", "naturalearth_lowres",
                         "naturalearth_lowres.shp"),
        ]
        for _p in _candidates:
            try:
                if os.path.exists(_p):
                    _r = gpd.read_file(_p)
                    if not _r.empty:
                        return _r
            except Exception:
                pass
        # Último intento: API deprecada de geopandas (funciona en < 1.0)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        except Exception:
            pass
        return None

    _world = _try_naturalearth()

    if _world is not None:
        _world.to_crs(epsg=4326).plot(
            ax=ax_loc, color="#d8d8d8", edgecolor="#aaa", linewidth=0.25,
        )
        wb   = gdf_wgs84.total_bounds
        dx_w = wb[2] - wb[0]
        dy_w = wb[3] - wb[1]
        ax_loc.add_patch(mpatches.Rectangle(
            (wb[0], wb[1]), dx_w, dy_w,
            facecolor="#e74c3c", edgecolor="#c0392b", linewidth=1.5, alpha=0.60,
        ))
        if dx_w < 0.5 and dy_w < 0.5:
            ax_loc.plot(
                (wb[0] + wb[2]) / 2, (wb[1] + wb[3]) / 2,
                "ro", markersize=5, zorder=10,
            )
        cx_d   = (wb[0] + wb[2]) / 2
        cy_d   = (wb[1] + wb[3]) / 2
        margin = max(8.0, max(dx_w, dy_w) * 4)
        ax_loc.set_xlim(cx_d - margin, cx_d + margin)
        ax_loc.set_ylim(cy_d - margin, cy_d + margin)
    else:
        ax_loc.set_facecolor("#eeeeee")
        ax_loc.text(0.5, 0.5, "Sin datos\nde contexto",
                    transform=ax_loc.transAxes,
                    ha="center", va="center", fontsize=6, color="#bbb")

    ax_loc.set_aspect("equal", adjustable="datalim")
    ax_loc.axis("off")
    for sp in ax_loc.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor("#bbb")
        sp.set_linewidth(0.5)

    # ── BARRA INFERIOR ───────────────────────────────────────────────────────
    # Separador superior
    ax_bottom.plot([0, 1], [1, 1], transform=ax_bottom.transAxes,
                   color="#555", linewidth=1.5, solid_capstyle="round")
    # Separadores internos verticales: 0.07 (norte↔escala), 0.36 (escala↔params)
    for x_div in (0.07, 0.36):
        ax_bottom.plot([x_div, x_div], [0.05, 0.95], transform=ax_bottom.transAxes,
                       color="#888", linewidth=1.0)

    # ─ Norte (izquierda, x 0→0.07) ───────────────────────────────────────────
    # Flecha de norte: línea + punta de flecha + "N"
    _nx = 0.035
    ax_bottom.annotate(
        "", xy=(_nx, 0.82), xytext=(_nx, 0.30),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="#111",
                        lw=1.0, mutation_scale=9),
    )
    ax_bottom.text(
        _nx, 0.16, "N",
        transform=ax_bottom.transAxes,
        ha="center", va="center", fontsize=10, color="#111", fontweight="bold",
    )

    # ─ Barra de escala (centro, x 0.07→0.36) ─────────────────────────────────
    b3857    = gdf_3857.total_bounds
    _corr    = _mercator_lat_correction((b3857[1] + b3857[3]) / 2)
    map_w_m  = abs(b3857[2] - b3857[0]) * 1.2 * _corr
    nice_vals = [
        1, 2, 5, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000,
        10000, 20000, 50000, 100000, 200000, 500000, 1000000,
    ]
    target_m = map_w_m * 0.25
    bar_m    = min(nice_vals, key=lambda v: abs(v - target_m))
    bar_lbl  = f"{bar_m:,} m" if bar_m < 1000 else f"{bar_m // 1000:,} km"

    # El bloque de la escala ocupa x=0.07..0.36 del axes (29 % del total)
    # Dibujamos la barra en x=0.10..0.32 del axes, centrada en esa banda
    sb_x0 = 0.10
    sb_x1 = 0.33
    sb_mid = (sb_x0 + sb_x1) / 2
    sb_w   = sb_x1 - sb_x0
    sb_y   = 0.52
    sb_h   = 0.22
    for i, fc in enumerate(["#111", "#eee"]):
        ax_bottom.add_patch(mpatches.Rectangle(
            (sb_x0 + i * sb_w / 2, sb_y - sb_h / 2), sb_w / 2, sb_h,
            transform=ax_bottom.transAxes,
            facecolor=fc, edgecolor="#111", linewidth=0.5, zorder=10,
        ))
    ax_bottom.text(sb_x0, sb_y + sb_h * 0.65, "0",
                   transform=ax_bottom.transAxes,
                   ha="center", va="bottom", fontsize=7.5, color="#111")
    ax_bottom.text(sb_mid, sb_y + sb_h * 0.65, bar_lbl,
                   transform=ax_bottom.transAxes,
                   ha="center", va="bottom", fontsize=7.5, color="#111")
    ax_bottom.text(sb_mid, 0.14, "ESCALA GRÁFICA",
                   transform=ax_bottom.transAxes,
                   ha="center", va="center", fontsize=6.5, color="#444")

    # ─ Parámetros / fuente (derecha, x 0.36→1.00) ────────────────────────────
    credit_text = credits or "Fuente: [No especificada] | Elaborado con GeoAnalisis MCP"
    wb84        = gdf_wgs84.total_bounds
    params_lines = [
        credit_text,
        f"CRS datos: {crs_orig}  |  Renderizado: WGS84 Pseudo-Mercator EPSG:3857",
        (f"Ext: [{wb84[0]:.4f}, {wb84[1]:.4f}, "
         f"{wb84[2]:.4f}, {wb84[3]:.4f}]  (WGS84)"),
        f"Generado: {date.today().isoformat()}  |  GeoAnalisis MCP",
    ]
    for k, line in enumerate(params_lines):
        ax_bottom.text(
            0.38, 0.86 - k * 0.22, line,
            transform=ax_bottom.transAxes,
            va="top", fontsize=6.5, color="#333",
            fontfamily="monospace" if k == 2 else "sans-serif",
        )

    # ── Marco exterior de la lámina ───────────────────────────────────────────
    for r in (
        mpatches.Rectangle((0.003, 0.003), 0.994, 0.994,
                            fill=False, edgecolor="#333", linewidth=2.2,
                            transform=fig.transFigure, zorder=200),
        mpatches.Rectangle((0.008, 0.008), 0.984, 0.984,
                            fill=False, edgecolor="#888", linewidth=0.8,
                            transform=fig.transFigure, zorder=200),
    ):
        fig.add_artist(r)

    # ── Layout: subplots llenan el espacio interior al borde de la lámina ────
    fig.subplots_adjust(
        left=0.010, right=0.990,
        top=0.990,  bottom=0.010,
        hspace=0.0, wspace=0.0,
    )

    # ── Etiquetas (tras el layout final: la detección de colisiones usa
    #     coordenadas de pantalla, que cambian con subplots_adjust) ───────────
    if label_by and label_by in gdf_3857.columns:
        _place_labels(ax_map, gdf_3857, label_by, fontsize=7)

    # ── Guardar en disco (no aborta si falla) ─────────────────────────────────
    disk_path = _save_figure(fig, output_path, path, resolved_layer, "cartographic",
                             dpi, jpeg_quality=92)

    # ── Reporte de simbología + imagen inline para el chat ───────────────────
    report = _symbology_report(
        resolved_layer, effective_field, style_type,
        len(legend_patches), crs_note, extra_notes, disk_path,
    )
    return [report] + _inline_image(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
