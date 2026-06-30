#!/usr/bin/env python3
"""GeoAnalisis MCP Server — Herramientas de lectura y análisis de datos espaciales vectoriales."""

from __future__ import annotations

import json
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from mcp.server.fastmcp import FastMCP
from mcp.types import EmbeddedResource, ImageContent, TextResourceContents
from shapely import set_precision, to_wkt


mcp = FastMCP(
    "GeoAnalisis",
    instructions=(
        "Servidor MCP para análisis de datos espaciales vectoriales. "
        "Soporta FileGDB (.gdb), Shapefile (.shp), GeoJSON, GeoPackage (.gpkg), KML y "
        "cualquier formato vectorial compatible con GDAL/OGR."
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


# Presupuesto de caracteres para el GeoJSON inline.
# El HTML + JS del template ocupa ~18 KB; el límite de output del tool en Claude
# es ~55 K caracteres, dejando ~37 K para los datos.
_MAX_GEOJSON_CHARS = 37_000


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
            crs = info.get("crs")
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

    crs = info.get("crs")
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
    gdf = gpd.read_file(path, layer=layer, rows=max_features, engine="pyogrio")
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
    """
    if bbox and len(bbox) != 4:
        return _json({"error": "bbox debe ser una lista de 4 números: [xmin, ymin, xmax, ymax]"})

    gdf = _read_gdf(path, layer, limit, where, bbox)
    return gdf.to_json(indent=2, na="null")


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
    if gdf.crs is None:
        return _err("La capa no tiene CRS definido. Defínelo antes de renderizar.")

    resolved_layer = layer or pyogrio.list_layers(path)[0][0]
    gdf_wgs = gdf.to_crs(epsg=4326)
    for col in gdf_wgs.columns:
        if pd.api.types.is_datetime64_any_dtype(gdf_wgs[col]):
            gdf_wgs[col] = gdf_wgs[col].astype(str)

    geojson_str, size_warning = _geojson_for_render(gdf_wgs)
    feature_count = len(gdf_wgs)

    color_map: dict = {}
    if color_by and color_by in gdf_wgs.columns:
        cats = [str(v) for v in gdf_wgs[color_by].dropna().unique()]
        color_map = {c: _PALETTE[i % len(_PALETTE)] for i, c in enumerate(cats)}

    color_map_js = json.dumps(color_map)
    color_by_js  = json.dumps(color_by)

    warning_suffix = f" ⚠ {size_warning}" if size_warning else ""
    title_text = f"{resolved_layer} — {feature_count:,} features{warning_suffix} | GeoAnalisis MCP"

    legend_js = ""
    if color_map:
        items_js = json.dumps([{"label": cat, "color": col} for cat, col in color_map.items()])
        legend_js = f"""
const legend = L.control({{position: 'bottomright'}});
legend.onAdd = () => {{
  const d = L.DomUtil.create('div', 'ga-legend');
  const items = {items_js};
  d.innerHTML = '<div class="ga-lt">{color_by}</div>' +
    items.map(i => `<div class="ga-li"><span class="ga-sw" style="background:${{i.color}}"></span><span class="ga-lb">${{i.label}}</span></div>`).join('');
  return d;
}};
legend.addTo(map);"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{resolved_layer} — GeoAnalisis MCP</title>
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
      .map(([k, v]) => `<tr><td style="font-weight:600;padding-right:8px;white-space:nowrap;color:#555">${{k}}</td><td>${{v}}</td></tr>`)
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
            text=html,
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
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    values = gdf[field].dropna()
    if values.empty:
        return "#2980b9", [], None

    vmin, vmax = float(values.min()), float(values.max())
    if isinstance(breaks_param, list):
        breaks = [float(b) for b in breaks_param]
    else:
        n = max(2, int(breaks_param)) if breaks_param else 5
        breaks = list(np.linspace(vmin, vmax, n + 1))

    cmap = cm.get_cmap(ramp or "RdYlGn_r")
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


def _add_scalebar(ax) -> None:
    """Barra de escala en esquina inferior izquierda (ejes en metros, EPSG:3857)."""
    import matplotlib.patches as mpatches

    xlim = ax.get_xlim()
    map_width_m = abs(xlim[1] - xlim[0])
    if map_width_m == 0:
        return

    target_m = map_width_m * 0.18
    nice_vals = [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000,
                 10000, 20000, 50000, 100000, 200000]
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
) -> list[ImageContent]:
    """
    Genera una imagen del mapa con basemap CartoDB Positron mostrada INLINE en el chat
    y guardada en disco en alta resolución.

    Args:
        path:          Ruta al archivo o directorio espacial.
        layer:         Nombre de la capa (None = primera capa).
        limit:         Máximo de features (default 5000).
        color_by:      Campo para colorear por categoría con colores automáticos.
                       Alternativa simple a `style`.
        where:         Filtro SQL OGR.
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
        figsize:       Tamaño [ancho, alto] en pulgadas, ej. [14, 8.5] para A4 horizontal.
                       Default [10, 6].
        scalebar:      Mostrar barra de escala en esquina inferior izquierda (default True).
        north_arrow:   Mostrar flecha de norte en esquina superior izquierda (default True).
        credits:       Texto de créditos en pie de mapa,
                       ej. "Fuente: INEGI 2024 | Elaborado con GeoAnalisis MCP".
        label_by:      Campo para etiquetar cada feature en su centroide.
        extra_layers:  Capas adicionales a superponer sobre el basemap, en orden ascendente
                       (primera entrada = capa más baja). Cada entrada es un dict con:
                         path (str, requerido), layer (str), limit (int, default 10000),
                         color (str, default "#888"), alpha (float, default 0.7),
                         linewidth (float, default 0.5), edgecolor (str, default "none"),
                         markersize (float, default 4).
    """
    import base64
    import io
    import os
    import sys

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import contextily as cx

    if bbox and len(bbox) != 4:
        raise ValueError("bbox debe ser [xmin, ymin, xmax, ymax].")

    gdf = _read_gdf(path, layer, limit, where, bbox)
    if gdf.empty:
        raise ValueError("La capa no tiene features para renderizar.")
    if gdf.crs is None:
        raise ValueError("La capa no tiene CRS definido.")

    resolved_layer = layer or pyogrio.list_layers(path)[0][0]
    gdf_3857 = gdf.to_crs(epsg=3857)

    # --- Resolver colores según tipo de estilo ---
    effective_field = color_by
    custom_categories: dict = {}
    style_type = "single"
    if style:
        style_type = style.get("type", "single")
        if style_type in ("categorized", "graduated"):
            effective_field = style.get("field", color_by)
        if style_type == "categorized":
            custom_categories = style.get("categories") or {}

    legend_patches: list = []

    if style_type == "graduated" and effective_field and effective_field in gdf_3857.columns:
        plot_colors, legend_patches, _ = _graduated_colors(
            gdf_3857, effective_field,
            style.get("ramp") if style else None,
            style.get("breaks", 5) if style else 5,
        )
    elif effective_field and effective_field in gdf_3857.columns:
        cats = [str(v) for v in gdf_3857[effective_field].dropna().unique()]
        if custom_categories:
            color_map = {
                c: custom_categories.get(c, _PALETTE[i % len(_PALETTE)])
                for i, c in enumerate(cats)
            }
        else:
            color_map = {c: _PALETTE[i % len(_PALETTE)] for i, c in enumerate(cats)}
        plot_colors = gdf_3857[effective_field].apply(
            lambda v: _to_rgba(color_map.get(str(v) if pd.notna(v) else "", "#aaaaaa"))
        )
        legend_patches = [mpatches.Patch(color=col, label=cat) for cat, col in color_map.items()]
    else:
        plot_colors = "#2980b9"
        legend_patches = [mpatches.Patch(color="#2980b9", label=resolved_layer)]
        effective_field = None

    # --- Figura: tamaño adaptado al aspect ratio de los datos ---
    if figsize and len(figsize) == 2:
        fw, fh = float(figsize[0]), float(figsize[1])
    else:
        b = gdf_3857.total_bounds  # [xmin, ymin, xmax, ymax]
        dx = max(float(b[2] - b[0]), 1.0)
        dy = max(float(b[3] - b[1]), 1.0)
        ratio = dy / dx
        fw = 10.0 / max(ratio, 1.0) if ratio >= 1.0 else 10.0
        fh = fw * ratio
        fw = max(5.0, min(fw, 14.0))
        fh = max(5.0, min(fh, 12.0))
    fig, ax = plt.subplots(figsize=(fw, fh))
    fig.patch.set_facecolor("#f5f4f1")
    ax.set_facecolor("#e8e5e0")

    # --- Capas adicionales (debajo de la principal) ---
    if extra_layers:
        for extra in extra_layers:
            try:
                e_gdf = _read_gdf(
                    extra["path"], extra.get("layer"),
                    extra.get("limit", 10000), None, None,
                )
                if not e_gdf.empty and e_gdf.crs is not None:
                    e_gdf.to_crs(epsg=3857).plot(
                        ax=ax,
                        color=extra.get("color", "#888888"),
                        alpha=extra.get("alpha", 0.7),
                        linewidth=extra.get("linewidth", 0.5),
                        edgecolor=extra.get("edgecolor", "none"),
                        markersize=extra.get("markersize", 4),
                    )
            except Exception as exc:
                print(f"[GeoAnalisis MCP] extra_layer ignorada: {exc}", file=sys.stderr)

    # --- Capa principal ---
    gdf_3857.plot(
        ax=ax,
        color=plot_colors,
        edgecolor=(0, 0, 0, 0.25),
        linewidth=0.6,
        markersize=6,
        alpha=0.75,
    )

    # --- Basemap ---
    cx.add_basemap(
        ax,
        crs=gdf_3857.crs,
        source=cx.providers.CartoDB.Positron,
        attribution="© OpenStreetMap contributors © CARTO",
    )
    ax.set_axis_off()
    ax.add_patch(mpatches.Rectangle(
        (0, 0), 1, 1,
        transform=ax.transAxes,
        fill=False, edgecolor="#888888", linewidth=0.8, zorder=30,
    ))

    # --- Etiquetas por campo ---
    if label_by and label_by in gdf_3857.columns:
        _place_labels(ax, gdf_3857, label_by, fontsize=8)

    # --- Barra de escala ---
    if scalebar:
        _add_scalebar(ax)

    # --- Símbolo de norte (SVG path, sin fondo) ---
    if north_arrow:
        _draw_north_svg(ax, cx=0.075, cy=0.910, size=0.07, zorder=16)

    # --- Leyenda ---
    if legend and legend_patches:
        ax.legend(
            handles=legend_patches,
            title=effective_field,
            title_fontsize=10,
            fontsize=9,
            loc="lower right",
            framealpha=0.93,
            edgecolor="#cccccc",
        )

    # --- Créditos en pie ---
    if credits:
        fig.text(0.5, 0.01, credits,
                 ha="center", va="bottom", fontsize=7, color="#555", style="italic")

    ax.set_title(
        f"{resolved_layer}  |  GeoAnalisis MCP",
        fontsize=12, fontweight="bold", color="#222",
        pad=8,
    )
    plt.tight_layout(rect=[0, 0.03 if credits else 0.0, 1, 1.0])

    # --- Guardar en disco ---
    if output_path:
        disk_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(disk_path) or ".", exist_ok=True)
        ext = os.path.splitext(disk_path)[1].lower()
        if ext == ".png":
            fig.savefig(disk_path, format="png", dpi=dpi, bbox_inches="tight")
        elif ext == ".pdf":
            fig.savefig(disk_path, format="pdf", dpi=dpi, bbox_inches="tight")
        elif ext == ".svg":
            fig.savefig(disk_path, format="svg", bbox_inches="tight")
        else:
            fig.savefig(disk_path, format="jpeg", dpi=dpi, bbox_inches="tight",
                        pil_kwargs={"quality": 90})
    else:
        src_dir = os.path.dirname(os.path.abspath(path))
        safe_name = resolved_layer.replace(" ", "_").replace("/", "-")
        disk_path = os.path.join(src_dir, f"{safe_name}_map.jpg")
        fig.savefig(disk_path, format="jpeg", dpi=dpi, bbox_inches="tight",
                    pil_kwargs={"quality": 90})

    # --- Imagen inline para el chat (72 DPI, JPEG → ~30-80 KB) ---
    buf = io.BytesIO()
    fig.savefig(buf, format="jpeg", dpi=72, bbox_inches="tight", pil_kwargs={"quality": 70})
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()

    return [ImageContent(type="image", data=img_b64, mimeType="image/jpeg")]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
