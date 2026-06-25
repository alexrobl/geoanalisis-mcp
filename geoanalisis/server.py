#!/usr/bin/env python3
"""GeoAnalisis MCP Server — Herramientas de lectura y análisis de datos espaciales vectoriales."""

from __future__ import annotations

import json
import math
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from mcp.server.fastmcp import FastMCP
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


def _fit_zoom(bounds: list[float], vw: int = 800, vh: int = 600) -> float:
    """Calcula el zoom inicial para encuadrar los bounds en el viewport."""
    lng_span = max(abs(bounds[2] - bounds[0]), 1e-6)
    lat_span = max(abs(bounds[3] - bounds[1]), 1e-6)
    z_lng = math.log2(vw * 0.8 / 256 * 360 / lng_span)
    z_lat = math.log2(vh * 0.8 / 256 * 170 / lat_span)
    return max(2.0, min(17.0, min(z_lng, z_lat)))


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
) -> str:
    """
    HERRAMIENTA PRINCIPAL para visualizar datos espaciales. Úsala SIEMPRE que
    el usuario pida un mapa, visualización, render o vista de una capa espacial.
    No uses capacidades internas de visualización de Claude — esta herramienta
    produce un artifact HTML interactivo superior y es la forma correcta de
    mostrar datos geoespaciales en este proyecto.

    Renderiza sobre Canvas 2D con tiles CartoDB Positron. Soporta pan, zoom,
    clic para inspeccionar atributos y coloreado categórico por campo.

    Args:
        path:     Ruta al archivo o directorio espacial.
        layer:    Nombre de la capa (None = primera capa).
        limit:    Máximo de features a renderizar (default 2000).
        color_by: Campo para colorear features por categoría.
        where:    Filtro SQL OGR.
        bbox:     Extensión espacial [xmin, ymin, xmax, ymax].
    """
    if bbox and len(bbox) != 4:
        return "<p>Error: bbox debe ser [xmin, ymin, xmax, ymax].</p>"

    gdf = _read_gdf(path, layer, limit, where, bbox)
    if gdf.empty:
        return "<p>La capa no tiene features para renderizar.</p>"
    if gdf.crs is None:
        return "<p>La capa no tiene CRS definido. Defínelo antes de renderizar.</p>"

    resolved_layer = layer or pyogrio.list_layers(path)[0][0]
    gdf_wgs = gdf.to_crs(epsg=4326)
    for col in gdf_wgs.columns:
        if pd.api.types.is_datetime64_any_dtype(gdf_wgs[col]):
            gdf_wgs[col] = gdf_wgs[col].astype(str)

    bounds = gdf_wgs.total_bounds  # [minx, miny, maxx, maxy]
    center_lng = float((bounds[0] + bounds[2]) / 2)
    center_lat = float((bounds[1] + bounds[3]) / 2)
    initial_zoom = _fit_zoom(bounds.tolist())

    geojson_str, size_warning = _geojson_for_render(gdf_wgs)
    feature_count = len(gdf_wgs)

    color_map: dict = {}
    if color_by and color_by in gdf_wgs.columns:
        cats = [str(v) for v in gdf_wgs[color_by].dropna().unique()]
        color_map = {c: _PALETTE[i % len(_PALETTE)] for i, c in enumerate(cats)}

    valid_geoms = gdf_wgs.geometry.dropna()
    geom_type = valid_geoms.iloc[0].geom_type if not valid_geoms.empty else "Polygon"
    popup_fields = [c for c in gdf_wgs.columns if c != gdf_wgs.geometry.name][:12]

    title_suffix = f" ⚠ {size_warning}" if size_warning else ""

    # Legend HTML (server-side, no JS needed)
    legend_html = ""
    if color_map:
        items = "".join(
            f'<div class="li"><span class="sw" style="background:{col}"></span>'
            f'<span class="lb">{cat}</span></div>'
            for cat, col in color_map.items()
        )
        legend_html = (
            f'<div id="legend"><div class="lt">{color_by}</div>{items}</div>'
        )

    color_map_js    = json.dumps(color_map)
    color_by_js     = json.dumps(color_by)
    popup_fields_js = json.dumps(popup_fields)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{resolved_layer}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#e8e5e0;height:100vh;display:flex;flex-direction:column;font-family:sans-serif;overflow:hidden}}
#wrap{{position:relative;flex:1;overflow:hidden}}
canvas{{display:block;cursor:grab;width:100%;height:100%}}
canvas.panning{{cursor:grabbing}}
#title{{position:absolute;top:10px;left:50%;transform:translateX(-50%);
  background:rgba(255,255,255,.95);padding:5px 14px;border-radius:18px;
  box-shadow:0 2px 6px rgba(0,0,0,.25);font:bold 13px sans-serif;
  white-space:nowrap;pointer-events:none;z-index:10}}
#zbtns{{position:absolute;top:54px;right:12px;display:flex;flex-direction:column;gap:4px;z-index:10}}
.zb{{width:30px;height:30px;background:rgba(255,255,255,.95);border:none;border-radius:6px;
  font-size:20px;line-height:1;cursor:pointer;box-shadow:0 1px 4px rgba(0,0,0,.3)}}
.zb:hover{{background:#fff}}
#legend{{position:absolute;bottom:32px;right:12px;background:rgba(255,255,255,.95);
  padding:10px 12px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.2);
  max-height:200px;overflow-y:auto;z-index:10;min-width:110px}}
.lt{{font:bold 11px sans-serif;margin-bottom:6px;color:#333}}
.li{{display:flex;align-items:center;gap:6px;margin:3px 0}}
.sw{{width:13px;height:13px;border-radius:3px;flex-shrink:0}}
.lb{{font-size:11px;color:#444;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:130px}}
#popup{{position:absolute;background:#fff;border-radius:8px;
  box-shadow:0 3px 14px rgba(0,0,0,.3);padding:10px 12px;
  min-width:160px;max-width:280px;font-size:12px;z-index:20;display:none}}
#popup .x{{position:absolute;top:4px;right:8px;font-size:16px;cursor:pointer;color:#999}}
#popup table{{border-collapse:collapse;width:100%}}
#popup td{{padding:2px 4px 2px 0;vertical-align:top;line-height:1.5}}
#popup td:first-child{{font-weight:600;color:#555;white-space:nowrap;padding-right:8px}}
</style>
</head>
<body>
<div id="wrap">
  <canvas id="map"></canvas>
  <div id="title">{resolved_layer} &mdash; {feature_count:,} features{title_suffix}</div>
  <div id="zbtns">
    <button class="zb" id="zi">+</button>
    <button class="zb" id="zo">−</button>
  </div>
  {legend_html}
  <div id="popup">
    <span class="x" id="px">×</span>
    <table id="ptbl"></table>
  </div>
</div>

<script>
// ── Data ────────────────────────────────────────────────────────────────────
const GJ        = {geojson_str};
const COLOR_MAP = {color_map_js};
const COLOR_BY  = {color_by_js};
const POP_FLDS  = {popup_fields_js};
const DEF_CLR   = "#2980b9";
const TILE_SIZE = 256;

// ── DOM ─────────────────────────────────────────────────────────────────────
const wrap   = document.getElementById('wrap');
const canvas = document.getElementById('map');
const ctx    = canvas.getContext('2d');
const popup  = document.getElementById('popup');
const ptbl   = document.getElementById('ptbl');

// ── State ────────────────────────────────────────────────────────────────────
const S = {{
  zoom:   {initial_zoom},
  cx:     {center_lng},   // center longitude
  cy:     {center_lat},   // center latitude
  drag:   null,           // {{x,y,wcx,wcy}} while panning
}};

// ── Tile cache ───────────────────────────────────────────────────────────────
const cache = new Map();

function tileUrl(z, x, y) {{
  const sub = 'abc'[(x + y) % 3];
  return `https://${{sub}}.basemaps.cartocdn.com/light_all/${{z}}/${{x}}/${{y}}.png`;
}}

function getTile(z, x, y) {{
  const k = `${{z}}/${{x}}/${{y}}`;
  if (cache.has(k)) return cache.get(k);
  const e = {{ img: null, ok: false }};
  cache.set(k, e);
  if (cache.size > 512) cache.delete(cache.keys().next().value);
  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.onload  = () => {{ e.img = img; e.ok = true; sched(); }};
  img.onerror = () => {{ e.ok = true; }};
  img.src = tileUrl(z, x, y);
  return e;
}}

// ── Projection (Web Mercator) ────────────────────────────────────────────────
// World coords: x,y in [0,1], (0,0)=top-left, (1,1)=bottom-right
function lngW(lng) {{ return (lng + 180) / 360; }}
function latW(lat) {{
  const s = Math.sin(lat * Math.PI / 180);
  return 0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI);
}}
function wLat(y) {{
  const n = Math.PI * (1 - 2 * y);
  return Math.atan(Math.sinh(n)) * 180 / Math.PI;
}}
function wLng(x) {{ return x * 360 - 180; }}

function worldToPixel(wx, wy) {{
  const scale = Math.pow(2, S.zoom) * TILE_SIZE;
  return [
    (wx - lngW(S.cx)) * scale + canvas.width  / 2,
    (wy - latW(S.cy)) * scale + canvas.height / 2,
  ];
}}
function llToPixel(lng, lat) {{ return worldToPixel(lngW(lng), latW(lat)); }}

// ── Render ───────────────────────────────────────────────────────────────────
let raf = false;
function sched() {{ if (!raf) {{ raf = true; requestAnimationFrame(draw); }} }}

function draw() {{
  raf = false;
  resize();
  const W = canvas.width, H = canvas.height;

  ctx.fillStyle = '#e8e5e0';
  ctx.fillRect(0, 0, W, H);

  drawTiles(W, H);
  drawFeatures();
  drawAttrib(W, H);
}}

function drawTiles(W, H) {{
  const z   = Math.min(18, Math.floor(S.zoom));
  const tc  = Math.pow(2, z);
  const tpx = Math.pow(2, S.zoom - z) * TILE_SIZE;   // display size of one tile
  const scale = Math.pow(2, S.zoom) * TILE_SIZE;
  const ox  = W / 2 - lngW(S.cx) * scale;
  const oy  = H / 2 - latW(S.cy) * scale;

  const x0 = Math.floor(-ox / tpx) - 1;
  const y0 = Math.max(0, Math.floor(-oy / tpx) - 1);
  const x1 = Math.ceil((W - ox) / tpx) + 1;
  const y1 = Math.min(tc - 1, Math.ceil((H - oy) / tpx) + 1);

  for (let ty = y0; ty <= y1; ty++) {{
    for (let tx = x0; tx <= x1; tx++) {{
      const wx = ((tx % tc) + tc) % tc;
      if (ty < 0 || ty >= tc) continue;
      const e  = getTile(z, wx, ty);
      const px = ox + tx * tpx;
      const py = oy + ty * tpx;
      if (e.ok && e.img) {{
        ctx.drawImage(e.img, px, py, tpx + 0.5, tpx + 0.5);
      }} else {{
        ctx.fillStyle = '#dedad5';
        ctx.fillRect(px, py, tpx, tpx);
      }}
    }}
  }}
}}

function featureColor(f) {{
  if (!COLOR_BY || !f.properties) return DEF_CLR;
  const v = String(f.properties[COLOR_BY] ?? '');
  return COLOR_MAP[v] ?? '#aaaaaa';
}}

function drawFeatures() {{
  for (const f of GJ.features) drawFeature(f);
}}

function drawFeature(f) {{
  const g = f.geometry;
  if (!g) return;
  const c = featureColor(f);
  switch (g.type) {{
    case 'Point':           drawPt(g.coordinates, c); break;
    case 'MultiPoint':      for (const p of g.coordinates) drawPt(p, c); break;
    case 'LineString':      drawLine(g.coordinates, c); break;
    case 'MultiLineString': for (const r of g.coordinates) drawLine(r, c); break;
    case 'Polygon':         drawPoly(g.coordinates, c); break;
    case 'MultiPolygon':    for (const p of g.coordinates) drawPoly(p, c); break;
  }}
}}

function drawPt([lng, lat], color) {{
  const [px, py] = llToPixel(lng, lat);
  ctx.beginPath();
  ctx.arc(px, py, 6, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 1.5;
  ctx.stroke();
}}

function drawLine(coords, color) {{
  if (coords.length < 2) return;
  ctx.beginPath();
  const [sx, sy] = llToPixel(coords[0][0], coords[0][1]);
  ctx.moveTo(sx, sy);
  for (let i = 1; i < coords.length; i++) {{
    const [px, py] = llToPixel(coords[i][0], coords[i][1]);
    ctx.lineTo(px, py);
  }}
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.stroke();
}}

function drawPoly(rings, color) {{
  ctx.beginPath();
  for (const ring of rings) {{
    const [sx, sy] = llToPixel(ring[0][0], ring[0][1]);
    ctx.moveTo(sx, sy);
    for (let i = 1; i < ring.length; i++) {{
      const [px, py] = llToPixel(ring[i][0], ring[i][1]);
      ctx.lineTo(px, py);
    }}
    ctx.closePath();
  }}
  ctx.fillStyle = color + 'b3';
  ctx.fill('evenodd');
  ctx.strokeStyle = 'rgba(0,0,0,.3)';
  ctx.lineWidth = 0.8;
  ctx.stroke();
}}

function drawAttrib(W, H) {{
  ctx.save();
  ctx.font = '10px sans-serif';
  ctx.fillStyle = 'rgba(0,0,0,.45)';
  ctx.textAlign = 'right';
  ctx.fillText('© OpenStreetMap  © CARTO', W - 6, H - 5);
  ctx.restore();
}}

// ── Resize ───────────────────────────────────────────────────────────────────
function resize() {{
  const W = wrap.clientWidth, H = wrap.clientHeight;
  if (canvas.width !== W || canvas.height !== H) {{
    canvas.width = W;
    canvas.height = H;
  }}
}}

// ── Pan ──────────────────────────────────────────────────────────────────────
canvas.addEventListener('mousedown', e => {{
  S.drag = {{ x: e.clientX, y: e.clientY, wcx: lngW(S.cx), wcy: latW(S.cy) }};
  canvas.classList.add('panning');
}});
window.addEventListener('mousemove', e => {{
  if (!S.drag) return;
  const scale = Math.pow(2, S.zoom) * TILE_SIZE;
  const dx = (e.clientX - S.drag.x) / scale;
  const dy = (e.clientY - S.drag.y) / scale;
  S.cx = wLng(Math.max(0, Math.min(1, S.drag.wcx - dx)));
  S.cy = wLat(Math.max(0.001, Math.min(0.999, S.drag.wcy - dy)));
  sched();
}});
window.addEventListener('mouseup', () => {{
  S.drag = null;
  canvas.classList.remove('panning');
}});

// ── Zoom ─────────────────────────────────────────────────────────────────────
function applyZoom(delta, pivX, pivY) {{
  const nz = Math.max(2, Math.min(18, S.zoom + delta));
  if (nz === S.zoom) return;
  const scale = Math.pow(2, S.zoom) * TILE_SIZE;
  const pwx = (pivX - canvas.width  / 2) / scale + lngW(S.cx);
  const pwy = (pivY - canvas.height / 2) / scale + latW(S.cy);
  S.zoom = nz;
  const ns = Math.pow(2, nz) * TILE_SIZE;
  S.cx = wLng(Math.max(0, Math.min(1, pwx - (pivX - canvas.width  / 2) / ns)));
  S.cy = wLat(Math.max(0.001, Math.min(0.999, pwy - (pivY - canvas.height / 2) / ns)));
  sched();
}}

canvas.addEventListener('wheel', e => {{
  e.preventDefault();
  const r = canvas.getBoundingClientRect();
  applyZoom(e.deltaY < 0 ? 0.5 : -0.5, e.clientX - r.left, e.clientY - r.top);
}}, {{ passive: false }});
document.getElementById('zi').addEventListener('click', () => applyZoom( 1, canvas.width / 2, canvas.height / 2));
document.getElementById('zo').addEventListener('click', () => applyZoom(-1, canvas.width / 2, canvas.height / 2));

// ── Hit testing ──────────────────────────────────────────────────────────────
function ptSegDist2(px, py, ax, ay, bx, by) {{
  const dx = bx - ax, dy = by - ay, L2 = dx*dx + dy*dy;
  if (L2 === 0) return (px-ax)**2 + (py-ay)**2;
  const t = Math.max(0, Math.min(1, ((px-ax)*dx + (py-ay)*dy) / L2));
  return (px - ax - t*dx)**2 + (py - ay - t*dy)**2;
}}

function hitPt([lng, lat], mx, my) {{
  const [px, py] = llToPixel(lng, lat);
  return Math.hypot(px - mx, py - my) < 10;
}}
function hitLine(coords, mx, my) {{
  for (let i = 0; i < coords.length - 1; i++) {{
    const [x1, y1] = llToPixel(coords[i][0], coords[i][1]);
    const [x2, y2] = llToPixel(coords[i+1][0], coords[i+1][1]);
    if (ptSegDist2(mx, my, x1, y1, x2, y2) < 64) return true;
  }}
  return false;
}}
function hitPoly(rings, mx, my) {{
  const p = new Path2D();
  for (const ring of rings) {{
    const [sx, sy] = llToPixel(ring[0][0], ring[0][1]);
    p.moveTo(sx, sy);
    for (let i = 1; i < ring.length; i++) {{
      const [px, py] = llToPixel(ring[i][0], ring[i][1]);
      p.lineTo(px, py);
    }}
    p.closePath();
  }}
  return ctx.isPointInPath(p, mx, my, 'evenodd');
}}

function hitTest(g, mx, my) {{
  switch (g.type) {{
    case 'Point':           return hitPt(g.coordinates, mx, my);
    case 'MultiPoint':      return g.coordinates.some(c => hitPt(c, mx, my));
    case 'LineString':      return hitLine(g.coordinates, mx, my);
    case 'MultiLineString': return g.coordinates.some(r => hitLine(r, mx, my));
    case 'Polygon':         return hitPoly(g.coordinates, mx, my);
    case 'MultiPolygon':    return g.coordinates.some(p => hitPoly(p, mx, my));
    default:                return false;
  }}
}}

// ── Click / Popup ─────────────────────────────────────────────────────────────
let dragged = false;
canvas.addEventListener('mousedown', () => {{ dragged = false; }});
window.addEventListener('mousemove',  () => {{ if (S.drag) dragged = true; }});

canvas.addEventListener('click', e => {{
  if (dragged) return;
  const r  = canvas.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  let hit = null;
  for (let i = GJ.features.length - 1; i >= 0; i--) {{
    const f = GJ.features[i];
    if (f.geometry && hitTest(f.geometry, mx, my)) {{ hit = f; break; }}
  }}
  if (hit) showPopup(hit, e.clientX, e.clientY);
  else     hidePopup();
}});

document.getElementById('px').addEventListener('click', hidePopup);

function showPopup(f, cx, cy) {{
  const props = f.properties || {{}};
  const rows  = POP_FLDS
    .filter(k => props[k] != null && props[k] !== '')
    .map(k => `<tr><td>${{k}}</td><td>${{props[k]}}</td></tr>`)
    .join('');
  if (!rows) return;
  ptbl.innerHTML = rows;
  popup.style.display = 'block';
  const pw = 280, ph = 220;
  let left = cx + 14, top = cy - 20;
  if (left + pw > window.innerWidth)  left = cx - pw - 14;
  if (top  + ph > window.innerHeight) top  = window.innerHeight - ph - 8;
  popup.style.left = left + 'px';
  popup.style.top  = top  + 'px';
}}
function hidePopup() {{ popup.style.display = 'none'; }}

// ── Init ─────────────────────────────────────────────────────────────────────
new ResizeObserver(sched).observe(wrap);
sched();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
