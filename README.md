# GeoAnalisis MCP

Servidor MCP para lectura y análisis de datos espaciales vectoriales, integrado con Claude Desktop.

## Herramientas

| Tool | Descripción |
|------|-------------|
| `list_layers` | Lista las capas de un archivo espacial con tipo de geometría, feature count y CRS |
| `get_layer_schema` | Esquema completo de una capa: campos, tipos, bbox, CRS |
| `scan_field_stats` | Estadísticas descriptivas por campo (numéricos y categóricos) |
| `read_features` | Lee features como GeoJSON FeatureCollection con filtros WHERE y bbox |
| `preview_geometries` | Vista previa de geometrías en WKT |
| `render_map` | Mapa interactivo HTML (Canvas 2D, pan/zoom/click, coloreado por campo) |

**Formatos soportados:** FileGDB (`.gdb`), Shapefile (`.shp`), GeoJSON, GeoPackage (`.gpkg`), KML y cualquier formato vectorial compatible con GDAL/OGR.

## Instalación

Requiere Python ≥ 3.11 y [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/dalexrobles/geoanalisis-mcp
cd geoanalisis-mcp
uv sync
```

## Configuración en Claude Desktop

Agrega esto a tu `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "geoanalisis": {
      "command": "/ruta/al/repo/.venv/bin/geoanalisis-mcp"
    }
  }
}
```

## render_map

Genera un artifact HTML interactivo directamente en Claude. Usa Canvas 2D API con tiles CartoDB Positron — sin WebWorkers, compatible con el sandbox de Claude.

- Pan arrastrando, zoom con rueda del ratón o botones
- Clic en un feature para ver sus atributos
- Parámetro `color_by` para colorear por campo categórico
- Simplificación geométrica automática para capas grandes
