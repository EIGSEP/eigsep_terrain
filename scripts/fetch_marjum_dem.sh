#!/usr/bin/env bash
# Download the Marjum DEM files (12 TIFs + 1 XML metadata) into the package
# data dir. The thor-f5.er.usgs.gov host that originally served the XML now
# returns 403, so we pull it from the Wayback Machine instead. TIFs come
# straight from the public USGS S3 bucket.
set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/eigsep_terrain/data"
mkdir -p "$DEST"

FILE_BASE='USGS_OPR_UT_WestEast_B22_12STJ'
TIF_BASE='https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/OPR/Projects/UT_WestEast_B22/UT_WestEast_7_B22/TIFF'
# Wayback snapshot of the (now 403) thor-f5 metadata URL.
XML_WAYBACK='http://web.archive.org/web/20250530174128id_/https://thor-f5.er.usgs.gov/ngtoc/metadata/waf/elevation/opr_dem/geotiff/UT_WestEast_7_B22'

QUADS=(9145 9146 9147 9245 9246 9247 9345 9346 9347 9445 9446 9447)
MIN_QUAD=9145  # MarjumDEM only fetches the XML for the SW-most tile

fetch() {
    local url="$1" out="$2"
    if [[ -s "$out" ]]; then
        echo "skip (exists): $out"
        return
    fi
    echo "fetching: $url"
    curl -fSL --retry 3 -o "$out" "$url"
}

fetch "${XML_WAYBACK}/${FILE_BASE}${MIN_QUAD}.xml" \
      "${DEST}/${FILE_BASE}${MIN_QUAD}.xml"

for q in "${QUADS[@]}"; do
    fetch "${TIF_BASE}/${FILE_BASE}${q}.tif" \
          "${DEST}/${FILE_BASE}${q}.tif"
done

echo "done -> $DEST"
