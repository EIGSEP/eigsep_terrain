import os
import glob
import numpy as np
from urllib.request import urlretrieve

from .data import DATA_PATH
from .dem import DEM
from .dem import DEFAULT_BACKEND

SURVEY_OFFSET = np.array([-11, 36, 3])

#USGS_OPR_UT_WestEast_B22_12STJ%04d.tif'

NUM_EAST = 3
NUM_NORTH = 4
QUADS = tuple(L * 100 + B for L in range(91, 91 + NUM_NORTH)
                          for B in range(45, 45 + NUM_EAST))

FILE_BASE = 'USGS_OPR_UT_WestEast_B22_12STJ%04d'
URL_XML_BASE = 'https://thor-f5.er.usgs.gov/ngtoc/metadata/waf/elevation/opr_dem/geotiff/UT_WestEast_7_B22'
URL_TIF_BASE = 'https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/OPR/Projects/UT_WestEast_B22/UT_WestEast_7_B22/TIFF'

def get_xml_file(quad=min(QUADS), verbose=True):
    filename = (FILE_BASE % quad) + '.xml'
    outfile = os.path.join(DATA_PATH, filename)
    if not os.path.exists(outfile):
        if verbose:
            print('Downloading', os.path.join(URL_XML_BASE, filename))
        urlretrieve(os.path.join(URL_XML_BASE, filename), outfile)
    return outfile


def get_tif_files(quads=QUADS, verbose=True):
    filenames = [(FILE_BASE % quad) + '.tif' for quad in quads]
    outfiles = [os.path.join(DATA_PATH, f) for f in filenames]
    for f, outfile in zip(filenames, outfiles):
        if not os.path.exists(outfile):
            if verbose:
                print('Downloading', os.path.join(URL_TIF_BASE, f))
            urlretrieve(os.path.join(URL_TIF_BASE, f), outfile)
    outfiles = np.array(outfiles).reshape(NUM_NORTH, NUM_EAST)
    return outfiles

class MarjumDEM(DEM):
    def __init__(self, cache_file=None, clear_cache=False,
                 xml_file=None, tif_files=None,
                 survey_offset=SURVEY_OFFSET, verbose=True, backend=DEFAULT_BACKEND):
        DEM.__init__(self, cache_file=cache_file, clear_cache=clear_cache,
                     backend=backend)
        if self._cache_file == None or not os.path.exists(self._cache_file):
            xml_file = get_xml_file(verbose=verbose)
            tif_files = get_tif_files(verbose=verbose)
            self.load_xml(xml_file)
            self.load_tif(tif_files, survey_offset=SURVEY_OFFSET)
            self.save_cache()
