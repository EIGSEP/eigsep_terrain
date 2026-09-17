import numpy as np

from eigsep_terrain.dem import DEM


def test_calc_horizon_crds_preserve_submeter_coordinates():
    dem = DEM()
    dem.res = 0.5
    dem.data = np.array([[-100, -100], [-100, 100]], dtype=np.float32)
    dem.map_crd = {'eastbc': 0, 'westbc': 0, 'northbc': 0, 'southbc': 0}
    dem.survey_offset = np.array([0, 0, 0])

    hangles, crds = dem.calc_horizon(0.0, 0.0, 0.0, n_az=64)

    assert np.issubdtype(crds.dtype, np.floating)
    assert np.any(np.isclose(crds[0], 0.5) & np.isclose(crds[1], 0.5))
    assert np.any(hangles > 0)
