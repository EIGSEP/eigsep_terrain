'''Utility functions for calculating distance and azimuth angles.'''
import numpy as np
from scipy.constants import c as C
import cv2

real_dtype = np.float32
C = real_dtype(C)

R_earth = 6378e3 # m


# copied from aipy.coord
def rot_m(ang, vec):
    """Return 3x3 matrix defined by rotation by 'ang' around the
    axis 'vec', according to the right-hand rule.  Both can be vectors,
    returning a vector of rotation matrices.  Rotation matrix will have a
    scaling of |vec| (i.e. normalize |vec|=1 for a pure rotation)."""
    c = np.cos(ang); s = np.sin(ang); C = 1-c 
    x,y,z = vec[...,0], vec[...,1], vec[...,2]
    xs,ys,zs = x*s, y*s, z*s 
    xC,yC,zC = x*C, y*C, z*C 
    xyC,yzC,zxC = x*yC, y*zC, z*xC
    rm = np.array([[x*xC+c, xyC-zs, zxC+ys],
                   [xyC+zs, y*yC+c, yzC-xs],
                   [zxC-ys, yzC+xs, z*zC+c]], dtype=np.double)
    if rm.ndim > 2:
        axes = list(range(rm.ndim))
        return rm.transpose(axes[-1:] + axes[:-1])
    else:
        return rm


def distance(a, b):
    '''Return distance from A to B.'''
    return np.linalg.norm(b - a)

def altitude(a, b):
    '''Return altitude of B relative to A, assuming ENU coordinates.'''
    return b[-1] - a[-1]

def azimuth(a, b):
    '''Return azimuthal angle [rad] of B viewed from A, assuming ENU coordinates.'''
    return np.arctan2(b[0] - a[0], b[1] - a[1])

def az_bin(e, n, n_az):
    '''Calculate azimuthal angle and round to nearest bin.'''
    az = np.arctan2(e[None, :], n[:, None])
    az = np.where(az < 0, 2 * np.pi + az, az)
    b = np.around(az / (2 * np.pi / n_az)).astype(int)
    return b
    
def calc_az_bin_range(e_edges, n_edges, e0, n0, n_az):
    '''Calculate the min/max az ranges a pixel could contain, based on
    where the pixel edges are.'''
    # (0, 0) is bottom-left
    # Letters are axis0: (t=top, m=middle, b=bottom),
    #             axis1: (l=left, c=center, r=right)
    de_edges = e_edges - e0
    dn_edges = n_edges - n0
    if n0 > n_edges[-1]:
        # b case
        b0, b1 = slice(0, -1), slice(1, None)
    elif n0 >= n_edges[0]:
        # full case
        n0_px = np.searchsorted(n_edges, n0) - 1
        b0, b1 = slice(      0, n0_px+0), slice(      1, n0_px+1)
        m0, m1 = slice(n0_px+0, n0_px+1), slice(n0_px+1, n0_px+2)
        t0, t1 = slice(n0_px+1,      -1), slice(n0_px+2,    None)
    else:  # n0_px < n_edges[0]
        # t case
        t0, t1 = slice(0, -1), slice(1, None)
    
    if e0 < e_edges[0]:
        # r case
        r0, r1 = slice(0, -1), slice(1, None)
        if n0 > n_edges[-1]:
            # br case
            ___a = az_bin(de_edges[r1], dn_edges[b1], n_az)
            ___b = az_bin(de_edges[r0], dn_edges[b0], n_az)
        elif n0 >= n_edges[0]:
            # r case
            b__a = az_bin(de_edges[r1], dn_edges[b1], n_az)
            b__b = az_bin(de_edges[r0], dn_edges[b0], n_az)
            m__a = az_bin(de_edges[r0], dn_edges[m1], n_az)
            m__b = az_bin(de_edges[r0], dn_edges[m0], n_az)
            t__a = az_bin(de_edges[r0], dn_edges[t1], n_az)
            t__b = az_bin(de_edges[r1], dn_edges[t0], n_az)
            ___a = np.concatenate([b__a, m__a, t__a], axis=0)
            ___b = np.concatenate([b__b, m__b, t__b], axis=0)
        else:  # n0 < n_edges[0]
            # tr case
            ___a = az_bin(de_edges[r0], dn_edges[t1], n_az)
            ___b = az_bin(de_edges[r1], dn_edges[t0], n_az)
    elif e0 <= e_edges[-1]:
        e0_px = np.searchsorted(e_edges, e0) - 1
        l0, l1 = slice(0, e0_px+0), slice(1, e0_px+1)
        c0, c1 = slice(e0_px+0, e0_px+1), slice(e0_px+1, e0_px+2)
        r0, r1 = slice(e0_px+1, -1), slice(e0_px+2, None)
        if n0 > n_edges[-1]:
            # bc case
            ___a = np.concatenate([az_bin(de_edges[se], dn_edges[ne], n_az)
                    for se, ne in zip([l1, c1, r1], [b0, b1, b1])], axis=1)
            ___b = np.concatenate([az_bin(de_edges[se], dn_edges[ne], n_az)
                    for se, ne in zip([l0, c0, r0], [b1, b1, b0])], axis=1)
        elif n0 >= n_edges[0]:
            # case
            b__a = np.concatenate([az_bin(de_edges[se], dn_edges[ne], n_az)
                    for se, ne in zip([l1, c1, r1], [b0, b1, b1])], axis=1)
            b__b = np.concatenate([az_bin(de_edges[se], dn_edges[ne], n_az)
                    for se, ne in zip([l0, c0, r0], [b1, b1, b0])], axis=1)
            m__a = np.concatenate([az_bin(de_edges[se], dn_edges[ne], n_az)
                    for se, ne in zip([l1, c1, r0], [m0, m0, m1])], axis=1)
            m__b = np.concatenate([az_bin(de_edges[se], dn_edges[ne], n_az)
                    for se, ne in zip([l1, c1, r0], [m1, m1, m0])], axis=1)
            t__a = np.concatenate([az_bin(de_edges[se], dn_edges[ne], n_az)
                    for se, ne in zip([l0, c0, r0], [t0, t0, t1])], axis=1)
            t__b = np.concatenate([az_bin(de_edges[se], dn_edges[ne], n_az)
                    for se, ne in zip([l1, c1, r1], [t1, t0, t0])], axis=1)
            ___a = np.concatenate([b__a, m__a, t__a], axis=0)
            ___b = np.concatenate([b__b, m__b, t__b], axis=0)
            # manually set middle pixel to full range
            ___a[n0_px, e0_px] = 0
            ___b[n0_px, e0_px] = n_az - 1
        else:  # n0 < n_edges[0]
            # tc case
            ___a = np.concatenate([az_bin(de_edges[se], dn_edges[ne], n_az)
                    for se, ne in zip([l0, c0, r0], [t0, t0, t1])], axis=1)
            ___b = np.concatenate([az_bin(de_edges[se], dn_edges[ne], n_az)
                    for se, ne in zip([l1, c1, r1], [t1, t0, t0])], axis=1)
    else:  # e0 > e_edges[-1]
        # l case
        l0, l1 = slice(0, -1), slice(1, None)
        if n0 > n_edges[-1]:
            # bl case
            ___a = az_bin(de_edges[l1], dn_edges[b0], n_az)
            ___b = az_bin(de_edges[l0], dn_edges[b1], n_az)
        elif n0 >= n_edges[0]:
            # l case
            b__a = az_bin(de_edges[l1], dn_edges[b0], n_az)
            b__b = az_bin(de_edges[l0], dn_edges[b1], n_az)
            m__a = az_bin(de_edges[l1], dn_edges[m0], n_az)
            m__b = az_bin(de_edges[l1], dn_edges[m1], n_az)
            t__a = az_bin(de_edges[l0], dn_edges[t0], n_az)
            t__b = az_bin(de_edges[l1], dn_edges[t1], n_az)
            ___a = np.concatenate([b__a, m__a, t__a], axis=0)
            ___b = np.concatenate([b__b, m__b, t__b], axis=0)
        else:  # n0 < n_edges[0]
            # tl case
            ___a = az_bin(de_edges[l0], dn_edges[t0], n_az)
            ___b = az_bin(de_edges[l1], dn_edges[t1], n_az)
            
    return ___a, ___b
    
def calc_rmin(e_edges, n_edges, e0, n0):
    '''Calculate the min r a pixel could contain, based on where the
    pixel edges are.'''
    # (0, 0) is bottom-left
    # Letters are axis0: (t=top, m=middle, b=bottom),
    #             axis1: (l=left, c=center, r=right)
    if n0 <= n_edges[0]:
        # t case
        t = slice(0, -1)
        dn = n_edges[t] - n0
    elif n0 < n_edges[-1]:
        # full case
        n0_px = np.searchsorted(n_edges, n0)
        b = slice(1, n0_px)
        t = slice(n0_px, -1)
        dn = np.concatenate([n_edges[b], np.array([n0]), n_edges[t]]) - n0
    else:  # n0 > n_edges[-1]:
        # b case
        b = slice(1, None)
        dn = n_edges[b] - n0
    
    if e0 <= e_edges[0]:
        # r case
        r = slice(0, -1)
        de = e_edges[r] - e0
    elif e0 < e_edges[-1]:
        e0_px = np.searchsorted(e_edges, e0)
        l = slice(1, e0_px)
        r = slice(e0_px, -1)
        de = np.concatenate([e_edges[l], np.array([e0]), e_edges[r]]) - e0
    else:  # e_edges[-1] < e0_px 
        # l case
        l = slice(1, None)
        de = e_edges[l] - e0
        
    return np.sqrt(dn[:, None]**2 + de[None, :]**2)

def pixel_delay_attenuation(radius_m, delta_freq):
    '''Calculate attenuation of high-delay reflected signals by the expected
    [tophat] bandpass width [Hz].'''
    tau_max = 1 / delta_freq  # effective width of a bandpass in delay
    radius_max = tau_max * C  # distance corresponding to maximum delay at speed of light
    attenuation = np.sinc(radius_m / radius_max)  # uses sinc as FT of tophat
    return attenuation

def pixel_coherence_angle_attenuation(radius_m, freq, nside):
    '''Calculate the expected attenuation arising from incoherence in different
    coherence patches within a HealPix pixel [with resolution defined by nside]
    at distance radius_m [m] scattering signals out of phase with each other
    at the given frequency [Hz].'''
    wavelen_m = C / freq
    omega_px = 4 * np.pi / (12 * nside**2)  # steradians
    area_m = omega_px * radius_m**2  # assumes small angles
    n_coherence_patches = area_m / wavelen_m**2  # number of incoherent scattering patches
    attenuation = 1 / np.sqrt(n_coherence_patches)  # assumes random incoherent scattering
    return attenuation

def horizon_angle_to_distance(angles, alt):
    '''Given an angle above the horizon (radians) and altitude (m) compute
    a visibility distance (m) accounting for earth curvature.'''
    th3 = np.arcsin(R_earth * np.sin(np.pi/2 + angles) / (R_earth + alt))
    return R_earth * (np.pi/2 - angles - th3)

def are_points_in_polygon(vertices, points):
    """Determine if multiple points are inside a polygon using ray-casting algorithm.

    Parameters:
    vertices (numpy.ndarray): A 2D array of shape (n, 2) representing the polygon vertices.
    points (numpy.ndarray): A 2D array of shape (m, 2) representing the points to be tested.

    Returns:
    numpy.ndarray: A boolean array of shape (m,) where each element indicates if the point is inside the polygon."""
    x_vertices, y_vertices = vertices
    x_points, y_points = points

    n = vertices.shape[1]
    inside = np.zeros(points.shape[1:], dtype=bool)

    # Vectorized ray-casting logic
    for i in range(n):
        j = (i - 1) % n
        xi, yi = x_vertices[i], y_vertices[i]
        xj, yj = x_vertices[j], y_vertices[j]
        # Check if the ray crosses the edge
        intersect = ((yi > y_points) != (yj > y_points)) & (
            x_points < (xj - xi) * (y_points - yi) / (yj - yi) + xi
        )

        inside ^= intersect  # Toggle inside state
    return inside


def mask_near_horizon(sky_mask, px_dist):
    """Generate mask of pixels within px_dist of edge in sky_mask.
    """
    m = (sky_mask > 0).astype(np.uint8) * 255 
    kernel = np.ones((3,3), np.uint8)
    inner_eroded = cv2.erode(m, kernel, iterations=1)
    edge = cv2.bitwise_xor(m, inner_eroded)
    inv_edge = cv2.bitwise_not(edge)
    dist = cv2.distanceTransform(inv_edge, cv2.DIST_L2, 5)
    near_horizon = (dist <= px_dist)
    return near_horizon, dist

def fill_psky_holes(psky, thr, fill_thresh, connectivity, px_dist):
    """Fill holes in probabilistically segmented sky.
    """
    sky = (psky >= thr)
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats((~sky).astype(np.uint8), connectivity=connectivity)
    for label in range(1, nlabels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < fill_thresh:
            sky[labels == label] = True
    horz, _ = mask_near_horizon(sky, px_dist)
    psky_filled = np.where(horz, psky, sky)
    return sky, psky_filled
