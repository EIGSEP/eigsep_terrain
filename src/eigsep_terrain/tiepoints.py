'''Sparse OpenCV tie-point matching between two HorizonImages, and a
cross-check that projects a tie point through one image's fitted pose and
compares it to where it actually landed in the other image. Distant canyon
terrain plus two photos from (near-)the same position means parallax is
negligible, so a tie point's ray direction from one camera should predict
its pixel in the other camera's frame directly from each image's own
(e, n, u, th, ph, ti, f) pose -- no essential-matrix/translation recovery
needed. Agreement (or lack of it) is a direct check on whether two
independently-fit poses (each only constrained by its own horizon and
antenna pick) are mutually consistent.'''

import numpy as np
import cv2


def match_tie_points(img1, img2, nfeatures=4000, ratio=0.75, ransac_thresh=3.0):
    '''SIFT-match img1 and img2, restricted to non-sky pixels (sky_mask
    excludes clouds, which move and aren't valid static tie points), a
    ratio-test to keep only unambiguous matches, and a RANSAC homography
    to reject outliers (valid here because both images are far enough from
    the terrain that any camera translation between them acts like pure
    rotation, up to noise).

    Returns (pts1, pts2, inliers): pts1/pts2 are (N, 2) arrays of (col,
    row) pixel coordinates (OpenCV keypoint convention) for every ratio-
    test match, and inliers is a boolean mask of which of those survived
    RANSAC.'''
    def gray_masked(img):
        gray = cv2.cvtColor(img.img, cv2.COLOR_RGB2GRAY)
        mask = (~img.sky_mask).astype(np.uint8) * 255
        return gray, mask

    g1, m1 = gray_masked(img1)
    g2, m2 = gray_masked(img2)

    sift = cv2.SIFT_create(nfeatures=nfeatures)
    kp1, des1 = sift.detectAndCompute(g1, m1)
    kp2, des2 = sift.detectAndCompute(g2, m2)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw_matches if m.distance < ratio * n.distance]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    if len(good) < 4:
        return pts1, pts2, np.zeros(len(good), dtype=bool)

    _H, mask_h = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh)
    inliers = mask_h.ravel().astype(bool)
    return pts1, pts2, inliers


def project_via_pose(img_src, row_col_src, img_dst):
    '''Take a pixel (row, col) in img_src, compute its ray direction under
    img_src's *current* pose (img_src.prms), and predict the (row, col) it
    would land at in img_dst under img_dst's current pose -- assuming
    negligible parallax (valid for distant terrain and two nearby camera
    positions). This is the exact algebraic inverse of HorizonImage.get_rays
    (rays = Rz(ph) @ Ry(th) @ Rz(ti) @ v0), so projecting a point through
    its own image's pose reproduces the original pixel to numerical
    precision -- a useful self-consistency check.'''
    ray = img_src.get_rays(pixels=np.array(row_col_src))
    ray = ray / np.linalg.norm(ray)
    return _project_direction(ray, img_dst)


def project_world_point(point_enu, img):
    '''Project a world ENU point (3,) into img's pixel coordinates (row,
    col) using img's *current* pose -- unlike project_via_pose, this uses
    img's own camera position too (point_enu - camera position), not just
    its orientation, so it's exact regardless of distance/parallax (there's
    no other image's ray direction being reused here). Returns (row, col);
    values outside [0, npix) mean the point isn't actually in frame.'''
    cam = np.array([img.prms['e'], img.prms['n'], img.prms['u']])
    d = np.asarray(point_enu, dtype=float) - cam
    d = d / np.linalg.norm(d)
    return _project_direction(d, img)


def _project_direction(ray, img_dst):
    '''Shared math for project_via_pose/project_world_point: given a world
    direction vector `ray` (3,) or (3, N), predict the (row, col) pixel(s)
    it projects to under img_dst's current pose. Algebraic inverse of
    HorizonImage.get_rays.'''
    a0, b0, c0 = ray
    ti, th, ph = img_dst.prms['ti'], img_dst.prms['th'], img_dst.prms['ph']
    cph, sph = np.cos(ph), np.sin(ph)
    a2 = cph * a0 + sph * b0
    b2 = -sph * a0 + cph * b0
    c2 = c0
    cth, sth = np.cos(th), np.sin(th)
    a1 = cth * a2 - sth * c2
    c1 = sth * a2 + cth * c2
    b1 = b2
    cti, sti = np.cos(ti), np.sin(ti)
    a_body = cti * a1 + sti * b1
    b_body = -sti * a1 + cti * b1
    c_body = c1

    f = img_dst.prms['f']
    row_pred = img_dst.npix_y // 2 - f * a_body / c_body
    col_pred = img_dst.npix_x // 2 - f * b_body / c_body
    return row_pred, col_pred


def triangulate_rays(origins, directions, weights=None):
    '''Closed-form weighted least-squares point closest to a set of 3D
    rays (each origins[i] + t*directions[i], t>=0 implicitly assumed but
    not enforced). Minimizes sum_i w_i * |p - origins[i]|_perp^2, the
    squared perpendicular distance from p to each line -- the standard
    "point closest to many lines" problem, solved via
    sum_i w_i (I - d_i d_i^T) p = sum_i w_i (I - d_i d_i^T) o_i.

    origins, directions: (N, 3) arrays (directions need not be
    pre-normalized). weights: (N,) array, defaults to all-ones (e.g. use
    each image's antenna-box-size-derived sigma to weight by 1/sigma**2
    for a proper weighted fit).

    Returns the (3,) point p. With N=2 this reproduces the two-ray
    closest-approach *midpoint* only in the limit of equal weights and
    small gap; for N=2 with a real gap, prefer computing both closest-
    approach points directly if the gap itself is diagnostic (see Phase 4).'''
    origins = np.asarray(origins, dtype=float)
    directions = np.asarray(directions, dtype=float)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    if weights is None:
        weights = np.ones(len(origins))
    A = np.zeros((3, 3))
    b = np.zeros(3)
    I = np.eye(3)
    for o, d, w in zip(origins, directions, weights):
        M = w * (I - np.outer(d, d))
        A += M
        b += M @ o
    return np.linalg.solve(A, b)


def refine_pose_from_ties(img_moving, img_fixed, pts_moving_xy, pts_fixed_xy,
                           free_keys=('e', 'n', 'u', 'th', 'ph', 'ti'),
                           th_range_deg=(-45, 21, 2.5), ph_range_deg=(-45, 46, 2.5)):
    '''Adjust img_moving's pose (img_fixed's held fixed) to minimize the
    tie-point reprojection residual against img_fixed.

    Stage 1: a coarse grid search over elevation/azimuth offsets, since (as
    in HorizonImage.refine_pose) the reprojection-residual objective is
    prone to bad local optima when the needed correction is large -- a
    Nelder-Mead started cold got stuck tens of degrees away from a good fit
    in practice. Stage 2: a Nelder-Mead polish of free_keys from the best
    grid point.

    This is a fast, deterministic warm start, not the final fit: a pose
    whose tie-point residual is hundreds of pixels (see
    cross_check_residuals) is likely too far from consistent for an MCMC
    sampler carrying a tie-point likelihood term (PositionSolver.
    add_tie_points) to find in reasonable time, so run this first to get
    close, then let the joint MCMC (horizon + antenna + tie-point
    likelihoods together) do the real, properly-balanced fit. Updates
    img_moving.prms in place and returns (prms_dict, OptimizeResult).'''
    from scipy.optimize import minimize
    from .img import PRM_ORDER

    base = dict(img_moving.prms)
    rows_m, cols_m = pts_moving_xy[:, 1], pts_moving_xy[:, 0]

    def mean_sq_resid(prms):
        img_moving.set_prms([prms[k] for k in PRM_ORDER])
        pred_row, pred_col = project_via_pose(img_moving, (rows_m, cols_m), img_fixed)
        dr = pred_row - pts_fixed_xy[:, 1]
        dc = pred_col - pts_fixed_xy[:, 0]
        return np.mean(dr**2 + dc**2)

    best = None
    for dth in np.deg2rad(np.arange(*th_range_deg)):
        for dph in np.deg2rad(np.arange(*ph_range_deg)):
            prms = dict(base)
            prms['th'] = base['th'] + dth
            prms['ph'] = base['ph'] + dph
            resid = mean_sq_resid(prms)
            if best is None or resid < best[0]:
                best = (resid, dth, dph)
    d1 = dict(base)
    d1['th'] = base['th'] + best[1]
    d1['ph'] = base['ph'] + best[2]

    def objective(x):
        prms = dict(d1)
        for k, v in zip(free_keys, x):
            prms[k] = v
        return mean_sq_resid(prms)

    x0 = np.array([d1[k] for k in free_keys], dtype=np.float64)
    default_steps = dict(e=5.0, n=5.0, u=5.0, th=np.deg2rad(1.0),
                          ph=np.deg2rad(1.0), ti=np.deg2rad(0.5),
                          f=base['f'] * 0.05)
    steps = np.array([default_steps[k] for k in free_keys])
    initial_simplex = np.array(
        [x0] + [x0 + steps[i] * np.eye(len(x0))[i] for i in range(len(x0))]
    )
    res = minimize(objective, x0, method='Nelder-Mead',
                    options=dict(xatol=1e-5, fatol=1e-3, maxiter=2000,
                                 maxfev=4000, initial_simplex=initial_simplex))
    prms_ref = dict(d1)
    for k, v in zip(free_keys, res.x):
        prms_ref[k] = v
    img_moving.set_prms([prms_ref[k] for k in PRM_ORDER])
    return prms_ref, res


def cross_check_residuals(img1, img2, pts1, pts2):
    '''For each matched point pair (pts1[i] in img1, pts2[i] in img2, both
    (col, row) OpenCV convention), project pts1[i] through img1's pose into
    img2's frame via project_via_pose and compare to the actual pts2[i].
    Returns (pred_pts2, residual_px): pred_pts2 is (N, 2) in (col, row)
    order to match pts2, and residual_px is the per-point Euclidean pixel
    distance between prediction and match.'''
    pred = []
    for (col1, row1) in pts1:
        row_pred, col_pred = project_via_pose(img1, (row1, col1), img2)
        pred.append((col_pred, row_pred))
    pred = np.array(pred)
    residual = np.linalg.norm(pred - pts2, axis=1)
    return pred, residual
