"""Joint parallax posterior and blocked Metropolis pilot for Marjum.

All computations use float64. Priors come from EXIF and Config, never the CV
solution. CV states provide numerical origins, initial values and fixed proposal
geometry only. The target samples absolute ENU altitudes, log focal lengths,
one shared antenna, every 3D landmark, a common GPS offset, and excess antenna
label scatter. The deterministic sampled skyline is an approximate likelihood;
this pilot does not claim a calibrated final uncertainty budget.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import time

import numpy as np
from eigsep_terrain import exif as terrain_exif
from scipy.optimize import least_squares
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse import lil_matrix

from marjum_bundle import Terrain, PRM_ORDER, project, rays


@dataclass(frozen=True)
class Config:
    tie_sigma_px: float = 4.
    antenna_label_sigma_px: float = 3.
    antenna_extra_prior_px: float = 10.
    horizon_sigma_angular_px: float = 20.
    terrain_sigma_m: float = 5.
    gps_independent_floor_m: float = 10.
    gps_common_sigma_m: float = 20.
    altitude_sigma_m: float = 30.
    heading_sigma_rad: float = .5
    elevation_sigma_rad: float = .5
    roll_sigma_rad: float = .2
    log_f_sigma: float = .25
    log_f_sigma_by_camera: dict | None = None
    student_df: float = 4.
    skyline_samples: int = 768


def focal_prior_widths(config, keys):
    """Resolve fixed log-f widths by camera, rejecting misspelled overrides."""
    overrides = config.log_f_sigma_by_camera or {}
    unknown = set(overrides)-set(keys)
    if unknown:
        raise ValueError(f'Unknown focal-prior cameras: {sorted(unknown)}')
    widths = np.array([overrides.get(k, config.log_f_sigma) for k in keys], float)
    if not np.isfinite(config.log_f_sigma) or config.log_f_sigma <= 0 or np.any(~np.isfinite(widths) | (widths <= 0)):
        raise ValueError('Focal-prior widths must be finite and positive')
    return widths


def student_residual(r, df, dimension=1):
    """Residual transform whose squared norm gives Student-t negative log L."""
    r = np.asarray(r, float)
    s = np.sum(r*r, axis=-1, keepdims=True) if dimension > 1 else r*r
    factor = np.sqrt((df+dimension)*np.log1p(s/df)/np.maximum(s,1e-30))
    factor = np.where(s < 1e-20, np.sqrt((df+dimension)/df), factor)
    return r*factor


def digest(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


class Posterior:
    def __init__(self, experiment='cv_initialization_v3', dem_file='marjum_dem_sw.npz', config=None):
        from eigsep_terrain.fitio import load_fit
        from eigsep_terrain.marjum_dem import MarjumDEM
        self.config = config or Config()
        self.experiment = Path(experiment)
        self.dem_file = dem_file
        self.terrain = Terrain(MarjumDEM(cache_file=dem_file))
        with np.load(self.experiment/'state_0.npz') as s:
            self.keys = list(s['keys'].astype(str))
            self.shapes = s['shapes'].copy()
            self.oc = s['obs_cam'].copy()
            self.op = s['obs_point'].copy()
            self.xy = s['obs_xy'].astype(float)
        self.nc = len(self.keys)
        self.focal_sigma = focal_prior_widths(self.config, self.keys)
        self.np = int(self.op.max())+1
        self.ng = self.nc*7+6
        self.by_camera = [np.flatnonzero(self.oc==i) for i in range(self.nc)]
        meta = json.loads(Path('meta.json').read_text())
        self.ai = np.array([i for i,k in enumerate(self.keys) if 'ant_px' in meta.get(k,{})])
        self.axy = np.array([meta[self.keys[i]]['ant_px'] for i in self.ai])
        self.horizon = []
        self.input_files = [Path(__file__), Path(terrain_exif.__file__), Path('marjum_bundle.py'), Path('meta.json'),
                            Path(dem_file), Path('marjum_2026_07_exif.npz')]
        for k in self.keys:
            file = Path(f'cv_features/sift_{k}.npz')
            with np.load(file) as f:
                self.horizon.append(f['horizon'].astype(float))
            self.input_files.append(file)
        with np.load('marjum_2026_07_exif.npz') as ex:
            idx = [list(ex['keys']).index(k) for k in self.keys]
            self.gps = np.c_[ex['e_gps'][idx],ex['n_gps'][idx]]
            self.gps_sigma = np.maximum(ex['h_err_m'][idx],self.config.gps_independent_floor_m)
            self.alt = ex['u_gps'][idx]
            self.heading = np.pi/2-np.deg2rad(ex['heading_deg'][idx])
            self.focal = terrain_exif.focal_length_pixels(ex['focal_35mm'][idx],
                                                          self.shapes[:,1], self.shapes[:,0])
        if not all(np.all(np.isfinite(v)) for v in [self.gps,self.gps_sigma,self.alt,self.heading,self.focal]):
            raise ValueError('Missing independent EXIF prior information')
        starts = []
        for trial in (0,1):
            state_file = self.experiment/f'state_{trial}.npz'
            fit_file = self.experiment/f'candidate_{trial}.npz'
            self.input_files.extend([state_file,fit_file])
            poses, ant, _ = load_fit(fit_file)
            cam = np.array([[poses[k][q] for q in PRM_ORDER] for k in self.keys])
            cam[:,6] = np.log(cam[:,6])
            cam[:,4] = self.heading + (cam[:,4]-self.heading+np.pi)%(2*np.pi)-np.pi
            with np.load(state_file) as s:
                for name, array in [('obs_cam',self.oc),('obs_point',self.op),('obs_xy',self.xy),('keys',np.array(self.keys))]:
                    if not np.array_equal(s[name],array):
                        raise ValueError(f'Candidate landmark identities disagree: {name}')
                points = s['points'].astype(float)
            starts.append(np.r_[cam.ravel(), ant, [0.,0.,8.], points.ravel()])
        # Pure affine coordinate change: the reference does not appear in priors.
        self.origin = starts[0]
        self.scale = np.r_[np.tile([20,20,20,.1,.1,.05,.1],self.nc),
                           [20,20,20,10,10,5],np.full(self.np*3,20.)]
        self.starts = [(s-self.origin)/self.scale for s in starts]
        self.sparsity = self._sparsity()

    def unpack(self,z):
        value = self.origin + np.asarray(z,dtype=np.float64)*self.scale
        cam = value[:7*self.nc].reshape(self.nc,7).copy()
        with np.errstate(over='ignore'):
            cam[:,6] = np.exp(cam[:,6])
        return cam, value[7*self.nc:7*self.nc+3], value[7*self.nc+3:7*self.nc+5], value[self.ng-1], value[self.ng:].reshape(self.np,3)

    def valid(self,z):
        if not np.all(np.isfinite(z)):
            return False
        cam,ant,bias,extra,points = self.unpack(z)
        t = self.terrain
        positions = np.vstack((cam[:,:3],ant,points))
        if extra < 0 or not np.all(np.isfinite(positions)):
            return False
        if np.any((positions[:,0] <= t.e[0]+2)|(positions[:,0] >= t.e[-1]-2)|
                  (positions[:,1] <= t.n[0]+2)|(positions[:,1] >= t.n[-1]-2)):
            return False
        # Finite physical support and unique orientation branch, independent of CV.
        if np.any((cam[:,3] <= 0)|(cam[:,3] >= np.pi)):
            return False
        if np.any(abs(cam[:,4]-self.heading) >= np.pi) or np.any(abs(cam[:,5]) >= np.pi):
            return False
        if np.any((cam[:,6] < 100)|(cam[:,6] > 50000)):
            return False
        if np.any(cam[:,2] <= t.height(cam[:,0],cam[:,1])+.1):
            return False
        if not (float(t.height(*ant[:2])) < ant[2] < 4000.):
            return False
        return True

    def point_residuals(self,z):
        cam,ant,bias,extra,points = self.unpack(z)
        pred = np.empty_like(self.xy)
        depth = np.empty(len(self.xy))
        for i,idx in enumerate(self.by_camera):
            pred[idx],depth[idx] = project(cam[i],self.shapes[i],points[self.op[idx]])
        c = self.config
        tie = student_residual((pred-self.xy)/c.tie_sigma_px,c.student_df,2)
        terrain = student_residual((points[:,2]-self.terrain.height(points[:,0],points[:,1]))/c.terrain_sigma_m,c.student_df)
        return tie,terrain,depth

    def point_logp(self,z):
        """One independent conditional log density per 3D landmark."""
        cam,ant,bias,extra,points = self.unpack(z)
        tie,terrain,depth = self.point_residuals(z)
        result = -.5*(np.bincount(self.op,weights=np.sum(tie*tie,axis=1),minlength=self.np)+terrain*terrain)
        invalid = np.bincount(self.op,weights=(depth <= .1),minlength=self.np)>0
        t = self.terrain
        invalid |= (points[:,0] <= t.e[0]+2)|(points[:,0] >= t.e[-1]-2)|\
                   (points[:,1] <= t.n[0]+2)|(points[:,1] >= t.n[-1]-2)
        result[invalid|~np.isfinite(result)] = -np.inf
        return result

    def global_residuals(self,z):
        cam,ant,bias,extra,points = self.unpack(z)
        c = self.config
        sigma = np.hypot(c.antenna_label_sigma_px,extra)
        ap = np.array([project(cam[i],self.shapes[i],ant)[0][0] for i in self.ai])
        horizon = []
        for i,xy in enumerate(self.horizon):
            d = rays(cam[i],self.shapes[i],xy)
            elevation = np.arctan2(d[:,2],np.hypot(d[:,0],d[:,1]))
            skyline = self.terrain.skyline(cam[i,:3],np.arctan2(d[:,1],d[:,0]),c.skyline_samples)
            # Fixed angular conversion derived from EXIF, not the sampled focal length.
            horizon.extend(student_residual((elevation-skyline)*self.focal[i]/c.horizon_sigma_angular_px,c.student_df))
        return np.r_[(ap-self.axy).ravel()/sigma,
                      np.full(len(self.ai),np.sqrt(4*np.log(sigma/c.antenna_label_sigma_px))),
                      horizon, ((cam[:,:2]+bias-self.gps)/self.gps_sigma[:,None]).ravel(),
                      (cam[:,2]-self.alt)/c.altitude_sigma_m,
                      (cam[:,3]-np.pi/2)/c.elevation_sigma_rad,
                      (cam[:,4]-self.heading)/c.heading_sigma_rad,
                      cam[:,5]/c.roll_sigma_rad,
                      np.log(cam[:,6]/self.focal)/self.focal_sigma,
                      bias/c.gps_common_sigma_m,extra/c.antenna_extra_prior_px]

    def residuals(self,z):
        tie,terrain,depth = self.point_residuals(z)
        return np.r_[tie.ravel(),terrain,self.global_residuals(z)]

    def logp(self,z):
        if not self.valid(z):
            return -np.inf
        cam,ant,*_ = self.unpack(z)
        if any(project(cam[i],self.shapes[i],ant)[1][0] <= .1 for i in self.ai):
            return -np.inf
        p = self.point_logp(z)
        g = self.global_residuals(z)
        value = p.sum()-.5*(g@g)
        return float(value) if np.isfinite(value) else -np.inf

    def _sparsity(self):
        rows = []
        camera = lambda i:list(range(i*7,i*7+7))
        point = lambda j:list(range(self.ng+j*3,self.ng+j*3+3))
        ant = list(range(7*self.nc,7*self.nc+3))
        bias = list(range(7*self.nc+3,7*self.nc+5))
        extra = [self.ng-1]
        for i,j in zip(self.oc,self.op):
            rows.extend([camera(i)+point(j)]*2)
        rows.extend(point(j) for j in range(self.np))
        for i in self.ai:
            rows.extend([camera(i)+ant+extra]*2)
        rows.extend([extra]*len(self.ai))
        for i,h in enumerate(self.horizon):
            rows.extend([camera(i)]*len(h))
        for i in range(self.nc):
            rows.extend([camera(i)+bias]*2)
        for _ in range(5):
            rows.extend(camera(i) for i in range(self.nc))
        rows.extend([bias,bias,extra])
        mat = lil_matrix((len(rows),len(self.origin)),dtype=int)
        for i,cols in enumerate(rows):
            mat[i,cols] = 1
        return mat.tocsr()

    def proposal_geometry(self,z):
        """Fixed Gaussian curvature and its exact block factorization.

        The curvature approximates the target only to construct a proposal.
        Every sampling acceptance ratio uses the full nonlinear posterior.
        """
        jac = approx_derivative(self.residuals,z,method='3-point',abs_step=1e-4,sparsity=self.sparsity)
        jg,jp = jac[:,:self.ng],jac[:,self.ng:]
        a = (jg.T@jg).toarray()
        cross = (jp.T@jg).toarray().reshape(self.np,3,self.ng)
        hpp = (jp.T@jp).tocsr()
        b = np.array([hpp[3*j:3*j+3,3*j:3*j+3].toarray() for j in range(self.np)])
        # Proposal-only ridge makes weak directions finite, without modifying logp.
        b += np.eye(3)[None]*1e-4
        conditional = np.linalg.inv(b)
        response = -conditional@cross
        schur = a+np.einsum('nig,nij->gj',cross,response)
        schur = .5*(schur+schur.T)+np.eye(self.ng)*1e-3
        eig,vec = np.linalg.eigh(schur)
        global_chol = vec@np.diag(1/np.sqrt(np.maximum(eig,1e-3)))
        return global_chol,np.linalg.cholesky(conditional),response


def metropolis_sweep(model,z,logp,geometry,rng,global_scale,point_scale):
    """Symmetric global move with landmark response, then independent point moves."""
    gl,pl,response = geometry
    step = global_scale*(gl@rng.normal(size=model.ng))
    proposed = z.copy()
    proposed[:model.ng] += step
    proposed[model.ng:] += (response@step).ravel()
    proposed_logp = model.logp(proposed)
    accepted_global = np.log(rng.random()) < proposed_logp-logp
    if accepted_global:
        z,logp = proposed,proposed_logp
    before = model.point_logp(z)
    proposed = z.copy()
    proposed[model.ng:] += (point_scale*np.einsum('nij,nj->ni',pl,rng.normal(size=(model.np,3)))).ravel()
    after = model.point_logp(proposed)
    accepted_points = np.log(rng.random(model.np)) < after-before
    z[model.ng:].reshape(-1,3)[accepted_points] = proposed[model.ng:].reshape(-1,3)[accepted_points]
    logp += np.sum(after[accepted_points]-before[accepted_points])
    return z,float(logp),bool(accepted_global),float(accepted_points.mean())


def run(output='mcmc_parallax_pilot_v2',tune=1000,draws=1000,chains=4,seed=20260910):
    import arviz as az
    from eigsep_terrain.fitio import save_fit
    out = Path(output)
    out.mkdir(exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Choose a fresh output directory; traces are never silently reused.')
    model = Posterior()
    shutil.copyfile(__file__,out/'sampler_source.py')
    hashes = {str(p):digest(p) for p in model.input_files}
    manifest = dict(config=asdict(model.config),input_sha256=hashes,seed=seed,tune=tune,draws=draws,chains=chains,
                    keys=model.keys,landmarks=model.np,observations=len(model.xy),
                    description='Approximate skyline posterior; all landmarks sampled, no dense photometric term.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    np.savez_compressed(out/'observations.npz',obs_cam=model.oc,obs_point=model.op,obs_xy=model.xy,
                        **{f'horizon_{k}':h for k,h in zip(model.keys,model.horizon)})
    anchors = []
    for i,start in enumerate(model.starts):
        print(f'Preparing posterior start {i}; logp={model.logp(start):.1f}',flush=True)
        # This is only preparation; acceptance below enforces exact hard support.
        result = least_squares(model.residuals,start,jac_sparsity=model.sparsity,
                               max_nfev=60,x_scale='jac',ftol=1e-5,diff_step=1e-4)
        anchor = result.x if np.isfinite(model.logp(result.x)) and model.logp(result.x)>model.logp(start) else start.copy()
        anchors.append(anchor)
        print(f'Prepared logp={model.logp(anchor):.1f}; computing fixed correlated proposals',flush=True)
    geometry = model.proposal_geometry(anchors[np.argmax([model.logp(z) for z in anchors])])
    np.savez_compressed(out/'proposal.npz',global_chol=geometry[0],point_chol=geometry[1],response=geometry[2],
                        origin=model.origin,scale=model.scale,anchors=np.array(anchors))
    posterior,stats,landmark_samples = [],[],[]
    clock = time.monotonic()
    for chain in range(chains):
        rng = np.random.default_rng(np.random.SeedSequence([seed,chain]))
        z = anchors[chain%2].copy()
        lp = model.logp(z)
        gs,ps = 2.38/np.sqrt(model.ng),2.38/np.sqrt(3)
        chain_geometry = geometry
        # Distinct initial states, obtained with valid MH transitions.
        for _ in range(3+chain):
            z,lp,_,_ = metropolis_sweep(model,z,lp,chain_geometry,rng,gs*.1,ps*.1)
        records,st,landmarks = [],[],[]
        window_g,window_p = [],[]
        for iteration in range(tune+draws):
            z,lp,ag,ap = metropolis_sweep(model,z,lp,chain_geometry,rng,gs,ps)
            window_g.append(ag);window_p.append(ap)
            if iteration<tune and (iteration+1)%10==0:
                # Adapt only during discarded warmup; fixed kernels during draws.
                rate = min(2.,5/np.sqrt((iteration+1)/10))
                gs *= np.exp(rate*(np.mean(window_g)-.234))
                ps *= np.exp(rate*(np.mean(window_p)-.35))
                window_g.clear();window_p.clear()
            if iteration<tune and (iteration+1)%250==0:
                # Conditional landmark geometry can change far from its mode.
                # Refresh only in discarded warmup; retained kernels stay fixed.
                chain_geometry = model.proposal_geometry(z)
                np.savez_compressed(out/f'proposal_chain_{chain}.npz',
                                    global_chol=chain_geometry[0],point_chol=chain_geometry[1],
                                    response=chain_geometry[2],reference_z=z,iteration=iteration+1)
            if iteration>=tune:
                physical = model.origin+z*model.scale
                records.append(physical[:model.ng].copy())
                st.append([lp,ag,ap,gs,ps])
                landmarks.append(physical[model.ng:].reshape(model.np,3).copy())
            if (iteration+1)%100==0:
                print(f'chain {chain+1}/{chains} {iteration+1}/{tune+draws}: logp={lp:.1f} global_scale={gs:.3g}, elapsed={time.monotonic()-clock:.0f}s',flush=True)
        posterior.append(np.array(records));stats.append(np.array(st));landmark_samples.append(np.array(landmarks))
        np.savez_compressed(out/f'chain_{chain}.npz',global_samples=posterior[-1],landmarks=landmark_samples[-1],stats=stats[-1],
                            final_z=z,rng_state=json.dumps(rng.bit_generator.state))
    values = np.array(posterior); sample_stats=np.array(stats)
    data = {}
    for i,key in enumerate(model.keys):
        for j,q in enumerate(PRM_ORDER):
            data[f'{key}_{q}'] = np.exp(values[:,:,7*i+j]) if q=='f' else values[:,:,7*i+j]
    for j,q in enumerate(['ant_e','ant_n','ant_u','gps_bias_e','gps_bias_n','antenna_extra_px']):
        data[q] = values[:,:,7*model.nc+j]
    data['landmark_enu'] = np.array(landmark_samples)
    trace=az.from_dict(posterior=data,sample_stats={k:sample_stats[:,:,j] for j,k in enumerate(['lp','accepted_global','accepted_landmark_fraction','global_scale','landmark_scale'])},
                       coords={'landmark':np.arange(model.np),'enu':['e','n','u']},dims={'landmark_enu':['landmark','enu']})
    trace.attrs['manifest_sha256']=digest(out/'manifest.json')
    trace.to_netcdf(out/'trace.nc')
    summary=az.summary(trace,var_names=[k for k in data if k!='landmark_enu'])
    summary.to_csv(out/'diagnostics.csv')
    ant_names=['ant_e','ant_n','ant_u']
    passed=bool((summary.loc[ant_names,'r_hat']<1.01).all() and (summary.loc[ant_names,['ess_bulk','ess_tail']]>400).all().all())
    status=dict(antenna_diagnostics_pass=passed,production_ready=False,
                acceptance_global=sample_stats[:,:,1].mean(axis=1).tolist(),
                acceptance_landmarks=sample_stats[:,:,2].mean(axis=1).tolist(),
                note='Pilot only. Diagnose all camera/landmark chains and model sensitivity before reporting uncertainty.')
    (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    # Representative sampled pose, not a mixture of independently averaged geometry.
    best_chain,best_draw=np.unravel_index(np.argmax(sample_stats[:,:,0]),sample_stats[:,:,0].shape)
    sample=values[best_chain,best_draw]
    poses={k:{q:float(np.exp(sample[7*i+j]) if q=='f' else sample[7*i+j]) for j,q in enumerate(PRM_ORDER)} for i,k in enumerate(model.keys)}
    save_fit(out/'fit_best_sample.npz',poses,sample[7*model.nc:7*model.nc+3])
    print(summary.loc[ant_names+['gps_bias_e','gps_bias_n','antenna_extra_px']].to_string(),flush=True)
    print(json.dumps(status,indent=2),flush=True)
    return trace


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='mcmc_parallax_pilot_v2')
    p.add_argument('--tune',type=int,default=1000)
    p.add_argument('--draws',type=int,default=1000)
    p.add_argument('--chains',type=int,default=4)
    p.add_argument('--seed',type=int,default=20260910)
    args=p.parse_args()
    if min(args.tune,args.draws,args.chains)<1:
        p.error('tune, draws and chains must be positive')
    run(**vars(args))
