"""Rotation laws and NumPyro model definitions."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from tinygp import kernels
import sys
sys.path.append("/mnt/ogawa/work/jaxspin")
from jaxspin import SpinModel

jax.config.update("jax_enable_x64", True)

R_SUN_KM = 695700.0
DAY_S = 86400.0

# The GP is defined in log10(age/Gyr), with enough knots to describe smooth
# structure without constructing a covariance matrix over every isochrone
# posterior sample.
GP_LOGAGE_KNOTS = jnp.linspace(-1.0, jnp.log10(13.8), 12)
GP_JITTER = 1.0e-6

JAXSPIN_DATA_DIR = "/mnt/ogawa/work/jaxspin/jaxspin/data"
jaxspin_model = SpinModel(data_dir=JAXSPIN_DATA_DIR)

class RotEvolModelMixin:
    def model(
        self,
        predict_age_dist=True,
        predict_P_rot=False,
        rotation_law="power",
        break_age_bounds=(0.1, 13.8),
        ro_wmb_factor_bounds=(0.1, 2.0),
        jaxspin_feh_bounds=(-0.5, 0.5),
    ):
        """Rotation law using one shared logage distribution for all stars.

        When ``predict_P_rot`` is true, the law predicts rotation period
        and the representative stellar radii convert it to equatorial
        velocity for the observed ``vsini`` likelihood.
        """
        if rotation_law == "jaxspin" and not predict_P_rot:
            raise ValueError("rotation_law=jaxspin requires predict_P_rot=True")
        slopes, b, break_age = _sample_rotation_parameters(
            rotation_law,
            predict_P_rot=predict_P_rot,
            break_age_bounds=break_age_bounds,
            ro_wmb_factor_bounds=ro_wmb_factor_bounds,
            jaxspin_feh_bounds=jaxspin_feh_bounds,
        )
        if predict_P_rot:
            radius = jnp.asarray(self.radius)
        if predict_age_dist:
            lognorm_age_mu = numpyro.sample(
                "lognorm_age_mu",
                dist.Uniform(8, 10),
            )
            loglognorm_age_sigma = numpyro.sample(
                "loglognorm_age_sigma",
                dist.Uniform(-2, 0),
            )
            lognorm_age_sigma = numpyro.deterministic(
                "lognorm_age_sigma", 10**loglognorm_age_sigma
            )
        else:
            lognorm_age_mu = self.logage_prior_loc
            lognorm_age_sigma = self.logage_prior_scale
        with numpyro.plate("stars", self.N):
            cosi = numpyro.sample("cosi", dist.Uniform(0.0, 1.0))
            logage = numpyro.sample(
                "logage",
                dist.Normal(
                    lognorm_age_mu,
                    lognorm_age_sigma,
                ),
            )
            age = numpyro.deterministic("age", 10**(logage - 9))
            # age = numpyro.sample("age", dist.Uniform(0.01, 13.8))
            rotation_relation = evaluate_rotation_law(
                age, rotation_law, slopes, b, break_age,
                mass=self.mass, feh=self.feh,
            )
            if predict_P_rot:
                rotation_period = numpyro.deterministic(
                    "rotation_period", rotation_relation
                )
                equatorial_velocity = (
                    2.0 * jnp.pi * radius * R_SUN_KM
                    / (rotation_period * DAY_S)
                )
            else:
                equatorial_velocity = rotation_relation
            v = numpyro.deterministic("v", equatorial_velocity)
            vsini = numpyro.deterministic(
                "vsini", v * jnp.sqrt(1.0 - cosi**2)
            )
            numpyro.sample(
                "obs1",
                dist.Normal(vsini, jnp.asarray(self.vsini_err)),
                obs=jnp.asarray(self.vsini),
            )
            numpyro.sample(
                "obs2",
                dist.Normal(age, jnp.asarray(self.age_err)),
                obs=jnp.asarray(self.age),
            )

    def importance_model(
        self,
        isochrone_posterior,
        predict_age_dist=True,
        predict_P_rot=False,
        rotation_law="power",
        break_age_bounds=(0.1, 13.8),
        ro_wmb_factor_bounds=(0.1, 2.0),
        jaxspin_feh_bounds=(-0.5, 0.5),
    ):
        """Rotation law marginalized over each stars isochrone posterior."""
        if rotation_law == "jaxspin" and not predict_P_rot:
            raise ValueError("rotation_law=jaxspin requires predict_P_rot=True")
        slopes, b, break_age = _sample_rotation_parameters(
            rotation_law,
            predict_P_rot=predict_P_rot,
            break_age_bounds=break_age_bounds,
            ro_wmb_factor_bounds=ro_wmb_factor_bounds,
            jaxspin_feh_bounds=jaxspin_feh_bounds,
        )
        if predict_age_dist:
            lognorm_age_mu = numpyro.sample(
                "lognorm_age_mu",
                dist.Uniform(8.0, 10.0),
            )
            loglognorm_age_sigma = numpyro.sample(
                "loglognorm_age_sigma",
                dist.Uniform(-2.0, 0.0),
            )
            # lognorm_age_sigma = numpyro.sample(
            #     "lognorm_age_sigma",
            #     dist.Uniform(0.01, 1.0),
            # )
            lognorm_age_sigma = numpyro.deterministic("lognorm_age_sigma", 10**loglognorm_age_sigma)
        age_samples = jnp.asarray(isochrone_posterior.age)  # (N, K)
        logage_samples = jnp.log10(age_samples) + 9.0

        if predict_P_rot:
            radius_samples = jnp.asarray(isochrone_posterior.radius)  # (N, K)

        with numpyro.plate("stars", self.N):
            # (N,)
            cosi = numpyro.sample(
                "cosi",
                dist.Uniform(0.0, 1.0),
            )
            sini = jnp.sqrt(1.0 - cosi[:, None]**2)

            # (N, 1)
            vsini_obs = jnp.asarray(self.vsini)[:, None]
            vsini_err = jnp.asarray(self.vsini_err)[:, None]
            # (N, K)
            if not predict_P_rot:
                loglike_vsini = dist.Normal(
                    evaluate_rotation_law(
                        age_samples, rotation_law, slopes, b, break_age,
                        mass=self.mass, feh=self.feh,
                    ) * sini,
                    vsini_err,
                ).log_prob(vsini_obs)
            else:
                rotation_period = evaluate_rotation_law(
                    age_samples, rotation_law, slopes, b, break_age,
                    mass=self.mass, feh=self.feh,
                )
                loglike_vsini = dist.Normal(
                    (
                        2.0 * jnp.pi * radius_samples * R_SUN_KM
                        / (rotation_period * DAY_S)
                    ) * sini,
                    vsini_err,
                ).log_prob(vsini_obs)
            # (N, K)
            if predict_age_dist:
                log_population_age = dist.Normal(
                    lognorm_age_mu,
                    lognorm_age_sigma,
                ).log_prob(logage_samples) - jnp.log(age_samples) # logage->age Jacobian
                logweight = (
                    log_population_age - jnp.log(pi0_linear(age_samples))
                )
            else:
                logweight = jnp.zeros(logage_samples.shape)

            # (N, K)
            log_terms = loglike_vsini + logweight
            # sample方向だけ周辺化 → (N,)
            loglike_per_star = (
                jax.scipy.special.logsumexp(log_terms, axis=1)
                - jnp.log(age_samples.shape[1])
            )
            numpyro.factor("loglikelihood",loglike_per_star,)

    def isochrone_fit_model(
            self,
            predict_age_dist=True,
            predict_P_rot=False,
            rotation_law="power",
            break_age_bounds=(0.1, 13.8),
            ro_wmb_factor_bounds=(0.1, 2.0),
            jaxspin_feh_bounds=(-0.5, 0.5),
            dist_scale=1.35,
        ):
            """Rotation law using one shared logage distribution for all stars.
    
            When ``predict_P_rot`` is true, the law predicts rotation period
            and the representative stellar radii convert it to equatorial
            velocity for the observed ``vsini`` likelihood.
            """
            required_observations = (
                "teff_mean", "teff_err", "feh_mean", "feh_err",
                "gmag3_mean", "gmag3_err", "parallax_mean", "parallax_err",
            )
            missing = [
                name for name in required_observations
                if getattr(self, name, None) is None
            ]
            if missing:
                raise ValueError(
                    "isochrone_fit_model requires observables loaded by "
                    f"RotEvol.from_analysis_results(); missing {missing}"
                )

            from jaxstar.mistfit.mistfit import (
                MistGridIso,
                check_mistgrid_path,
                smbound,
            )

            path = check_mistgrid_path()
            mg = MistGridIso(path)
            obskeys =["teff", "feh", "gmag3", "parallax"] 
            outkeys = ['kmag', 'teff', 'logg', 'mass', 'radius', 'feh_photosphere', 'star_mass',
                               'dmdeep', 'mmin', 'mmax', 'bpmag2', 'rpmag2',"gmag3"]
            mg.set_keys(outkeys)

            if rotation_law == "jaxspin" and not predict_P_rot:
                raise ValueError("rotation_law=jaxspin requires predict_P_rot=True")
            slopes, b, break_age = _sample_rotation_parameters(
                rotation_law,
                predict_P_rot=predict_P_rot,
                break_age_bounds=break_age_bounds,
                ro_wmb_factor_bounds=ro_wmb_factor_bounds,
                jaxspin_feh_bounds=jaxspin_feh_bounds,
            )
            if predict_P_rot:
                radius = jnp.asarray(self.radius)
            if predict_age_dist:
                lognorm_age_mu = numpyro.sample(
                    "lognorm_age_mu",
                    dist.Uniform(8, 10),
                )
                loglognorm_age_sigma = numpyro.sample(
                    "loglognorm_age_sigma",
                    dist.Uniform(-2, 0),
                )
                lognorm_age_sigma = numpyro.deterministic(
                    "lognorm_age_sigma", 10**loglognorm_age_sigma
                )
            else:
                lognorm_age_mu = self.logage_prior_loc
                lognorm_age_sigma = self.logage_prior_scale
            with numpyro.plate("stars", self.N):
                cosi = numpyro.sample("cosi", dist.Uniform(0.0, 1.0))
                logage = numpyro.sample(
                    "logage",
                    dist.Normal(
                        lognorm_age_mu,
                        lognorm_age_sigma,
                    ),
                )
                age = numpyro.deterministic("age", 10**(logage - 9))
                # age = numpyro.sample("age", dist.Uniform(0.01, 13.8))
                feh_init = numpyro.sample("feh_init", dist.Uniform(-1.0, 0.5))
                eep = numpyro.sample("eep", dist.Uniform(0, 600))
                distance = numpyro.sample("distance", dist.Gamma(
                            3, rate=1./dist_scale))  # BJ18, kpc
                parallax = numpyro.deterministic("parallax", 1. / distance)  # mas

                grid_values = mg.values(logage, feh_init, eep)
                params = {}
                grid_valid = jnp.ones_like(logage, dtype=bool)
                for key, value in zip(outkeys, grid_values):
                    if 'mag' in key:
                        value = value - 5 * jnp.log10(parallax) + 10
                    value_is_finite = jnp.isfinite(value)
                    grid_valid = grid_valid & value_is_finite
                    params[key] = jnp.where(value_is_finite, value, 0.0)
                    numpyro.deterministic(key, params[key])
                invalid_log_prob = jnp.asarray(-1.0e30, dtype=logage.dtype)
                numpyro.factor(
                    "mist_grid_support",
                    jnp.where(grid_valid, 0.0, invalid_log_prob),
                )
                # "feh" is set to be photospheric value
                params["feh"] = numpyro.deterministic("feh", params["feh_photosphere"])
                radius = params["radius"]
                mass = params["star_mass"]
                feh = params["feh"]

                rotation_relation = evaluate_rotation_law(
                    age, rotation_law, slopes, b, break_age,
                    mass=mass, feh=feh,
                )
                if predict_P_rot:
                    rotation_period = numpyro.deterministic(
                        "rotation_period", rotation_relation
                    )
                    equatorial_velocity = (
                        2.0 * jnp.pi * radius * R_SUN_KM
                        / (rotation_period * DAY_S)
                    )
                else:
                    equatorial_velocity = rotation_relation
                v = numpyro.deterministic("v", equatorial_velocity)
                vsini = numpyro.deterministic(
                    "vsini", v * jnp.sqrt(1.0 - cosi**2)
                )
                numpyro.sample(
                    "obs1",
                    dist.Normal(vsini, jnp.asarray(self.vsini_err)),
                    obs=jnp.asarray(self.vsini),
                )

                obsvals = jnp.stack(
                    [
                        jnp.asarray(self.teff_mean),
                        jnp.asarray(self.feh_mean),
                        jnp.asarray(self.gmag3_mean),
                        jnp.asarray(self.parallax_mean),
                    ],
                    axis=-1,
                )
                obserrs = jnp.stack(
                    [
                        jnp.asarray(self.teff_err),
                        jnp.asarray(self.feh_err),
                        jnp.asarray(self.gmag3_err),
                        jnp.asarray(self.parallax_err),
                    ],
                    axis=-1,
                )
                params['parallax'] = parallax
                obsparams = jnp.stack([params[key] for key in obskeys], axis=-1)
                numpyro.sample(
                    "obs2",
                    dist.Normal(obsparams, obserrs).to_event(1),
                    obs=obsvals,
                )
                # mass prior
                logjac = jnp.log(params['dmdeep'])
                logjac += smbound(params['mass'], 0.1, 2.5)
                logjac = jnp.where(jnp.isfinite(logjac), logjac, invalid_log_prob)
                numpyro.factor("logjac", logjac)


def v_power(age, a, b, vf=0.0):
    """Power law with an additive asymptotic rotation floor.

    ``b`` is the power-law contribution at 4.6 Gyr and ``vf`` has the same
    units as the returned velocity or period.
    """
    return vf + b * (jnp.asarray(age) / 4.6) ** a

def broken_power_law(age, a1, a2, b, break_age=4.6):
    """Continuous broken power law with ``b`` defined at ``break_age``."""
    scaled_age = jnp.asarray(age) / break_age
    return b * jnp.where(
        scaled_age <= 1.0,
        scaled_age**a1,
        scaled_age**a2,
    )

def gp_rotation_law(age, gp_latent, gp_amplitude, gp_scale, b):
    """Evaluate a positive Matérn-3/2 GP rotation law.

    Ages are in Gyr. The GP coordinate is log10(age), ``gp_latent`` contains
    independent standard-normal values at ``GP_LOGAGE_KNOTS``, and
    ``gp_scale`` is measured in dex. The curve is centered so that its value
    at the solar age (4.6 Gyr) is exactly ``b``.
    """
    age = jnp.asarray(age)
    gp_latent = jnp.asarray(gp_latent)

    # Plotting passes a leading posterior-draw axis. Vectorize those draws
    # while keeping all age dimensions as the output grid.
    if gp_latent.ndim > 1:
        if gp_latent.shape[-2] == 1:
            gp_latent = jnp.squeeze(gp_latent, axis=-2)
        draw_shape = gp_latent.shape[:-1]
        amplitude = jnp.broadcast_to(jnp.squeeze(gp_amplitude), draw_shape)
        scale = jnp.broadcast_to(jnp.squeeze(gp_scale), draw_shape)
        normalization = jnp.broadcast_to(jnp.squeeze(b), draw_shape)
        evaluation_age = age[0] if age.ndim > 1 and age.shape[0] == 1 else age
        flat_latent = gp_latent.reshape((-1, gp_latent.shape[-1]))
        values = jax.vmap(gp_rotation_law, in_axes=(None, 0, 0, 0, 0))(
            evaluation_age,
            flat_latent,
            amplitude.reshape(-1),
            scale.reshape(-1),
            normalization.reshape(-1),
        )
        return values.reshape(draw_shape + evaluation_age.shape)

    logage = jnp.log10(age)
    kernel = gp_amplitude**2 * kernels.Matern32(scale=gp_scale)

    covariance = jax.vmap(
        lambda x: jax.vmap(lambda y: kernel.evaluate(x, y))(GP_LOGAGE_KNOTS)
    )(GP_LOGAGE_KNOTS)
    covariance = covariance + GP_JITTER * jnp.eye(GP_LOGAGE_KNOTS.size)
    knot_values = jnp.linalg.cholesky(covariance) @ gp_latent
    alpha = jax.scipy.linalg.solve(covariance, knot_values, assume_a="pos")

    def interpolate(x):
        cross_covariance = jax.vmap(lambda knot: kernel.evaluate(x, knot))(
            GP_LOGAGE_KNOTS
        )
        return cross_covariance @ alpha

    flat_prediction = jax.vmap(interpolate)(logage.reshape(-1))
    solar_prediction = interpolate(jnp.log10(4.6))
    log_relative_rotation = (
        flat_prediction.reshape(age.shape) - solar_prediction
    )
    return b * jnp.exp(log_relative_rotation)

def jaxspin_rotation_law(
    age,
    mass,
    feh,
    Ro_wmb_factor,
    P0_day=3.0,
    age_start=1.0e7,
    age_end=13.8e9,
    n_age=10001,
    P_sun=26.51,
    tau_sun=16.525,
    Ro_sat=0.1,
    T_sun_e30=None,
    feh_finite_difference_step=1.0e-3,
    ro_wmb_finite_difference_step=3.0e-2,
):
    """Evaluate a differentiable JAXSpin period track at ages in Gyr.

    ``mass`` is a fixed scalar. ``feh`` and ``Ro_wmb_factor`` may be NumPyro
    parameters. Stellar tracks are calculated by :meth:`SpinModel.compute`;
    finite differences provide derivatives in those two parameters, while
    interpolation onto ``age`` remains JAX differentiable.
    """
    mass = float(mass)
    output_spec = jax.ShapeDtypeStruct((n_age,), jnp.float64)
    age_grid = jnp.linspace(float(age_start), float(age_end), n_age)

    def compute_track(feh_value, ro_wmb_factor_value):
        result = jaxspin_model.compute(
            mass=mass,
            feh=float(np.asarray(feh_value)),
            P0_day=P0_day,
            age_start=age_start,
            age_end=age_end,
            n_age=n_age,
            P_sun=P_sun,
            tau_sun=tau_sun,
            Ro_sat=Ro_sat,
            Ro_wmb_factor=float(np.asarray(ro_wmb_factor_value)),
            T_sun_e30=T_sun_e30,
        )
        return np.asarray(result["Prot"], dtype=np.float64)

    @jax.custom_jvp
    def period_track(feh_value, ro_wmb_factor_value):
        return jax.pure_callback(
            compute_track,
            output_spec,
            feh_value,
            ro_wmb_factor_value,
        )

    @period_track.defjvp
    def period_track_jvp(primals, tangents):
        feh_value, ro_wmb_factor_value = primals
        feh_tangent, ro_wmb_factor_tangent = tangents
        feh_step = feh_finite_difference_step
        ro_wmb_step = ro_wmb_finite_difference_step
        track = period_track(feh_value, ro_wmb_factor_value)
        derivative_feh = (
            period_track(feh_value + feh_step, ro_wmb_factor_value)
            - period_track(feh_value - feh_step, ro_wmb_factor_value)
        ) / (2.0 * feh_step)
        derivative_ro_wmb = (
            period_track(feh_value, ro_wmb_factor_value + ro_wmb_step)
            - period_track(feh_value, ro_wmb_factor_value - ro_wmb_step)
        ) / (2.0 * ro_wmb_step)
        tangent = (
            derivative_feh * feh_tangent
            + derivative_ro_wmb * ro_wmb_factor_tangent
        )
        return track, tangent

    track = period_track(jnp.asarray(feh), jnp.asarray(Ro_wmb_factor))
    age_year = jnp.asarray(age) * 1.0e9
    return jnp.interp(age_year, age_grid, track)

def normalize_rotation_law(rotation_law):
    """Return the canonical name of a supported rotation law."""
    aliases = {
        "power": "power",
        "v_power": "power",
        "broken": "broken_power",
        "broken_power": "broken_power",
        "broken_power_law": "broken_power",
        "gp": "gp",
        "jaxspin": "jaxspin",
    }
    try:
        return aliases[rotation_law]
    except (KeyError, TypeError):
        raise ValueError(
            "rotation_law must be one of: power, broken_power, gp, jaxspin"
        ) from None

def rotation_parameter_names(rotation_law):
    """Names of the physical parameters for a rotation law."""
    rotation_law = normalize_rotation_law(rotation_law)
    if rotation_law == "power":
        return ("a", "b", "vf")
    if rotation_law == "broken_power":
        return ("a1", "a2", "b", "break_age")
    if rotation_law == "gp":
        return ("gp_latent", "gp_amplitude", "gp_scale", "b")
    if rotation_law == "jaxspin":
        return ("Ro_wmb_factor", "feh")
    raise NotImplementedError(
        f"rotation_law={rotation_law} is reserved for a future implementation"
    )

def rotation_slope_names(rotation_law):
    """Names of slope parameters for a rotation law."""
    rotation_law = normalize_rotation_law(rotation_law)
    if rotation_law == "power":
        return ("a", "vf")
    if rotation_law == "broken_power":
        return ("a1", "a2")
    if rotation_law == "gp":
        return ("gp_latent", "gp_amplitude", "gp_scale")
    if rotation_law == "jaxspin":
        return ("Ro_wmb_factor", "feh")
    raise NotImplementedError(
        f"rotation_law={rotation_law} is reserved for a future implementation"
    )


def evaluate_rotation_law(
    age, rotation_law, slopes, b=None, break_age=4.6, *, mass=None, feh=None
):
    """Evaluate a selected rotation law from its slope tuple and scale."""
    rotation_law = normalize_rotation_law(rotation_law)
    if rotation_law == "power":
        vf = slopes[1] if len(slopes) > 1 else 0.0
        return v_power(age, slopes[0], b, vf)
    if rotation_law == "broken_power":
        return broken_power_law(age, slopes[0], slopes[1], b, break_age)
    if rotation_law == "gp":
        return gp_rotation_law(age, *slopes, b)
    if mass is None:
        raise ValueError("mass is required for rotation_law=jaxspin")
    fixed_mass = float(np.median(np.asarray(mass, dtype=float)))
    ro_wmb_factor, inferred_feh = slopes
    return jaxspin_rotation_law(
        age, fixed_mass, inferred_feh, ro_wmb_factor
    )

def pi0_linear(age):
    """Unnormalized isochrone interim prior: uniform in linear age."""
    age = jnp.asarray(age)
    return jnp.ones_like(age)

def loglike_vsini_single(vsini_obs, vsini_err, v, cosi):
    """Gaussian log likelihood over each star's isochrone-sample axis."""
    vsini_model = v * jnp.sqrt(1.0 - cosi[:, None] ** 2)
    return -0.5 * (
        (vsini_obs[:, None] - vsini_model) ** 2 / vsini_err[:, None] ** 2
    )

def _ordered_bounds(bounds, name):
    if len(bounds) != 2:
        raise ValueError(f"{name} must contain (lower, upper)")
    lower, upper = map(float, bounds)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError(f"{name} must contain finite ordered bounds")
    return lower, upper


def _ordered_positive_bounds(bounds, name):
    lower, upper = _ordered_bounds(bounds, name)
    if lower <= 0.0:
        raise ValueError(f"{name} lower bound must be positive")
    return lower, upper


def _sample_rotation_parameters(
    rotation_law,
    predict_P_rot=False,
    break_age_bounds=(3, 6),
    ro_wmb_factor_bounds=(0.1, 2.0),
    jaxspin_feh_bounds=(-0.5, 0.5),
):
    """Sample parameters for the selected velocity or period relation."""
    rotation_law = normalize_rotation_law(rotation_law)
    if rotation_law == "jaxspin":
        if not predict_P_rot:
            raise ValueError("rotation_law=jaxspin requires predict_P_rot=True")
        ro_lower, ro_upper = _ordered_positive_bounds(
            ro_wmb_factor_bounds, "ro_wmb_factor_bounds"
        )
        feh_lower, feh_upper = _ordered_bounds(
            jaxspin_feh_bounds, "jaxspin_feh_bounds"
        )
        ro_wmb_factor = numpyro.sample(
            "Ro_wmb_factor", dist.Uniform(ro_lower, ro_upper)
        )
        inferred_feh = numpyro.sample(
            "feh", dist.Uniform(feh_lower, feh_upper)
        )
        return (ro_wmb_factor, inferred_feh), None, None
    slope_prior = dist.Uniform(0, 5) if predict_P_rot else dist.Uniform(-5, 0)
    if rotation_law == "power":
        logvf = numpyro.sample("logvf", dist.Uniform(-3, 2))
        vf = numpyro.deterministic("vf", 10**logvf)
        slopes = (numpyro.sample("a", slope_prior), vf)
        break_age = None
    elif rotation_law == "gp":
        gp_latent = numpyro.sample(
            "gp_latent",
            dist.Normal(0.0, 1.0).expand((GP_LOGAGE_KNOTS.size,)).to_event(1),
        )
        log_gp_amplitude = numpyro.sample(
            "log_gp_amplitude", dist.Uniform(-2.0, 1.0)
        )
        log_gp_scale = numpyro.sample(
            "log_gp_scale", dist.Uniform(-1.5, 0.5)
        )
        gp_amplitude = numpyro.deterministic(
            "gp_amplitude", 10**log_gp_amplitude
        )
        gp_scale = numpyro.deterministic("gp_scale", 10**log_gp_scale)
        slopes = (gp_latent, gp_amplitude, gp_scale)
        break_age = None
    else:
        slopes = (
            numpyro.sample("a1", slope_prior),
            numpyro.sample("a2", dist.Uniform(0, 0.3)),
        )
        if len(break_age_bounds) != 2:
            raise ValueError("break_age_bounds must contain (lower, upper)")
        lower, upper = map(float, break_age_bounds)
        if not 0 < lower < upper:
            raise ValueError("break_age_bounds must satisfy 0 < lower < upper")
        break_age = numpyro.sample("break_age", dist.Uniform(lower, upper))
    logb = numpyro.sample("logb", dist.Uniform(-1, 2))
    b = numpyro.deterministic("b", 10**logb)
    return slopes, b, break_age
