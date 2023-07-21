from typing import Tuple, Optional, Union
import jax
import jax.numpy as jnp
import chex
from flax import struct
from evosax import Strategy
from core import GradientOptimizer, OptState, OptParams, exp_decay

@struct.dataclass
class EvoState:
    mean: chex.Array
    sigma: float
    opt_state: OptState
    best_member: chex.Array
    recent_utilities: chex.Array
    rng: chex.PRNGKey
    num_comms: chex.Array
    mean_err: float = 0.0
    step: int = 0
    best_fitness: float = jnp.finfo(jnp.float32).max
    gen_counter: int = 0


@struct.dataclass
class EvoParams:
    opt_params: OptParams
    sigma_init: float = 0.04
    sigma_decay: float = 0.999
    sigma_limit: float = 0.01
    init_min: float = 0.0
    init_max: float = 0.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class ReducedOpenES(Strategy):
    def __init__(
        self,
        popsize: int,
        commsize: int,
        horizon_length: int,
        rng: chex.PRNGKey,
        num_gens: int,
        num_dims: Optional[int] = None,
        pholder_params: Optional[Union[chex.ArrayTree, chex.Array]] = None,
        opt_name: str = "adam",
        lrate_init: float = 0.05,
        lrate_decay: float = 1.0,
        lrate_limit: float = 0.001,
        sigma_init: float = 0.03,
        sigma_decay: float = 1.0,
        sigma_limit: float = 0.01,
        mean_decay: float = 0.0,
        n_devices: Optional[int] = None,
        **fitness_kwargs: Union[bool, int, float]
    ):
        """OpenAI-ES (Salimans et al. (2017)
        Reference: https://arxiv.org/pdf/1703.03864.pdf
        Inspired by: https://github.com/hardmaru/estool/blob/master/es.py"""
        super().__init__(
            popsize,
            num_dims,
            pholder_params,
            mean_decay,
            n_devices,
            **fitness_kwargs
        )
        assert not self.popsize & 1, "Population size must be even"
        assert opt_name in ["sgd", "adam", "rmsprop", "clipup", "adan"]
        self.optimizer = GradientOptimizer[opt_name](self.num_dims)
        self.strategy_name = "ReducedOpenES"

        self.commsize = commsize
        self.horizon_length = horizon_length
        self.rng=rng
        self.num_gens = num_gens
        # Set core kwargs es_params (lrate/sigma schedules)
        self.lrate_init = lrate_init
        self.lrate_decay = lrate_decay
        self.lrate_limit = lrate_limit
        self.sigma_init = sigma_init
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit

    @property
    def params_strategy(self) -> EvoParams:
        """Return default parameters of evolution strategy."""
        opt_params = self.optimizer.default_params.replace(
            lrate_init=self.lrate_init,
            lrate_decay=self.lrate_decay,
            lrate_limit=self.lrate_limit,
        )
        return EvoParams(
            opt_params=opt_params,
            sigma_init=self.sigma_init,
            sigma_decay=self.sigma_decay,
            sigma_limit=self.sigma_limit,
        )

    def initialize_strategy(
        self, rng: chex.PRNGKey, params: EvoParams
    ) -> EvoState:
        """`initialize` the evolution strategy."""
        rng, _rng = jax.random.split(rng)
        initialization = jax.random.uniform(
            _rng,
            (self.num_dims,),
            minval=params.init_min,
            maxval=params.init_max,
        )
        initial_utilities = jnp.zeros((int(self.popsize / 2), self.horizon_length))
        state = EvoState(
            mean=initialization,
            sigma=params.sigma_init,
            opt_state=self.optimizer.initialize(params.opt_params),
            best_member=initialization,
            recent_utilities=initial_utilities,
            rng=rng,
            num_comms=jnp.zeros(self.num_gens)
        )
        return state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: EvoState, params: EvoParams
    ) -> Tuple[chex.Array, EvoState]:
        """`ask` for new parameter candidates to evaluate next."""
        # Antithetic sampling of noise
        z_plus = jax.random.normal(
            rng,
            (int(self.popsize / 2), self.num_dims),
        )
        z = jnp.concatenate([z_plus, -1.0 * z_plus])
        x = state.mean + state.sigma * z
        return x, state

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: EvoState,
        params: EvoParams,
    ) -> EvoState:
        """`tell` performance data for strategy state update."""
        # Reconstruct noise from last mean/std estimates
        pivot = int(fitness.size/2)
        x_hat = x[:pivot,:]
        fitness_hat = fitness.reshape(2,-1)
        fitness_hat = fitness_hat * jnp.array([[1],[-1]])
        fitness_hat = jnp.sum(fitness_hat,axis=0)
        noise = (x_hat - state.mean) / state.sigma
        noise_norm = jnp.linalg.norm(noise, axis=1)
        utility = noise_norm*fitness_hat
        recent_utilities = jnp.roll(state.recent_utilities,-1)
        recent_utilities = recent_utilities.at[:,-1].set(utility)

        means = jnp.mean(recent_utilities, axis=1)
        stds = jnp.std(recent_utilities, axis=1)
        p_norm = jax.scipy.stats.norm.cdf((utility-means)/stds)
        # scipy.stats.binom.cdf(k,n,p) ~~ jax.scipy.special.betainc(n - k, k + 1, 1 - p)
        p_comms = 1-jax.scipy.special.betainc(self.commsize-1, int(self.popsize / 2)-self.commsize+1, 1 - p_norm)



        rng, _rng = jax.random.split(state.rng)
        comm_mask = jax.random.uniform(_rng, shape=(int(self.popsize / 2),)) < p_comms

        num_comm = jnp.sum(comm_mask)
        num_comms = state.num_comms.at[state.step].set(num_comm)
        masked_fitness = jax.lax.select(comm_mask, fitness_hat, jnp.zeros(int(self.popsize / 2)))
        theta_grad = (
            1.0 / (2 * jnp.max(jnp.array([num_comm, 1])) * state.sigma) * jnp.dot(noise.T, masked_fitness)
        )

        open_es_theta_grad = (
            1.0 / (self.popsize * state.sigma) * jnp.dot(noise.T, fitness_hat)
        )
        # err = jnp.linalg.norm(theta_grad-open_es_theta_grad)
        err = jnp.sum(theta_grad-open_es_theta_grad)
        # err = 1-jnp.dot(theta_grad/jnp.linalg.norm(theta_grad), open_es_theta_grad/jnp.linalg.norm(theta_grad))
        # mean_err = (state.step * state.mean_err + err)/(state.step+1)
        mean_err = state.mean_err + err

        # Grad update using optimizer instance - decay lrate if desired
        mean, opt_state = self.optimizer.step(
            state.mean, theta_grad, state.opt_state, params.opt_params
        )
        opt_state = self.optimizer.update(opt_state, params.opt_params)
        sigma = exp_decay(state.sigma, params.sigma_decay, params.sigma_limit)
        return state.replace(mean=mean, sigma=sigma, opt_state=opt_state, rng=rng, recent_utilities=recent_utilities, mean_err=mean_err, step=state.step+1, num_comms=num_comms)