import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from functools import partial


class Model():
    """A toy quadratic problem based on a Kronecker factorisation."""

    def __init__(self, num_samples, dimension):
        self.dimension = dimension
        self.linear_coeff = np.random.randn(dimension**2)

        self.left_vectors = np.random.randn(num_samples, dimension)
        self.right_vectors = np.random.randn(num_samples, dimension)

        left_factors = np.stack(
            [np.outer(vector, vector)
             for vector in self.left_vectors])
        right_factors = np.stack(
            [np.outer(vector, vector)
                for vector in self.right_vectors])

        full_matrices = np.stack(
            [np.kron(left, right)
                for left, right in zip(left_factors, right_factors)])

        self.true_curvature = full_matrices.mean(axis=0)
        self.avg_left_factors = left_factors.mean(axis=0)
        self.avg_right_factors = right_factors.mean(axis=0)

        self.grad = jax.jit(jax.grad(self.__call__))
        self.min_value = self(jnp.linalg.solve(self.true_curvature, -self.linear_coeff))

    @partial(jax.jit, static_argnums=0)
    def __call__(self, data):
        return ((0.5 * data.T @ self.true_curvature @ data)
                + (data.T @ self.linear_coeff))

    @partial(jax.jit, static_argnums=0)
    def line_search(self, point, direction):
        local_grad = self.grad(point)
        return (-local_grad.T @ direction) / (direction.T @ self.true_curvature @ direction)

    def regret(self, data):
        return self(data) - self.min_value


@partial(jax.jit, static_argnames=['mode', 'factored'])
def curvature_loss(true_curvature, gradient, *inputs, mode, factored, damping=0):
    if factored:
        # inputs = (left_factor, right_factor, left_corr, right_corr)
        assert len(inputs) == 4
        approx_curvature = jnp.kron(inputs[0] + inputs[2],
                                    inputs[1] + inputs[3])
    else:
        # inputs = (approx_matrix)
        assert len(inputs) == 1
        approx_curvature = inputs[0]

    approx_curvature += damping*jnp.eye(approx_curvature.shape[0])

    if mode == 'approx_true_grad':
        metric_vector = (jnp.linalg.solve(approx_curvature,
                                          true_curvature) @ gradient
                         - gradient)
    elif mode == 'true_approx_grad':
        metric_vector = (true_curvature @ jnp.linalg.solve(approx_curvature,
                                                           gradient)
                         - gradient)
    elif mode == 'separate_product':
        metric_vector = true_curvature @ gradient - approx_curvature @ gradient

    return jnp.linalg.norm(metric_vector, ord=2)


def compute_regrets(step_fn, initial_state, model, num_optimisation_iterations):
    state = initial_state.copy()
    state['trajectory'] = (jnp.empty((num_optimisation_iterations + 1,
                                      *state['point'].shape))
                           .at[0].set(state['point']))
    state = jax.lax.fori_loop(0,
                              num_optimisation_iterations,
                              step_fn,
                              state)
    regrets = jax.vmap(model.regret)(state['trajectory'])
    return regrets


def compute_sgd_trajectory(initial_point, model, num_optimisation_iterations):
    def step(iteration, state):
        state = state.copy()
        direction = -model.grad(state['point'])
        step_size = model.line_search(state['point'], direction)
        state['point'] = state['point'] + step_size*direction
        state['trajectory'] = state['trajectory'].at[iteration+1].set(state['point'])
        return state

    initial_state = dict(point=initial_point)
    return compute_regrets(step, initial_state, model, num_optimisation_iterations)


def compute_adam_trajectory(initial_point, model, num_optimisation_iterations,
                            betas=(0.9, 0.999), eps=1e-8, v_hat_transform=jnp.sqrt):
    def step(iteration, state):
        state = state.copy()
        gradient = model.grad(state['point'])
        state['m'] = betas[0]*state['m'] + (1-betas[0])*gradient
        state['v'] = betas[1]*state['v'] + (1-betas[1])*gradient**2
        state['m_hat'] = state['m'] / (1 - betas[0]**(iteration+1))
        state['v_hat'] = state['v'] / (1 - betas[1]**(iteration+1))

        direction = -state['m_hat'] / (v_hat_transform(state['v_hat']) + eps)
        step_size = model.line_search(state['point'], direction)
        state['point'] = state['point'] + step_size*direction
        state['trajectory'] = state['trajectory'].at[iteration+1].set(state['point'])
        return state

    initial_state = dict(point=initial_point,
                         m=jnp.zeros_like(initial_point),
                         v=jnp.zeros_like(initial_point),
                         m_hat=jnp.zeros_like(initial_point),
                         v_hat=jnp.zeros_like(initial_point))
    return compute_regrets(step, initial_state, model, num_optimisation_iterations)


def compute_kfac_trajectory(initial_point, model, num_optimisation_iterations, damping=0):
    def step(iteration, state):
        state = state.copy()
        gradient = model.grad(state['point'])
        direction = -jnp.linalg.lstsq(
            jnp.kron(model.avg_left_factors, model.avg_right_factors) + damping*jnp.eye(model.true_curvature.shape[0]),
            gradient)[0]
        step_size = model.line_search(state['point'], direction)
        state['point'] = state['point'] + step_size*direction
        state['trajectory'] = state['trajectory'].at[iteration+1].set(state['point'])
        return state

    initial_state = dict(point=initial_point)
    return compute_regrets(step, initial_state, model, num_optimisation_iterations)


def construct_kfac_correction_learning_step(loss_grad_fn, model, correction_lr):
    def correction_learning_step(iteration, state):
        state = state.copy()
        loss_gradient = loss_grad_fn(model.true_curvature,
                                     state['gradient'],
                                     model.avg_left_factors,
                                     model.avg_right_factors,
                                     state['left_correction'],
                                     state['right_correction'])
        state['left_correction'] = state['left_correction'] - correction_lr * loss_gradient[0]
        state['right_correction'] = state['right_correction'] - correction_lr * loss_gradient[1]
        return state

    return correction_learning_step


def compute_kfac_constant_trajectory(initial_point, model, num_optimisation_iterations, num_correction_iterations, loss_fn, correction_lr=0.001, damping=0):
    loss_grad_fn = jax.grad(loss_fn, argnums=(2, 3))
    initial_prelearning_state = dict(
        left_correction=jnp.zeros_like(model.avg_left_factors),
        right_correction=jnp.zeros_like(model.avg_right_factors),
        gradient=model.grad(initial_point))
    final_prelearning_state = jax.lax.fori_loop(0,
                                                num_correction_iterations * num_optimisation_iterations,
                                                construct_kfac_correction_learning_step(loss_grad_fn, model, correction_lr),
                                                initial_prelearning_state)
    left_correction = final_prelearning_state['left_correction']
    right_correction = final_prelearning_state['right_correction']

    def step(iteration, state):
        state = state.copy()
        gradient = model.grad(state['point'])
        direction = -jnp.linalg.lstsq(
            jnp.kron(model.avg_left_factors + left_correction,
                     model.avg_right_factors + right_correction) + damping*jnp.eye(model.true_curvature.shape[0]),
            gradient)[0]
        step_size = model.line_search(state['point'], direction)
        state['point'] = state['point'] + step_size * direction
        state['trajectory'] = state['trajectory'].at[iteration+1].set(state['point'])
        return state

    initial_state = dict(point=initial_point)
    return compute_regrets(step, initial_state, model, num_optimisation_iterations)


def compute_kfac_dynamic_trajectory(initial_point, model, num_optimisation_iterations, num_correction_iterations, loss_fn, correction_lr=0.001, damping=0):
    def step(iteration, state):
        state = state.copy()
        gradient = model.grad(state['point'])
        state['correction_state']['gradient'] = gradient
        state['correction_state'] = jax.lax.fori_loop(0,
                                                      num_correction_iterations,
                                                      construct_kfac_correction_learning_step(loss_grad_fn, model, correction_lr),
                                                      state['correction_state'])

        direction = -jnp.linalg.lstsq(
            jnp.kron(model.avg_left_factors + state['correction_state']['left_correction'],
                     model.avg_right_factors + state['correction_state']['right_correction'])
            + damping*jnp.eye(model.true_curvature.shape[0]),
            gradient)[0]
        step_size = model.line_search(state['point'], direction)
        state['point'] = state['point'] + step_size * direction
        state['trajectory'] = state['trajectory'].at[iteration+1].set(state['point'])
        return state

    loss_grad_fn = jax.grad(loss_fn, argnums=(2, 3))
    initial_state = dict(
        point=initial_point,
        correction_state=dict(
            left_correction=jnp.zeros_like(model.avg_left_factors),
            right_correction=jnp.zeros_like(model.avg_right_factors),
            gradient=jnp.empty_like(initial_point)))
    return compute_regrets(step, initial_state, model, num_optimisation_iterations)


def construct_matrix_lr_learning_step(loss_grad_fn, model, correction_lr):
    def matrix_lr_learning_step(iteration, state):
        state = state.copy()
        loss_gradient = loss_grad_fn(model.true_curvature,
                                     state['gradient'],
                                     state['matrix_lr'])
        state['matrix_lr'] = state['matrix_lr'] - correction_lr * loss_gradient[0]
        return state

    return matrix_lr_learning_step


def compute_matrix_lr_constant_trajectory(initial_point, model, num_optimisation_iterations, num_correction_iterations, loss_fn, correction_lr=0.001, damping=0):
    loss_grad_fn = jax.grad(loss_fn, argnums=(2))
    initial_matrix_lr_state = dict(gradient=model.grad(initial_point),
                                   matrix_lr=jnp.eye(model.true_curvature.shape[0]))
    final_matrix_lr_state = jax.lax.fori_loop(0,
                                              num_correction_iterations * num_optimisation_iterations,
                                              construct_matrix_lr_learning_step(loss_grad_fn, model, correction_lr),
                                              initial_matrix_lr_state)
    matrix_lr = final_matrix_lr_state['matrix_lr']

    def step(iteration, state):
        gradient = model.grad(state['point'])
        direction = -jnp.linalg.lstsq(matrix_lr + damping*jnp.eye(matrix_lr.shape[0]), gradient)[0]
        state['point'] = state['point'] + direction
        state['trajectory'] = state['trajectory'].at[iteration+1].set(state['point'])
        return state

    initial_state = dict(point=initial_point)
    return compute_regrets(step, initial_state, model, num_optimisation_iterations)


def compute_matrix_lr_dynamic_trajectory(initial_point, model, num_optimisation_iterations, num_correction_iterations, loss_fn, correction_lr=0.001, damping=0):
    loss_grad_fn = jax.grad(loss_fn, argnums=(2,))

    def step(iteration, state):
        state = state.copy()
        gradient = model.grad(state['point'])
        state['matrix_lr_state']['gradient'] = gradient
        state['matrix_lr_state'] = jax.lax.fori_loop(
            0,
            num_correction_iterations,
            construct_matrix_lr_learning_step(loss_grad_fn, model, correction_lr),
            state['matrix_lr_state'])

        direction = (-jnp.linalg.lstsq(state['matrix_lr_state']['matrix_lr']
                                       + damping*jnp.eye(state['matrix_lr_state']['matrix_lr'].shape[0]),
                                       gradient)[0])
        state['point'] = state['point'] + direction
        state['trajectory'] = state['trajectory'].at[iteration+1].set(state['point'])
        return state

    initial_state = dict(
        point=initial_point,
        matrix_lr_state=dict(
            matrix_lr=jnp.eye(model.true_curvature.shape[0]),
            gradient=jnp.empty_like(initial_point)))
    return compute_regrets(step, initial_state, model, num_optimisation_iterations)


def compute_diagonal_curvature_trajectory(initial_point, model, num_optimisation_iterations, damping=0):
    def step(iteration, state):
        state = state.copy()
        gradient = model.grad(state['point'])
        direction = -gradient / (jnp.diag(model.true_curvature) + damping)
        step_size = model.line_search(state['point'], direction)
        state['point'] = state['point'] + step_size * direction
        state['trajectory'] = state['trajectory'].at[iteration+1].set(state['point'])
        return state

    initial_state = dict(point=initial_point)
    return compute_regrets(step, initial_state, model, num_optimisation_iterations)


def compute_newton_trajectory(initial_point, model, num_optimisation_iterations, damping=0):
    def step(iteration, state):
        state = state.copy()
        gradient = model.grad(state['point'])
        direction = -jnp.linalg.lstsq(model.true_curvature + damping*jnp.eye(model.true_curvature.shape[0]), gradient)[0]
        state['point'] = state['point'] + direction
        state['trajectory'] = state['trajectory'].at[iteration+1].set(state['point'])
        return state

    initial_state = dict(point=initial_point)
    return compute_regrets(step, initial_state, model, num_optimisation_iterations)


def compute_low_rank_moore_penrose_trajectory(initial_point, model, num_optimisation_iterations, num_samples, damping=0):
    base_vectors = [np.kron(left_vector, right_vector)
                    for left_vector, right_vector in zip(model.left_vectors,
                                                         model.right_vectors)]

    def step(iteration, state):
        state = state.copy()
        gradient = model.grad(state['point'])
        approx_inverse_curvature = np.sum(
            [np.outer(vector, vector) / np.linalg.norm(vector, ord=2)**4
             for vector in base_vectors],
            axis=0)
        direction = -(approx_inverse_curvature + damping*jnp.eye(approx_inverse_curvature.shape[0])) @ gradient
        step_size = num_samples
        state['point'] = state['point'] + step_size * direction
        state['trajectory'] = state['trajectory'].at[iteration+1].set(state['point'])
        return state

    initial_state = dict(point=initial_point)
    return compute_regrets(step, initial_state, model, num_optimisation_iterations)


def compute_low_rank_sgd_fallback_trajectory(initial_point, model, num_optimisation_iterations, num_samples, meta_lr=0.001, damping=0):
    base_vectors = [np.kron(left_vector, right_vector)
                    for left_vector, right_vector in zip(model.left_vectors,
                                                         model.right_vectors)]

    def step(iteration, state):
        state = state.copy()
        gradient = model.grad(state['point'])
        approx_inverse_curvature = meta_lr * np.eye(model.true_curvature.shape[0]) + np.sum(
            [(1 / np.linalg.norm(vector, ord=2)**2 - meta_lr)
             * (1 / np.linalg.norm(vector, ord=2)**2)
             * np.outer(vector, vector)
             for vector in base_vectors],
            axis=0)
        direction = -(approx_inverse_curvature + damping*jnp.eye(approx_inverse_curvature.shape[0])) @ gradient
        step_size = num_samples
        state['point'] = state['point'] + step_size * direction
        state['trajectory'] = state['trajectory'].at[iteration+1].set(state['point'])
        return state

    initial_state = dict(point=initial_point)
    return compute_regrets(step, initial_state, model, num_optimisation_iterations)


def compute_eva_trajectory(initial_point, model, num_optimisation_iterations, damping=0):
    avg_left_vectors = model.left_vectors.mean(axis=0)
    avg_right_vectors = model.right_vectors.mean(axis=0)

    def step(iteration, state):
        state = state.copy()
        gradient = model.grad(state['point'])
        direction = -jnp.linalg.lstsq(
            jnp.kron(jnp.outer(avg_left_vectors, avg_left_vectors),
                     jnp.outer(avg_right_vectors, avg_right_vectors))
            + damping*jnp.eye(model.true_curvature.shape[0]),
            gradient)[0]
        step_size = model.line_search(state['point'], direction)
        state['point'] = state['point'] + step_size*direction
        state['trajectory'] = state['trajectory'].at[iteration+1].set(state['point'])
        return state

    initial_state = dict(point=initial_point)
    return compute_regrets(step, initial_state, model, num_optimisation_iterations)

def construct_hessian_line_search_step(hessian, model, damping):
    def step(iteration, state):
        state = state.copy()
        gradient = model.grad(state['point'])
        direction = -jnp.linalg.lstsq(hessian + damping*jnp.eye(model.true_curvature.shape[0]), gradient)[0]
        step_size = model.line_search(state['point'], direction)
        state['point'] = state['point'] + step_size*direction
        state['trajectory'] = state['trajectory'].at[iteration+1].set(state['point'])
        return state

    return step

def compute_rank_one_hessian_trajectory(initial_point, model, num_optimisation_iterations, damping=0):
    u, s, vh = jnp.linalg.svd(model.true_curvature)
    s = s.at[1:].set(0)
    rank_one_hessian = u @ np.diag(s) @ vh

    initial_state = dict(point=initial_point)
    return compute_regrets(
        construct_hessian_line_search_step(rank_one_hessian, model, damping),
        initial_state,
        model,
        num_optimisation_iterations)


def compute_rank_one_permuted_hessian_trajectory(initial_point, model, num_optimisation_iterations, damping=0):
    left_factor_shape = (10, 10)
    right_factor_shape = (10, 10)
    hessian = model.true_curvature
    stacked_blocks = jnp.concatenate(
        jnp.split(hessian, left_factor_shape[1], axis=1),
        axis=0)
    permuted_hessian = jnp.stack([block.T.reshape(-1)
                                  for block in jnp.split(stacked_blocks,
                                                         left_factor_shape[0]*left_factor_shape[1])])

    u, s, vh = jnp.linalg.svd(permuted_hessian)
    s = s.at[1:].set(0)
    rank_one_permuted_hessian = u @ np.diag(s) @ vh

    rank_one_stacked_blocks = jnp.concatenate([row.reshape(*right_factor_shape).T
                                               for row in rank_one_permuted_hessian])
    rank_one_hessian = jnp.concatenate(
        np.split(rank_one_stacked_blocks, left_factor_shape[1]), axis=1)

    initial_state = dict(point=initial_point)
    return compute_regrets(
        construct_hessian_line_search_step(rank_one_hessian, model, damping),
        initial_state,
        model,
        num_optimisation_iterations)


def plot_quadratic_minimisers(dimension=10,
                              num_samples=150,
                              num_correction_iterations=1,
                              num_optimisation_iterations=200,
                              lr=0.001,
                              damping=1e-4):
    model = Model(num_samples, dimension)
    initial_point = np.random.randn(dimension**2)

    common_args = (initial_point, model, num_optimisation_iterations)
    trajectories = {
        'SGD': compute_sgd_trajectory(*common_args),
        'Adam': compute_adam_trajectory(*common_args),
        # 'Adam (Unaltered v_hat)': compute_adam_trajectory(*common_args, v_hat_transform=lambda x: x),
        # 'Adam (Unit v_hat)': compute_adam_trajectory(*common_args, v_hat_transform=lambda x: jnp.ones_like(x)),
        'KFAC': compute_kfac_trajectory(*common_args, damping=damping),
        r'KFAC Curvature (constant correction, $\Vert \widehat{\mathbf{H}}^{-1} \mathbf{H} \mathbf{g} - \mathbf{g} \Vert$)':
        compute_kfac_constant_trajectory(*common_args,
                                         num_correction_iterations,
                                         partial(curvature_loss,
                                                 mode='approx_true_grad',
                                                 factored=True,
                                                 damping=damping)),
        r'KFAC Curvature (constant correction, $\Vert \mathbf{H} \widehat{\mathbf{H}}^{-1} \mathbf{g} - \mathbf{g} \Vert$)':
        compute_kfac_constant_trajectory(*common_args,
                                         num_correction_iterations,
                                         partial(curvature_loss,
                                                 mode='true_approx_grad',
                                                 factored=True,
                                                 damping=damping)),
        r'KFAC Curvature (constant correction, $\Vert \mathbf{H}\mathbf{g} - \widehat{\mathbf{H}}\mathbf{g} \Vert$)':
        compute_kfac_constant_trajectory(*common_args,
                                         num_correction_iterations,
                                         partial(curvature_loss,
                                                 mode='separate_product',
                                                 factored=True,
                                                 damping=damping)),
        r'KFAC Curvature (dynamic correction, $\Vert \widehat{\mathbf{H}}^{-1} \mathbf{H} \mathbf{g} - \mathbf{g} \Vert$)':
        compute_kfac_dynamic_trajectory(*common_args,
                                        num_correction_iterations,
                                        partial(curvature_loss,
                                                mode='approx_true_grad',
                                                factored=True,
                                                damping=damping)),
        r'KFAC Curvature (dynamic correction, $\Vert \mathbf{H} \widehat{\mathbf{H}}^{-1} \mathbf{g} - \mathbf{g} \Vert$)':
        compute_kfac_dynamic_trajectory(*common_args,
                                        num_correction_iterations,
                                        partial(curvature_loss,
                                                mode='true_approx_grad',
                                                factored=True,
                                                damping=damping)),
        r'KFAC Curvature (dynamic correction, $\Vert \mathbf{H}\mathbf{g} - \widehat{\mathbf{H}}\mathbf{g} \Vert$)':
        compute_kfac_dynamic_trajectory(*common_args,
                                        num_correction_iterations,
                                        partial(curvature_loss,
                                                mode='separate_product',
                                                factored=True,
                                                damping=damping)),
        # r'Learned Matrix LR (constant, $\Vert \mathbf{M}^{-1} \mathbf{H} \mathbf{g} - \mathbf{g} \Vert$)':
        # compute_matrix_lr_constant_trajectory(*common_args,
        #                                       num_correction_iterations,
        #                                       partial(curvature_loss,
        #                                               mode='approx_true_grad',
        #                                               factored=False,
        #                                               damping=damping)),
        # r'Learned Matrix LR (constant, $\Vert \mathbf{H} \mathbf{M}^{-1} \mathbf{g} - \mathbf{g} \Vert$)':
        # compute_matrix_lr_constant_trajectory(*common_args,
        #                                       num_correction_iterations,
        #                                       partial(curvature_loss,
        #                                               mode='true_approx_grad',
        #                                               factored=False,
        #                                               damping=damping)),
        # r'Learned Matrix LR (constant, $\Vert \mathbf{H}\mathbf{g} - \mathbf{M}\mathbf{g} \Vert$)':
        # compute_matrix_lr_constant_trajectory(*common_args,
        #                                       num_correction_iterations,
        #                                       partial(curvature_loss,
        #                                               mode='separate_product',
        #                                               factored=False,
        #                                               damping=damping)),
        # r'Learned Matrix LR (dynamic, $\Vert \mathbf{M}^{-1} \mathbf{H} \mathbf{g} - \mathbf{g} \Vert$)':
        # compute_matrix_lr_dynamic_trajectory(*common_args,
        #                                      num_correction_iterations,
        #                                      partial(curvature_loss,
        #                                              mode='approx_true_grad',
        #                                              factored=False,
        #                                              damping=damping)),
        # r'Learned Matrix LR (dynamic, $\Vert \mathbf{H} \mathbf{M}^{-1} \mathbf{g} - \mathbf{g} \Vert$)':
        # compute_matrix_lr_dynamic_trajectory(*common_args,
        #                                      num_correction_iterations,
        #                                      partial(curvature_loss,
        #                                              mode='true_approx_grad',
        #                                              factored=False,
        #                                              damping=damping)),
        # r'Learned Matrix LR (dynamic, $\Vert \mathbf{H}\mathbf{g} - \mathbf{M}\mathbf{g} \Vert$)':
        # compute_matrix_lr_dynamic_trajectory(*common_args,
        #                                      num_correction_iterations,
        #                                      partial(curvature_loss,
        #                                              mode='separate_product',
        #                                              factored=False,
        #                                              damping=damping)),
        # 'Diagonal Curvature': compute_diagonal_curvature_trajectory(*common_args, damping=damping),
        'Newton': compute_newton_trajectory(*common_args, damping=damping),
        # 'Low-Rank Moore-Penrose': compute_low_rank_moore_penrose_trajectory(*common_args,
        #                                                                     num_samples,
        #                                                                     damping=damping),
        # 'Low-Rank SGD-Fallback': compute_low_rank_sgd_fallback_trajectory(*common_args,
        #                                                                   num_samples,
        #                                                                   damping=damping),
        # 'Eva': compute_eva_trajectory(*common_args, damping=damping),
        'Rank-1 Hessian': compute_rank_one_hessian_trajectory(*common_args, damping=damping),
        'Rank-1 Permuted Hessian': compute_rank_one_permuted_hessian_trajectory(*common_args, damping=damping),
    }

    plt.close()
    plot_lines = []
    for algorithm, trajectory in trajectories.items():
        print(f'Processing {algorithm}...')
        plot_lines.extend(
            plt.plot(trajectory, label=algorithm))

    print('Finalising plot...')
    legend = plt.legend()
    plt.xlabel('Optimisation Iteration')
    plt.ylabel('Function Regret')
    plt.yscale('log')

    line_map = {}
    fig = plt.gcf()
    for legend_line, plot_line in zip(legend.get_lines(), plot_lines):
        legend_line.set_picker(True)
        line_map[legend_line] = plot_line

    def on_pick(event):
        legend_line = event.artist
        original_line = line_map[legend_line]
        visible = not original_line.get_visible()
        original_line.set_visible(visible)
        legend_line.set_alpha(1.0 if visible else 0.2)
        fig.canvas.draw()

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()
