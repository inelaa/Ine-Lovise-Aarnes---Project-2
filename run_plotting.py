from plotting import plot_SGD_MSE_convergence_epoch_batch
from GD_class import *

"""
Script for plotting convergencegraph
"""


"""
Functions needed
"""

def cost_function_OLS(X, y, beta):
    n = len(y)  # Define the number of data points
    return (1.0/n) * jnp.sum((y - jnp.dot(X, beta))**2)

def analytical_gradient(X, y, beta):
    n = len(y)
    return (2.0/n)*jnp.dot(X.T, ((jnp.dot(X, beta))-y))

def make_design_matrix(x, degree):
    "Creates the design matrix for the given polynomial degree and ijnput data"

    X = np.zeros((len(x), degree + 1))

    for i in range(X.shape[1]):
        X[:, i] = np.power(x, i)

    return jnp.array(X)

"""
generate data and design matrix
"""

np.random.seed(1342)

true_beta = [2, 0.5, 3.2]

n = 1000

x = jnp.linspace(0, 1, n)
y = jnp.sum(jnp.asarray([x ** p * b for p, b in enumerate(true_beta)]),
                axis=0) + 0.2 * np.random.normal(size=len(x))

# Making a design matrix to use for linear regression part
degree = 2
X = make_design_matrix(x, degree)

# Set parameters
learning_rate = 0.1
tol=1e-3
momentum=0.5
delta= 1e-8
rho1 = 0.9
rho2 = 0.99

"""
make classes
"""


# OLS analytical Adam
grad_descentADAM = GradientDescentADAM(delta, rho1, rho2, X=X, y=y,
                                            learning_rate=learning_rate, tol=tol,
                                            cost_function=cost_function_OLS,
                                            analytic_gradient=analytical_gradient,
                                            skip_convergence_check=True,
                                            record_history=True)

# OLS analytical RMSprop
grad_descentRMS_prop = GradientDescentRMSprop(delta, rho1, X=X, y=y,
                                            learning_rate=learning_rate, tol=tol,
                                            cost_function=cost_function_OLS,
                                            analytic_gradient=analytical_gradient,
                                            skip_convergence_check=True,
                                            record_history=True)
# OLS analytical Adagrad
grad_descentAdagrad = GradientDescentAdagrad(delta, rho1, X=X, y=y,
                                            learning_rate=learning_rate, tol=tol,
                                            cost_function=cost_function_OLS,
                                            analytic_gradient=analytical_gradient,
                                            skip_convergence_check=True,
                                            record_history=True)
# OLS analytical momentum
grad_descentMomentum = GradientDescentMomentum(momentum=0.3, X=X, y=y,
                                            learning_rate=learning_rate, tol=tol,
                                            cost_function=cost_function_OLS,
                                            analytic_gradient=analytical_gradient,
                                            skip_convergence_check=True,
                                            record_history=True)

grad_descentNoMomentum = GradientDescentMomentum(momentum=0, X=X, y=y,
                                            learning_rate=learning_rate, tol=tol,
                                            cost_function=cost_function_OLS,
                                            analytic_gradient=analytical_gradient,
                                            skip_convergence_check=True,
                                            record_history=True)
"""
Now optimal num batches and epochs have been found. plot the convergence of this
"""

optimization_methods = ['no momentum', 'momentum', 'RMSprop', 'adagrad', 'adam']


cost_scores_optimization = []

for optimization in optimization_methods:

    if optimization == 'no momentum':
        grad_descentNoMomentum.iterate(iteration_method="Stochastic", max_epoch=200,
                                       num_batches=50)

        cost_scores_optimization.append(grad_descentNoMomentum.cost_scores)

    if  optimization == 'momentum':
        grad_descentMomentum.iterate(iteration_method="Stochastic", max_epoch=200,
                                        num_batches=50)
        cost_scores_optimization.append(grad_descentMomentum.cost_scores)

    if optimization == 'RMSprop':
        grad_descentRMS_prop.iterate(iteration_method="Stochastic", max_epoch=100,
                                        num_batches=50)
        cost_scores_optimization.append(grad_descentRMS_prop.cost_scores)

    if optimization == 'adagrad':
        grad_descentAdagrad.iterate(iteration_method="Stochastic", max_epoch=200,
                                        num_batches=50)
        cost_scores_optimization.append(grad_descentAdagrad.cost_scores)

    if optimization == 'adam':

        grad_descentADAM.iterate(iteration_method="Stochastic", max_epoch=50,
                                      num_batches=50)
        cost_scores_optimization.append(grad_descentADAM.cost_scores)



plot_SGD_MSE_convergence_epoch_batch(optimization_methods, cost_scores_optimization)

