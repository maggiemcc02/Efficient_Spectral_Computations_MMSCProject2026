from firedrake import *
import numpy as np
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
import os




# SET THE EIGENPROBLEM AND SOLVER
######################################################################



# Set a triangular mesh like in Matt's example (Fig 3.22)
N = 32

mesh = SquareMesh(N, N, pi, quadrilateral=False) 

# Decide if you want edge elements or Lagrange elements to discretize the space for E
hcurl = False
if hcurl:
    V0 = FunctionSpace(mesh, "N1curl", 2)
else:
    V0 = VectorFunctionSpace(mesh, "CG", 1, dim=2)

# The space for H is CG1
V1 = FunctionSpace(mesh, "CG", 1)

# We need a mixed function space here: (space for E) x (space for H)
Z = MixedFunctionSpace([V0, V1])

# Set zero tangential trace conditions for E:
# Edge elements only force continuity in tangential component so we just set zero conditions on boundary.
# If using Lagrange elements, we need so explicily set the tangential trace to zero.
if hcurl:
    bc = [DirichletBC(Z.sub(0), Constant((0, 0)), "on_boundary")]
else:
    bc = [DirichletBC(Z.sub(0).sub(0), 0, (3, 4)),
          DirichletBC(Z.sub(0).sub(1), 0, (1, 2))]


# Scalar rot
def rot_s(E):
    return E[1].dx(0) - E[0].dx(1)

# Vector rot
def rot_v(H):
    return as_vector([H.dx(1), -H.dx(0)])

# Set the complex unit as j
j = Constant(1j)  

# Test and trial functions
u = TrialFunction(Z)
(E, H) = split(u)
v = TestFunction(Z)
(F, G) = split(v)

# Placeholder for eigenvalue
z = Constant(0)

# Set Ln(z) = <(A-zI)u, (A-zI)v>
a = (
      # A^* M^{-1} A terms
      inner(rot_s(E), rot_s(F))*dx
    + inner(rot_v(H), rot_v(G))*dx
      # -2 z A terms
    - 2 * z * inner(j*rot_v(H), F)*dx
    + 2 * z * inner(j*rot_s(E), G)*dx
      # |z|^2 M terms
    + conj(z) * z * inner(E, F)*dx
    + conj(z) * z * inner(H, G)*dx
    )

# Set Gn = <u, v>
b = inner(E, F)*dx + inner(H, G)*dx


# Set the generalized eigenproblem Ln(z)x = lambda Gn x on restructed space
problem = LinearEigenproblem(a, b, bcs=bc, restrict=True)

# Solver parameters and subsequent solver
sp = {"eps_gen_hermitian": None,  # solver parameters, passed to SLEPc
      "eps_type": "krylovschur",
      "eps_monitor": None,
      "eps_smallest_magnitude": None,
      #"eps_view": None,  # uncomment to see the solver
      "eps_target": 0,
      "eps_target_real": None,
      "st_type": "sinvert",
      }
solver = LinearEigensolver(problem, n_evals=1, solver_parameters=sp) # ask for one eigenpair







# COMPUTE PHI AND INVG AT EACH POINT IN GRID(n)
######################################################################





# Set Grid(n) and empty lists for gammas and invg vals
h = 0.02
n = 1/h
grid = np.append(np.arange(0.01, 4, h),[4]) # Patrick makes this a list
phi_vals = []
invg_vals = []

# Set an empty function on the restricted space to hold our eigenfunction initial guess
cache_guess = Function(problem.restricted_space)

print()
print()
print(GREEN % f'Step 1 - Compute Phi and Invg at Each Grid Point')
print(GREEN % f'-'*100)
print()
print()

# Do the first pass through of the grid to compute the gammas
for curr_z in grid:


    z.assign(curr_z) # set z as current grid point

    nconv = solver.solve() # solver the eigenproblem


    min_eigval = solver.eigenvalue(0) # the first (smallest) eigenvalue is what we desire

    # This should be nonnegative and real (self-adjoint).
    # If it is negative, it may be because its really small or because of a geniune issue!
    # So we will flag the user when the eigenvalue is negative.
    neg_tol = -1e-8
    if min_eigval < 0:
      print(RED % f"WARNING - the min eigenval is negative: lambda_min = {min_eigval}")
      if min_eigval <= -neg_tol:
        print(RED % f"Since the eigenvalue is small and negative, we set lambda_min = 0")
        min_eigval = 0.0
      else:
        raise ValueError(RED % f"The eigenvalue is significantly negative and thats an issue so we quit!")


    phi = sqrt(min_eigval) # Our bound on distance to the spectrum - Phi_n = sqrt{lambda_min} >= dist(z, Sp(A))


    invg = (np.floor(n * phi) + 1)/n # Our grid rounded bound, used later for search ball.

    phi_vals.append(phi)
    invg_vals.append(invg)



    # Track the results
    print(BLUE % f"z = {curr_z}: |dist(z, spectrum)| <= {phi}") # Patrick's output choice

    # Cycle eigenspace from one solve to another. Krylov-Schur only takes
    # one initial guess as input, so we just need to cycle one vector
    with cache_guess.dat.vec_wo as vec_r:
        solver.es.getEigenvector(0, vec_r) # place the eigenfunction for lambda_min in vec_r
        solver.es.setInitialSpace(vec_r) # use the eigenfunction (in vec_r) as initial guess for next iter (next grid point)







# LOCAL MINIMIZATION
######################################################################



print()
print()
print(GREEN % f'Step 2 - Local Minimization to Form the Approx of Sp(A)')
print(GREEN % f'-'*100)
print()
print()


# Convert everything to numpy array so we can use numpy commands
phi_vals = np.array(phi_vals)
invg_vals = np.array(invg_vals)


# Now do a second pass through and determine if we will do local minimization
gamma_indexset = set()

for i in range(len(grid)):


  # pull the grid point and the associated phi and invg
  curr_z = grid[i]
  phi = phi_vals[i]
  invg = invg_vals[i]
  print(BLUE % f'Check if we want to locally minimize near z =  {curr_z} with Phi = {phi}')


  # check if we want to do local minimization
  minimize = False
  if phi <= (1 / (abs(curr_z)**2 + 1)):
    minimize = True
    print('Since phi <= (|z|^2 + 1)^{-1} we will minimize')

  # If we want to minimize then do it!
  if minimize:

    print(BLUE % f"We do want to minimize:")

    # form the invg ball
    B_invg_left = curr_z - invg
    B_invg_right = curr_z + invg

    print('B_{invg} = [', B_invg_left, ',', B_invg_right, ']')

    # form the search ball as intersection of invg ball with grid
    search_ball = np.where((grid >= B_invg_left) & (grid <= B_invg_right))[0]
    print('The search ball is', grid[search_ball])

    # pull the phi values on the search ball
    phi_check = phi_vals[search_ball]

    # find the goal min
    goal_min = np.min(phi_check)
    print('the goal min is Phi =', goal_min)

    # take all the values close to goal min as our local mins

    min_tol = 1e-6 # how close do we want to be to the goal min?

    find_localmins = phi_check <= (goal_min + min_tol) # list of Trues and Falses for each grid point in search ball

    local_min_inds = search_ball[find_localmins] # pull the indices of the local mins in the global grid
    print('The local mins are Phi =', phi_vals[local_min_inds])
    print('The local minimizers are z = ', grid[local_min_inds])

    gamma_indexset.update(local_min_inds) # save the local min indices to our gamma set

  else:
    print(BLUE % f"We dont want to minimize")
  print()


# Once we are done, gamma_indexset holds the grid indices corresponding
# to the local minimizers we use to approximate the spectrum.
# Let's pull the grid points as our approximate spectrum, Gamma.
# Let's also pull the associated invg values, which is our Error function En.
Gamma_inds =  np.array(sorted(gamma_indexset)) # sort the indices in the set and save them as Numpy array
Gamma = grid[Gamma_inds]
En = invg_vals[Gamma_inds]






# PLOTTING ROUTINE
######################################################################




os.makedirs("output/mixed/", exist_ok=True)


# desired eigenvalues
exact = np.array([n**2 + m**2 for n in range(10) for m in range(10) if n**2 + m**2 <= 4**2])

# phi and invg
plt.plot(grid, phi_vals, linewidth=2, label = r"$\Phi_n(z, M_{2D})$")
plt.plot(grid, invg_vals, linewidth=2, linestyle = "--", label = r"Invg($z, \Phi_n(z, M_{2D})$)")
plt.plot(np.sqrt(exact), 0*exact, 'ok', markersize=5, label = r'exact $\omega$')
plt.xlabel(r"$x$")
plt.title(rf"Approximations of $\Phi_n(z, M_{{2D}})$, $\operatorname{{Invg}}(z, \Phi_n(z, M_{{2D}}))$, and $\omega$ ($N = {N}$)")
plt.legend()
plt.savefig(f"output/mixed/phi_invg_plot_{N=}.pdf")
plt.close()

# The eigenvalue comparison

# computed eigenvalues
plt.plot([i for i in range(len(Gamma))], Gamma, 'or', markersize=5 , label = r"approx $\omega$")
# plot exact eigenvalues as dashed lines
for val in exact[:-1]:
    plt.axhline(np.sqrt(val), linestyle="--", linewidth=0.8, color = 'k')
plt.axhline(np.sqrt(exact[-1]), linestyle="--", linewidth=0.8, color = 'k', label = r"true $\omega$") # for the legend
plt.xlabel("Eigenvalue Number")
plt.ylabel(r"$\omega^2$")
plt.title(rf"Comparing Exact and Computed $\omega$ ($N = {N}$)")
plt.legend()
plt.savefig(f"output/mixed/omega_plot_{N=}.pdf")
plt.close()


