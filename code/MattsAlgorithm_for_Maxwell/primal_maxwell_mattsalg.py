from firedrake import *
import numpy as np
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
import os




# SET THE EIGENPROBLEM AND SOLVER
######################################################################





# Spectral discetization
N = 1
mesh = SquareMesh(N, N, pi, quadrilateral=True)
Z = VectorFunctionSpace(mesh, "Lagrange", 20, dim=2)


# Zero tangential trace 
bc = [DirichletBC(Z.sub(0), 0, (3, 4)),
      DirichletBC(Z.sub(1), 0, (1, 2))]


# Scalar rot
def rot_s(E):
    return E[1].dx(0) - E[0].dx(1)

# Vector rot
def rot_v(H):
    return as_vector([H.dx(1), -H.dx(0)])


# Set the test and trial functions
E = TrialFunction(Z)
F = TestFunction(Z)

# Placeholder for eigenvalue
z = Constant(0)

# Build Ln(z) = <(A-zI)E, (A-zI)F>
a = (
      # A^* M^{-1} A terms
      inner(rot_v(rot_s(E)), rot_v(rot_s(F)))*dx
      # -2 z A terms
    - 2 * z * inner(rot_s(E), rot_s(F))*dx
      # |z|^2 M terms
    + conj(z) * z * inner(E, F)*dx
    )

# Build Gn = <E, F>
b = inner(E, F)*dx


# Set the generalized eigenproblem, restrict the space based on boundary conditions
problem = LinearEigenproblem(a, b, bcs=bc, restrict=True)


# Set the solver parameters and the subsequent solver
sp = {"eps_gen_hermitian": None,  # solver parameters, passed to SLEPc
      "eps_type": "krylovschur",
      "eps_monitor": None,
      "eps_smallest_magnitude": None,
      #"eps_view": None,  # uncomment to see the solver
      "eps_target": 0,
      "eps_target_real": None,
      "st_type": "sinvert",
      }
solver = LinearEigensolver(problem, n_evals=1, solver_parameters=sp) # as for one eigenpair







# COMPUTE PHI AND INVG AT EACH POINT IN GRID(n)
######################################################################





# Set Grid(n) and empty lists for gammas and invg vals
h = 0.02
n = 1/h
grid = np.append(np.arange(0.01, 4, h),[4]) # Patrick makes this a list
phi_vals = []
invg_vals = []

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

    # Set initial space for next solve
    eigenfunctions = [solver.eigenfunction(i)[0] for i in range(nconv)] # pull out the converged eigenfunctions
    space = []
    for eigenfunction in eigenfunctions:
        with eigenfunction.dat.vec_ro as v: # take the underlying PETSc object (read only)
            space.append(v.copy()) # copy the PETSc object into the eigenspace
    solver.es.setInitialSpace(space) # set the eigenspace as the initial guess for the next iter (next grid point)







# LOCAL MINIMIZATION
######################################################################





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

os.makedirs("output/primal/", exist_ok=True)

# desired eigenvalues
exact = np.array([n**2 + m**2 for n in range(10) for m in range(10) if n**2 + m**2 <= 4])

# phi and invg
plt.plot(grid, phi_vals, linewidth=2, label = r"$\Phi_n(z, A)$")
plt.plot(grid, invg_vals, linewidth=2, linestyle = "--", label = r"Invg($z, \Phi_n(z, A)$)")
plt.plot(exact, 0*exact, 'ok', markersize=5, label = r'exact $\omega^2$')
plt.xlabel(r"$x$")
plt.title(rf"Approximations of $\Phi_n(z, A),\, Invg(z, \Phi_n(z, A)),$ and $\omega^2$ ($N = {N}$)")
plt.legend()
plt.savefig(f"output/primal/phi_invg_plot_{N=}.pdf")
plt.close()

# The eigenvalue comparison

# computed eigenvalues
plt.plot([i for i in range(len(Gamma))], Gamma, 'or', markersize=5 , label = r"approx $\omega^2$")
# plot exact eigenvalues as dashed lines
for val in exact[:-1]:
    plt.axhline(val, linestyle="--", linewidth=0.8, color = 'k')
plt.axhline(exact[-1], linestyle="--", linewidth=0.8, color = 'k', label = r"true $\omega^2$") # for the legend
plt.xlabel("Eigenvalue Number")
plt.ylabel(r"$\omega^2$")
plt.title(rf"Comparing Exact and Computed $\omega^2$ ($N = {N}$)")
plt.legend()
plt.savefig(f"output/primal/omega_plot_{N=}.pdf")
plt.close()
