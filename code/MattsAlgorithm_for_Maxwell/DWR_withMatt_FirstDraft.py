from firedrake import *
import numpy as np
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
import os




# EIGENPROBLEM SETUP (PRIMAL AND ENRICHED)
######################################################################################




# Set a triangular mesh like in Matt's example (Fig 3.22)
N = 32
mesh = SquareMesh(N, N, pi, quadrilateral=False)



# Decide if you want edge elements or Lagrange elements to discretize the space for E
hcurl = False
if hcurl:
    V0 = FunctionSpace(mesh, "N1curl", 2)
    V0p = FunctionSpace(mesh, "N1curl", 3) # enriched space
else:
    V0 = VectorFunctionSpace(mesh, "CG", 1, dim=2)
    V0p = VectorFunctionSpace(mesh, "CG", 2, dim=2) # enriched space

# The space for H is CG1
V1 = FunctionSpace(mesh, "CG", 1)
V1p = FunctionSpace(mesh, "CG", 2) # enriched space

# We need a mixed function space here: (space for E) x (space for H)
Z = MixedFunctionSpace([V0, V1])
Zp = MixedFunctionSpace([V0p, V1p]) # enriched space

# Set zero tangential trace conditions for E:
# Edge elements only force continuity in tangential component so we just set zero conditions on boundary.
# If using Lagrange elements, we need so explicily set the tangential trace to zero.
if hcurl:
    bc = [DirichletBC(Z.sub(0), Constant((0, 0)), "on_boundary")]
    bcp = [DirichletBC(Zp.sub(0), Constant((0, 0)), "on_boundary")]
else:
    bc = [DirichletBC(Z.sub(0).sub(0), 0, (3, 4)),
          DirichletBC(Z.sub(0).sub(1), 0, (1, 2))]
    bcp = [DirichletBC(Zp.sub(0).sub(0), 0, (3, 4)),
          DirichletBC(Zp.sub(0).sub(1), 0, (1, 2))]


# Scalar rot
def rot_s(E):
    return E[1].dx(0) - E[0].dx(1)

# Vector rot
def rot_v(H):
    return as_vector([H.dx(1), -H.dx(0)])

# Set the complex unit as j
j = Constant(1j)






# SET THE PRIMAL EIGENPROBLEM AND SOLVER
######################################################################################




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
primal_problem = LinearEigenproblem(a, b, bcs=bc, restrict=True)

# Solver parameters and subsequent solver
sp = {"eps_gen_hermitian": None,  # solver parameters, passed to SLEPc
      "eps_type": "krylovschur",
      "eps_tol": 1e-8,
      "eps_monitor": None,
      "eps_smallest_magnitude": None,
      #"eps_view": None,  # uncomment to see the solver
      "eps_target": 0,
      "eps_target_real": None,
      "st_type": "sinvert",
      }
primal_solver = LinearEigensolver(primal_problem, n_evals=1, solver_parameters=sp)







# SET THE ENRICHED EIGENPROBLEM AND SOLVER
######################################################################################





# Test and trial functions
up = TrialFunction(Zp)
(Ep, Hp) = split(up)
vp = TestFunction(Zp)
(Fp, Gp) = split(vp)

# Set Ln(z) = <(A-zI)u, (A-zI)v>
ap = (
      # A^* M^{-1} A terms
      inner(rot_s(Ep), rot_s(Fp))*dx
    + inner(rot_v(Hp), rot_v(Gp))*dx
      # -2 z A terms
    - 2 * z * inner(j*rot_v(Hp), Fp)*dx
    + 2 * z * inner(j*rot_s(Ep), Gp)*dx
      # |z|^2 M terms
    + conj(z) * z * inner(Ep, Fp)*dx
    + conj(z) * z * inner(Hp, Gp)*dx
    )

# Set Gn = <u, v>
bp = inner(Ep, Fp)*dx + inner(Hp, Gp)*dx


# Set the generalized eigenproblem Ln(z)x = lambda Gn x on restructed space
enriched_problem = LinearEigenproblem(ap, bp, bcs=bcp, restrict=True)


# solver
enriched_solver = LinearEigensolver(enriched_problem, n_evals=1, solver_parameters=sp)






# COMPUTE PHI AT EACH POINT IN GRID(n). ALSO COMPUTE THE DWR ERROR ESTIMATE
######################################################################################





# Set Grid(n) and empty lists for phi and error values
h = 0.02
n = 1/h
grid = np.append(np.arange(0.01, 4, h),[4]) # Patrick makes this a list
phi_vals = []
DWR_phi_vals = []
DWR_errors = []





print(GREEN % f'Step 1 - Compute Phi and Invg at Each Grid Point')
print(GREEN % f'-'*100)
print()
print()



# Iterate over z grid
for curr_z in grid:


    z.assign(curr_z) # set z as current grid point

    print(BLUE % f"Primal Solve:")


    # Solve the primal problem!!


    primal_nconv = primal_solver.solve() # solver the eigenproblem
    min_eigval = primal_solver.eigenvalue(0) # the first (smallest) eigenvalue is what we desire
    uh = primal_solver.eigenfunction(0)[0] # pull the corresponding eigenfunction

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


    
    # Compute Phi
    phi = sqrt(min_eigval) # Our bound on distance to the spectrum - Phi_n = sqrt{lambda_min} >= dist(z, Sp(A))
    phi_vals.append(phi)


    # Normalize
    Euh, Huh = split(uh)
    m_val_uh = assemble((inner(Euh, Euh) + inner(Huh, Huh))*dx)
    if abs(m_val_uh) < 1e-12:
      print(RED % f"WARNING - m(u, u) is really small!!")
    uh.assign(uh / sqrt(m_val_uh))



    # Solve the enriched problem !!
    

    print(BLUE % f"Enriched Solve:")

    # lift the coarse solution to the enriched space as an inital guess
    uh_lift = Function(Zp)
    if hcurl:
      uh_lift.project(uh, bcs=bcp)
    else:
      uh_lift.interpolate(uh, Zp)

    # Set it as initial eigenspace guess
    with uh_lift.dat.vec_ro as v:
      enriched_solver.es.setInitialSpace([v.copy()])


    # Solve enriched problem
    enriched_nconv = enriched_solver.solve() # solve the eigenproblem
    uph = enriched_solver.eigenfunction(0)[0]


    # Normalize 
    Eup, Hup = split(uph)
    m_val_up = assemble((inner(Eup, Eup) + inner(Hup, Hup))*dx)
    if abs(m_val_up) < 1e-12:
      print(RED % f"WARNING - m(u, u) is really small!!")
    uph.assign(uph / sqrt(m_val_up))


    # Attempt to align the signs of the eigenfunctions
    Eh, Hh = split(uh_lift)
    Eup, Hup = split(uph)
    sign_checker = assemble((inner(Eh, Eup) + inner(Hh, Hup))*dx)
    if sign_checker < 0:
      uph.assign(-uph)


    # Error estimate !!
  

    # Interpolate down to Z space
    Ih_up = Function(Z)
    if hcurl:
      Ih_up.project(uph, bcs=bc)
    else:
      Ih_up.interpolate(uph)

    # Lift the interpolant back to Zp
    IhIh_up = Function(Zp)
    if hcurl:
      IhIh_up.project(Ih_up, bcs=bcp)
    else:
      IhIh_up.interpolate(Ih_up)
    
    # Form wp
    wp = Function(Zp).assign(uph -  IhIh_up)

    # Compute the primal residual
    Eh, Hh = split(uh_lift)
    Eph, Hph = split(wp)

    a_primal = (
          # A^* M^{-1} A terms
          inner(rot_s(Eh), rot_s(Eph))*dx
        + inner(rot_v(Hh), rot_v(Hph))*dx
          # -2 z A terms
        - 2 * z * inner(j*rot_v(Hh), Eph)*dx
        + 2 * z * inner(j*rot_s(Eh), Hph)*dx
          # |z|^2 M terms
        + conj(z) * z * inner(Eh, Eph)*dx
        + conj(z) * z * inner(Hh, Hph)*dx
        )
    
    
    b_primal = inner(Eh, Eph)*dx + inner(Hh, Hph)*dx

    rho = a_primal - min_eigval * b_primal


    # assemble 
    eig_error = assemble(rho)

    # Check if really small
    if abs(np.imag(eig_error)) > 0.0:
      print(RED % f"WARNING - computed error has nontrivial imaginary part: {eig_error}")
      print(RED % f"We will take its real part")
      eig_error = np.real(eig_error)


    # Use this to make phi better
    improved_eig = min_eigval + eig_error
    DWR_phi = sqrt(improved_eig)
    DWR_phi_vals.append(DWR_phi)
    DWR_errors.append(eig_error)


    # Track the results
    print(BLUE % f"z = {curr_z}: |dist(z, spectrum)| <= {phi}") # Patrick's output choice
    print(BLUE % f"The error estimate is: {eig_error}")
    print(BLUE % f"The improved eigenvalue is: {improved_eig}")
    print()


    # Set initial spaces for the next solves
    primal_eigenfunctions = [primal_solver.eigenfunction(i)[0] for i in range(primal_nconv)] # pull out the converged eigenfunctions
    primal_space = []
    for eigenfunction in primal_eigenfunctions:
        with eigenfunction.dat.vec_ro as v: # take the underlying PETSc object (read only)
            primal_space.append(v.copy()) # copy the PETSc object into the eigenspace
    primal_solver.es.setInitialSpace(primal_space) # set the eigenspace as the initial guess for the next iter (next grid point)




# PLOTTING ROUTINE
######################################################################################





# desired eigenvalues
exact = np.array([n**2 + m**2 for n in range(10) for m in range(10) if n**2 + m**2 <= 4**2])

# phi and invg
plt.plot(grid, phi_vals, linewidth=2, label = r"$\Phi_n(z, A)$")
plt.plot(np.sqrt(exact), 0*exact, 'ok', markersize=5, label = r'exact $\omega^2$')
plt.plot(grid, DWR_phi_vals, label = r"DWR corrected $\Phi_n(z, A)$")
os.makedirs("output/mixed/", exist_ok=True)
plt.xlabel(r"$x$")
plt.title(rf"Approximations of $\Phi_n(z, A)$ and $\omega$ ($N = {N}$)")
plt.legend()
plt.savefig(f"output/mixed/phi_invg_plot_{N=}.pdf")
plt.close()

