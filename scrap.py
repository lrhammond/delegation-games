
def normalise(v):

    v_centred = v - v.mean()
    return(v_centred / norm(v_centred))

def new_test(rng, dims,x):

    n = len(dims)
    s = np.prod(dims)

    u = np.stack([normalise(rng.normal(size=dims)) for _ in dims])
    u_flat = np.reshape(u, n*s)

    def d(v):
        return norm(v - u_flat)
    # d = lambda v : norm(v - u_flat)

    def m(i):
        return lambda v : norm(v[i * s:(i+1) * s])

    def ca(v):

        mu = sum([v[i * s:(i+1) * s] for i in range(n)]) / n
        return sum([norm(mu - v[i * s:(i+1) * s]) for i in range (n)]) / (2*n)

    constraints = [NonlinearConstraint(m(i), 1, 1) for i in range(n)]
    constraints += [NonlinearConstraint(ca, x, x)]

    x_0 = []
    for _ in range(n):
        j = rng.randint(0,s)
        x_0 += [1 if i == j else 0 for i in range(s)]
    # x_0 = np.array(x_0)
    # mu = sum([x_0[i * s:(i+1) * s] for i in range(n)]) / n
    # x_0 = x_0 - np.sum([x_0[i * s:(i+1) * s] for i in range(n)]) / n])

    res = minimize(d, np.array(x_0), constraints = constraints)
    v = np.reshape(res.x, [n] + list(dims))

    y = sum([norm(np.sum(v,axis=0)/n - v[i]) for i in range(n)]) / (2*n)
    print(y)

    for i in range(n):
        print(norm(v[i]))

    return

# CA = 1.00

def test(rng, n, dim):

    vectors = [normalise(rng.normal(size=dim)) for _ in range(n)]
    mu = sum(vectors)
    vectors = [normalise(v - CA*mu) for v in vectors]

    mu = sum(vectors) / n
    mu_m = norm(mu)

    bases = [geom.orthogonal_project_planar(mu, v) for v in vectors]
    thetas = [np.arccos(np.dot(mu,v) / mu_m) for v in vectors]

    d = sum([norm(mu - v) for v in vectors]) / n
    j = 0

    while np.abs(d-CA) > 0.01 and j < 50:
        
        thetas = [min(CA/d * t, np.pi / 2) for t in thetas]
        vectors = [bases[i][0] * np.cos(thetas[i]) + bases[i][1] * np.sin(thetas[i]) for i in range(len(vectors))]
        mu = sum(vectors) / n
        d = sum([norm(mu - v) for v in vectors]) / n
        print(d)
        j += 1
    
    return