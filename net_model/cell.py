def heaviside(x):
    if x<=0: return False
    else: return True

class cell:
    def __init__(self, active, i, k, pre=[]):
        self.active = bool(active)
        self.pre = list(pre)
        #self.post = post
        self.i = i
        self.k = k

    def update(self, J, population, theta, ext):
        u = 0
        for j in range(len(self.pre)):
            u += J[self.k, population[self.pre[j]].k]*population[self.pre[j]].active
        u = u + ext[self.k] - theta[self.k]
        self.active = heaviside(u)

    def __repr__(self):
        return f'active={self.active}\nindex={self.i}\npopulation={self.k}\n{len(self.pre)} presynaptic cells'
