import numpy as np

def dfs(params, w):
    dfsHelper(params, 0, 1, w)

def dfsHelper(params, depth, wk, w):

    if(depth == len(params)):
        w.append(wk)
        return

    for r in range(0, len(params[depth])):
        for c in range(0, len(params[depth][r])):
            wkC = wk*params[depth][r][c]
            dfsHelper(params, depth+1, wkC, w)

if __name__ == '__main__':
    a1 = np.array([[1, 2], [4, 5]])

    a2 = np.array([[10, 11, 12], [12, 12, 13]])

    d = []
    d.append(a1)
    d.append(a2)

    for e in d:
        print(e.shape)

    w = []
    dfs(d, w)
    print(w)


