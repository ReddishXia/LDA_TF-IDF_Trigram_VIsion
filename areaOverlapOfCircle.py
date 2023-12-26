import math
eps = 0.0000000001
class Circle:

    def __init__(self, x=0, y=0, r=0, angle=0, d=0):
        self.x = x
        self.y = y
        self.r = r
        self.angle = angle
        self.d = d

N = 1010
area = [0 for _ in range(N)]
cir = [Circle() for _ in range(N)]
tp = []
def dcmp(x):
    if x < -eps:
        return -1
    else:
        return x > eps

def dis(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def cross(p0, p1, p2):
    return (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x)

def CirCrossCir(p1, r1, p2, r2, mycp1, mycp2):
    mx = p2.x - p1.x
    sx = p2.x + p1.x
    mx2 = mx * mx
    my = p2.y - p1.y
    sy = p2.y + p1.y
    my2 = my * my
    sq = mx2 + my2
    d = -((sq - (r1 - r2)**2) * (sq - (r1 + r2)**2))
    if d + eps < 0:
        return 0
    if d < eps:
        d = 0
    else:
        d = math.sqrt(d)
    x = mx * ((r1 + r2) * (r1 - r2) + mx * sx) + sx * my2
    y = my * ((r1 + r2) * (r1 - r2) + my * sy) + sy * mx2
    dx = mx * d
    dy = my * d
    sq *= 2
    mycp1.x = (x - dy) / sq
    mycp1.y = (y + dx) / sq
    mycp2.x = (x + dy) / sq
    mycp2.y = (y - dx) / sq
    if d > eps:
        return 2
    else:
        return 1

def circmp(u, v):
    return dcmp(u.r - v.r) < 0

def cmp(u, v):
    if dcmp(u.angle - v.angle):
        return u.angle < v.angle
    return u.d > v.d

def calc(cir, cp1, cp2):
    ans = (cp2.angle - cp1.angle) * (cir.r)**2 - cross(cir, cp1, cp2) + cross(Circle(0,0), cp1, cp2)
    return ans / 2

def CirUnion(cir):

    cir.sort(key=lambda x: x.r)
    for i in range(len(cir)):
        for j in range(i + 1, len(cir)):
            if dcmp(dis(cir[i], cir[j]) + cir[i].r - cir[j].r) <= 0:
                cir[i].d += 1
    for i in range(len(cir)):
        tn = 0
        cnt = 0
        for j in range(len(cir)):
            cp1 = Circle()
            cp2 = Circle()
            if i == j:
                continue
            if CirCrossCir(cir[i], cir[i].r, cir[j], cir[j].r, cp2, cp1) < 2:
                continue
            cp1.angle = math.atan2(cp1.y - cir[i].y, cp1.x - cir[i].x)
            cp2.angle = math.atan2(cp2.y - cir[i].y, cp2.x - cir[i].x)
            cp1.d = 1
            tp.append(cp1)
            tn += 1
            cp2.d = -1
            tp.append(cp2)
            tn += 1
            if dcmp(cp1.angle - cp2.angle) > 0:
                cnt += 1
        tp.append(Circle(cir[i].x - cir[i].r, cir[i].y,0, math.pi, -cnt))
        tn += 1
        tp.append(Circle(cir[i].x - cir[i].r, cir[i].y,0, -math.pi, cnt))
        tn += 1
        tp.sort(key=lambda x: (x.angle, -x.d))
        p = cir[i].d + tp[0].d
        s = cir[i].d + tp[0].d
        for j in range(1, tn-1):
            p = s
            s += tp[j].d
            area[p] += calc(cir[i], tp[j - 1], tp[j])
    i=0

def solve():
    global n
    n = int(input())
    cir = [Circle() for _ in range(n)]
    for i in range(n):
        x, y, r = map(float, input().split())
        cir[i] = Circle(x, y, r)
        cir[i].d = 1

    CirUnion(cir)
    for i in range(1, n + 1):
        area[i] -= area[i + 1]
    tot = sum(area[1:])
    print(area[n])

if __name__ == "__main__":
    solve()
