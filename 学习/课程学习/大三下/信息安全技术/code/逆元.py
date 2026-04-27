def exgcd(a,b):
    # 扩展欧几里得算法：返回 (g, x, y)，满足 ax + by = g = gcd(a, b)
    if b == 0:
        return(a,1,0)
    
    g,x1,y1 = exgcd(b,a%b)
    # 由子问题的解反推出当前层的系数
    x = y1
    y = x1 - (a//b)*y1
    return (g,x,y)

def mod_inverse(a,m):
    # 先把 a 约化到模 m 的范围内，再求 ax + my = gcd(a, m)
    g,x,y = exgcd(a%m,m)
    # 只有 gcd(a, m) = 1 时，a 在模 m 下才存在逆元
    if g!=1:
        return None

    # 把可能为负的 x 规范到 [0, m-1] 区间
    return (x%m+m) %m

def printInv(a,m,inv):
    if(inv == None):
        # 不互素时无逆元
        print(f"{a}没有关于模{m}的逆元")
        print()
    else:
        # 输出求得的逆元
        print(f"{a}关于模{m}的逆元:{inv}")
        print()

if __name__ == "__main__":
    # 61 与 105 互素，应存在逆元
    a1,m1 = 61,105
    inv1 = mod_inverse(a1,m1)
    printInv(a1,m1,inv1)

    # 31 与 105 互素，应存在逆元
    a2,m2 = 31,105
    inv2 = mod_inverse(a2,m2)
    printInv(a2,m2,inv2)


    # 87 与 255 不互素，不存在逆元
    a3,m3 = 87,255
    inv3 = mod_inverse(a3,m3)
    printInv(a3,m3,inv3)

    # 109 与 255 互素，应存在逆元
    a4,m4 = 109,255
    inv4 = mod_inverse(a4,m4)
    printInv(a4,m4,inv4)
