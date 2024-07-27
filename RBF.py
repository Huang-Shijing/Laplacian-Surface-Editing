import numpy as np

#rn为距离，r0为紧支半径，basis为基函数类型
#理论基础见《计算流体力学网格生成技术》p243p244
def RBF_func(rn, r0, basis):
    #全域型径向基函数
    if basis == 11:          # Volume Spline
        fai = rn
    elif basis == 12:       # Thin Plate Spline
        fai = rn**2 * np.log10(rn)
    elif basis == 13:       # Multi-Quadric
        c = 1e-4
        fai = np.sqrt(c**2 + rn**2)
    elif basis == 14:       # Inverse Multi-Quadric
        c = 1e-4
        fai = np.sqrt(c**2 + rn**2)
        fai = 1.0 / fai
    elif basis == 15:       # Inverse Quadric
        fai = 1.0 / (1 + rn**2)
    else:                  #紧支型径向基函数
        ksi = rn / r0
        if ksi >= 1:
            fai = 0.0
        else:
            if basis == 21:      # Wendland's C0
                fai = ( 1 - ksi )**2
            elif basis == 22:   # Wendland's C2
                fai = ( 1 - ksi )**4 * ( 4.0 * ksi + 1 )
            elif basis == 23:   # Wendland's C4
                fai = ( 1 - ksi )**6 * ( 35.0 * ksi**2 + 18.0 * ksi + 3 )
            elif basis == 24:   # Wendland's C6
                fai = ( 1 - ksi )**8 * ( 32.0 * ksi**3 + 25.0 * ksi**2 + 8.0 * ksi + 1 )
            elif basis == 31:   # compact TPS C0
                fai = ( 1 - ksi )**5
            elif basis == 32:   # compact TPS C1
                fai = 3 + 80 * ksi**2 - 120 * ksi**3 + 45 * ksi**4 - 8 * ksi**5 + 60 * ksi**2 * np.log(ksi)
            elif basis == 33:   # compact TPS C2a
                fai = 1 - 30 * ksi**2 - 10 * ksi**3 + 45 * ksi**4 - 6 * ksi**5 - 60 * ksi**3 * np.log(ksi)
            elif basis == 34:   # compact TPS C2b
                fai = 1 - 20 * ksi**2 + 80 * ksi**3 - 45 * ksi**4 - 16 * ksi**5 + 60 * ksi**4 * np.log(ksi)
    
    return fai