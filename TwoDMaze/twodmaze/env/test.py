#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sympy as sp

def main():
    # ----- symbols -----
    pr, ps, pd, qd = sp.symbols('p_r p_s p_d q_d', real=True)
    # We'll eliminate p_d using the simplex constraint later: p_d = 1 - p_r - p_s

    # ----- v_a(q_d) in your final reduced form -----
    # v_a = [ 0.5*pd + (1.5*pr + ps - 0.5)*qd ] / [ 1 + pr - ps + 0.5*pd + (0.5*pr + 3*ps - 1.5)*qd ]
    va = (sp.Rational(1,2)*pd + (sp.Rational(3,2)*pr + ps - sp.Rational(1,2))*qd) / \
         (1 + pr - ps + sp.Rational(1,2)*pd + (sp.Rational(1,2)*pr + 3*ps - sp.Rational(3,2))*qd)

    # Optional: eliminate pd via constraint p_r + p_s + p_d = 1
    subs_simplex = {pd: 1 - pr - ps}
    va_s = sp.simplify(va.subs(subs_simplex))

    # ----- derivatives wrt q_d -----
    d1 = sp.simplify(sp.diff(va_s, qd))
    d2 = sp.simplify(sp.diff(va_s, qd, 2))

    # Extract the clean “numerator over denominator^2” form for the first derivative:
    # d1 = (num1) / (den1)
    num1 = sp.simplify(sp.together(d1).as_numer_denom()[0])
    den1 = sp.simplify(sp.together(d1).as_numer_denom()[1])

    # It’s often useful to factor them:
    num1_f = sp.factor(num1)
    den1_f = sp.factor(den1)

    # Also show the “affine denominator” of v_a, for curvature sign:
    # v_a = (A + B*qd) / (C + D*qd)  => D = 0.5*pr + 3*ps - 1.5 (after substituting pd)
    # Let’s extract A,B,C,D directly to confirm:
    # Build A,B,C,D from the pre-substitution form, then substitute
    A = sp.Rational(1,2)*pd
    B = sp.Rational(3,2)*pr + ps - sp.Rational(1,2)
    C = 1 + pr - ps + sp.Rational(1,2)*pd
    D = sp.Rational(1,2)*pr + 3*ps - sp.Rational(3,2)

    A_s = sp.simplify(A.subs(subs_simplex))
    B_s = sp.simplify(B.subs(subs_simplex))
    C_s = sp.simplify(C.subs(subs_simplex))
    D_s = sp.simplify(D.subs(subs_simplex))

    # Pretty print results
    sp.init_printing(use_unicode=True)

    print("\n=== v_a(q_d) with p_d eliminated (p_d = 1 - p_r - p_s) ===")
    sp.pprint(va_s)

    print("\n=== dv_a/dq_d (simplified) ===")
    sp.pprint(d1)

    print("\n--- numerator of dv_a/dq_d (factored) ---")
    sp.pprint(num1_f)

    print("\n--- denominator of dv_a/dq_d (factored) ---")
    sp.pprint(den1_f)

    print("\n=== d^2 v_a / dq_d^2 (simplified) ===")
    sp.pprint(d2)

    print("\n=== Affine coefficients A,B,C,D in v_a = (A + B q_d) / (C + D q_d) (after substitution) ===")
    print("A = "); sp.pprint(A_s)
    print("B = "); sp.pprint(B_s)
    print("C = "); sp.pprint(C_s)
    print("D = "); sp.pprint(D_s)

    # Optional: verify the closed-form slope claimed earlier:
    # Expected: dv/dq = [4*pr*(1+pr)] / [ ( pr - 3*ps + 3 + (pr + 6*ps - 3)*qd )^2 ]
    # Build the RHS and check symbolic equality (up to simplification).
    rhs = (4*pr*(1+pr)) / ( (pr - 3*ps + 3) + (pr + 6*ps - 3)*qd )**2
    check_eq = sp.simplify(d1 - rhs)
    print("\n=== Check dv/dq vs claimed closed form (should simplify to 0) ===")
    sp.pprint(check_eq)

    # Small numeric probe (kept symbolic overall, but useful to see numbers):
    print("\n=== Numeric sanity check at pr=0.3, ps=0.4 (so pd=0.3), qd=0.2 ===")
    probe_vals = {pr: 0.3, ps: 0.4, qd: 0.2}
    print("v_a =", float(va_s.subs(probe_vals)))
    print("dv/dq =", float(d1.subs(probe_vals)))
    print("d2v/dq2 =", float(d2.subs(probe_vals)))
    print("D =", float(D_s.subs(probe_vals)), "(sign of D controls concavity: concave if D>0, convex if D<0)")

if __name__ == "__main__":
    main()
