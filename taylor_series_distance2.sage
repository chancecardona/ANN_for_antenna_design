var('x y z')
#r = var('r', domain='positive')
#a = var('a', domain='positive')

#r_def = r == sqrt(x^2+y^2+z^2)


# Doing this at a=1 since it doesn't come into play here. Remember to multiply back in otherwise it's normalized
# relative to a.
x_expr   = x^2 + y^2 + z^2 + 2*x
x_expr_n = x^2 + y^2 + z^2 - 2*x
y_expr   = x^2 + y^2 + z^2 + 2*y
y_expr_n = x^2 + y^2 + z^2 - 2*y
z_expr   = x^2 + y^2 + z^2 + 2*z
z_expr_n = x^2 + y^2 + z^2 - 2*z

f(x,y,z) = sum((1 / sqrt(1 + expr)) for expr in [x_expr, x_expr_n, y_expr, y_expr_n, z_expr, z_expr_n])
# CAREFUL here.
# I also tried .series and .taylor(x, 0, 4).taylor(...), but these did not simplify because of the missing cross product terms
# not considered by expanding to each order individually. 
# I also tried to do the taylor series for 1/sqrt(1+u) for u, then substituting back for each expr, but it didn't yield correct results, likely due to the analytical/numerical issues behind doing that theory-wise.
total_series = f.taylor((x, 0), (y, 0), (z, 0), 6)

print("Before filtering higher order terms:")
print(total_series.polynomial(QQ)) 

filtered_series = sum(t for t in total_series.iterator() if (t.degree(x) + t.degree(y) + t.degree(z)) < 6)
print("After:")
print(filtered_series.polynomial(QQ))
