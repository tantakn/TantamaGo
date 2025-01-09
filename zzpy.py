a = 1234
b = a
a = 0
print(a)
print(b)

s = "abc"
t = s
s = "def"
print(s)
print(t)

v = [1,2,3,4,5,6,7,8,9]
w = v
u = v.copy()
v[1] = 0
print(v)
print(w)
print(u)

m = {"a":1, "b":2}
n = m
l = m.copy()
m["a"] = 0
print(m)
print(n)
print(l)