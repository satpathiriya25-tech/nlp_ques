# a = [1,2,3,4]
# print(a.append([4,5,6,7]))
# print(a)
# a.extend([4,5,6,7])
# print(a)

def remove_puncts(s):
    for p in ["-", "--", "---", ".", "?", ",", "'"]:
        s = s.replace(p, " ")

    return s

if __name__ == "__main__":
    a = "ahjljl- jhoj, hkh;"
    print(remove_puncts(a))