
def produce(string, char_idx, p):
    new_string = string 
    if char_idx in "0123456789":
        if p == 0:
            p = int(char_idx)
        else:
            p = p * 10 + int(char_idx)
    else:
        move_ = p
        if p < len(string):
            move_ = p % len(string)
        new_string = string[move_:] + string[: move_]
        p = 0
        if char_idx == "R":
            new_string = new_string[::-1]
        else:
            new_string = new_string + char_idx
    return new_string, p
        

if __name__ == "__main__":
    # # N = int(input())
    # # # print(N)
    # # string = input()
    # n = int(input())
    # s = input()
    # result = []
    # for i in range(n):
    #     if i < 2:
    #         result.append("0")
    #     temp = int(result[-1])
    #     for j in range(i - 1):
    #         if s[i] == s[j]:
    #             temp += (i - j - 1)
    #     result.append(str(temp))
    # # print(result)
    # print(" ".join(result))
    
    n = int(input())
    a = []
    p = 0
    for i in range(n):
        a.append(input().strip())
    for string in a:
        p = 0
        t = ""
        for subs in string:
            t, p = produce(t, subs, p)
        print(t)
        
        