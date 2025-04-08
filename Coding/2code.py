
if __name__ == "__main__":
    
    n = int(input())
    arr = list(map(int, input().strip().split()))
    # print(arr)
    record = dict()
    for i in range(n - 1):
        for j in range(i + 1, n):
            if arr[i] * arr[j] not in record:
                record[arr[i] * arr[j]] = [i, j]
    found = False
    # print(record)
    for i in range(n):
        if arr[i] in record and not found:
            print(f"{record[arr[i]][0]} {record[arr[i]][1]} {i}")
            found = True 
            
    if not found:
        print("-1")
    