def find_kth_smallest(a, b, k):
    # assert len(a) + len(b) >= k, "Invalid k"
    left, right = 0, min(k, len(a))-1
    j = -1
    while left <= right:
        mid = int((left + right)/2)
        q = k-(mid+1)
        if q >= len(b) or b[q] >= a[mid]:
            j = mid
            left = mid+1
        else:
            right = mid-1

    return j, k-(j+1)-1



def find_median(a, b):
    n = len(a) + len(b)
    k = int(n/2)
    i, j = find_kth_smallest(a, b, k+1)

    if n % 2 == 1:
        x = a[i] if i >= 0 else -float("Inf")
        y = b[j] if j >= 0 else -float("Inf")

        return max(x, y)
    else:
        if i < 0:
            return (b[j] + b[j-1])/2.0
        elif j < 0:
            return (a[i] + a[i-1])/2.0
        else:
            if a[i] < b[j]:
                x = b[j]
                y = max(a[i], b[j-1]) if j > 0 else a[i]
            else:
                x = a[i]
                y = max(b[j], a[i-1]) if i > 0 else b[j]
            
            return (x + y)/2.0

a = [2, 4, 8, 11, 14, 15, 19]
b = [1, 3, 6, 13, 20, 22, 25]

# 1 2 3 4 6 8 11 13 14 15 19 20 22 25

# c = [1,1,2,3,4,6,7,8,8,10,11,12,15,17,26]

print(find_kth_smallest(a, b, 20))